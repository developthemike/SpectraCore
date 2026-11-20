#include "audio/listener.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "audio/limiter.hpp"
#include "audio/rate_control.hpp"
#include "audio/voice_playout.hpp"
#include "dsp/ambe_vocoder.hpp"
#include "util/thread_util.hpp"

namespace spectro {

namespace {

constexpr double kTapSeconds = 0.1;      // slack between the receive thread and the audio thread
constexpr size_t kMaxBlock   = 1 << 16;  // IQ samples demodulated per pass

constexpr float kGainSmoothing = 0.002f;      // volume changes take about 10 ms
constexpr float kSpeechSmoothing = 1.0f / 4800.0f;  // the level of decoded speech: over about 0.1 s
constexpr size_t kMaxSystems = 8;  // P25 systems remembered

// Search for the carrier: let the widened filter settle, then average the
// instantaneous frequency.  Broadcast audio holds more low-frequency energy,
// which takes longer to average out.
constexpr double kSearchSettleSeconds = 0.01;
constexpr double kSearchNarrowSeconds = 0.06;
constexpr double kSearchWideSeconds   = 0.2;
constexpr double kSearchWiden         = 2.5;  // filter widened by this much times the search range

size_t tap_capacity(double sample_rate)
{
    const double samples = std::clamp(sample_rate * kTapSeconds, 32768.0, 8388608.0);
    return std::bit_ceil(static_cast<size_t>(samples));
}

}  // namespace

Listener::Listener(IqSource& source, Acquisition& acquisition)
    : source_(source),
      acquisition_(acquisition),
      sample_rate_(source.sample_rate()),
      available_(FmDemodulator::supports(source.sample_rate(), kAudioRate, FmMode::Narrow)),
      ring_(available_ ? tap_capacity(source.sample_rate()) : 1024),
      fifo_(static_cast<size_t>(kAudioRate))  // one second
{
}

Listener::~Listener()
{
    stop();
}

void Listener::start()
{
    if (!available_ || thread_.joinable())
        return;
    stop_.store(false);
    stopped_ = false;
    thread_  = std::thread([this] { run(); });
}

void Listener::stop()
{
    if (stopped_)
        return;
    stopped_ = true;
    enabled_.store(false);
    fifo_.set_flowing(false);
    acquisition_.set_tap(nullptr);  // from here on nothing writes to the ring
    stop_.store(true, std::memory_order_release);
    ring_.close();
    if (thread_.joinable())
        thread_.join();
}

template <typename Change>
void Listener::change(Change&& change)
{
    {
        std::lock_guard<std::mutex> lock(params_mutex_);
        change(params_);
    }
    generation_.fetch_add(1, std::memory_order_release);
}

void Listener::set_enabled(bool enabled)
{
    if (!available_ || stopped_)
        return;
    bool changed = false;
    change([&](Params& p) {
        if (p.enabled == enabled)
            return;
        p.enabled = enabled;
        if (enabled)
            ++p.session;
        changed = true;
    });
    if (!changed)
        return;
    enabled_.store(enabled);
    if (enabled)
        fifo_.discard();  // whatever is left from the previous session
    else
        fifo_.set_flowing(false);
    // The audio thread sleeps until samples arrive, so this switches it on and off.
    acquisition_.set_tap(enabled ? this : nullptr);
}

void Listener::tune(double frequency_hz, double search_hz, double bandwidth_hz)
{
    forget_signal();
    change([&](Params& p) {
        p.frequency = frequency_hz;
        p.search    = std::max(0.0, search_hz);
        if (bandwidth_hz > 0.0)
            p.bandwidth = clamp_bandwidth(bandwidth_hz);
        ++p.tune_serial;
    });
}

// Between the two ranges, the nearer one.
double Listener::clamp_bandwidth(double hz)
{
    return hz < 0.5 * (kMaxNarrowBandwidth + kMinWideBandwidth) ? std::clamp(hz, kMinNarrowBandwidth, kMaxNarrowBandwidth)
                                                                : std::clamp(hz, kMinWideBandwidth, kMaxWideBandwidth);
}

void Listener::set_bandwidth(double hz)
{
    hz = clamp_bandwidth(hz);
    if (FmDemodulator::mode_for(hz) != status().mode)
        forget_signal();  // another kind of channel
    change([&](Params& p) { p.bandwidth = hz; });
}

// What the channel carried no longer counts: it is another channel now.
void Listener::forget_signal()
{
    std::lock_guard<std::mutex> lock(signal_mutex_);
    signal_ = {};
    dmr_    = {};
    p25_    = {};
    p25p2_  = {};
}

void Listener::set_muted(bool muted)
{
    change([&](Params& p) { p.muted = muted; });
}

void Listener::set_dmr_only(bool dmr_only)
{
    change([&](Params& p) { p.dmr_only = dmr_only; });
}

void Listener::set_p25_only(bool p25_only)
{
    change([&](Params& p) { p.p25_only = p25_only; });
}

void Listener::set_p25_phase2_slot(int slot)
{
    change([&](Params& p) { p.p25p2_slot = slot == 0 || slot == 1 ? slot : -1; });
}

double Listener::scan_frequency() const
{
    const ListenStatus s = status();
    return s.searching ? std::numeric_limits<double>::quiet_NaN() : s.frequency;
}

ScanContent Listener::scan_content() const
{
    std::lock_guard<std::mutex> lock(signal_mutex_);
    ScanContent c;
    // A call counts from its header, before its voice.
    c.dmr = dmr_.present;
    for (const DmrCall& call : dmr_.slot)
        c.dmr_voice |= dmr_.present && call.active && !call.encrypted;
    c.p25       = p25_.present || p25p2_.present;
    c.p25_voice = p25_.present && p25_.call.active && !p25_.call.encrypted;
    for (const P25Call& call : p25p2_.slot)
        c.p25_voice |= p25p2_.present && call.active && !call.encrypted;
    return c;
}

void Listener::set_deemphasis(double seconds)
{
    change([&](Params& p) { p.deemphasis = seconds; });
}

void Listener::set_squelch(float threshold_dbfs)
{
    change([&](Params& p) { p.squelch = threshold_dbfs; });
}

void Listener::set_volume(float volume)
{
    change([&](Params& p) { p.volume = std::clamp(volume, 0.0f, 1.0f); });
}

void Listener::set_boost(float db)
{
    const float factor = std::pow(10.0f, std::clamp(db, 0.0f, kMaxBoostDb) / 20.0f);
    change([&](Params& p) { p.boost = factor; });
}

ListenStatus Listener::status() const
{
    ListenStatus s;
    {
        std::lock_guard<std::mutex> lock(params_mutex_);
        s.enabled   = params_.enabled;
        s.frequency = params_.frequency;
        s.bandwidth = usable_bandwidth(params_.bandwidth);
        s.dmr_only  = params_.dmr_only;
        s.p25_only  = params_.p25_only;
        s.p25p2_slot = params_.p25p2_slot;
        // Until the audio thread has taken over the latest tune() - and found
        // the carrier, if asked to - the frequency is not final.
        s.searching = s.enabled && st_settled_serial_.load(std::memory_order_acquire) != params_.tune_serial;
    }
    s.mode          = FmDemodulator::mode_for(s.bandwidth);
    s.in_band       = st_in_band_.load(std::memory_order_relaxed);
    s.squelch_open  = st_squelch_open_.load(std::memory_order_relaxed);
    s.level_dbfs    = st_level_.load(std::memory_order_relaxed);
    s.audio_rms     = st_audio_rms_.load(std::memory_order_relaxed);
    s.load          = st_load_.load(std::memory_order_relaxed);
    s.rate_trim     = st_trim_.load(std::memory_order_relaxed);
    s.audio_samples = st_audio_samples_.load(std::memory_order_relaxed);
    s.overruns      = st_overruns_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(signal_mutex_);
        s.signal = signal_;
        s.dmr    = dmr_;
        s.p25    = p25_;
        s.p25p2  = p25p2_;
        s.p25_systems = p25_systems_;
    }
    s.vocoder       = AmbeDecoder::available();
    s.dropped_iq    = dropped_iq_.load(std::memory_order_relaxed);
    s.underruns     = fifo_.underruns();
    return s;
}

// Receive thread: hand the block to the audio thread, or count it as dropped
// if that thread has fallen so far behind that its ring is full.
void Listener::on_samples(const int16_t* iq, size_t samples)
{
    while (samples > 0) {
        const IqRing::WriteSpan span = ring_.write_span(samples);
        if (span.samples == 0) {
            dropped_iq_.fetch_add(samples, std::memory_order_relaxed);
            return;
        }
        std::memcpy(span.data, iq, span.samples * 2 * sizeof(int16_t));
        ring_.commit(span.samples);
        iq += 2 * span.samples;
        samples -= span.samples;
    }
}

void Listener::run()
{
    set_current_thread_name("spectro-audio");

    FmDemodulator      demod(sample_rate_, kAudioRate);
    SignalClassifier   classifier(kAudioRate);
    FmDemodulator::Analysis analysis;
    uint64_t           published_windows = 0;
    SignalKind         published_kind    = SignalKind::Analysing;
    int                published_dmr     = 0;  // what decoded: 0 nothing, 1 DMR, 2 P25, 3 P25 Phase 2
    AudioRateControl   pacing(kAudioRate);
    PeakLimiter        limiter(kAudioRate);
    std::vector<float> audio;
    audio.reserve(16384);
    double             trim = 0.0;  // the rate correction in force for this block

    // Digital voice: DMR and P25, Phase 1 and 2
    DmrDecoder            dmr(kAudioRate);
    P25Decoder            p25(kAudioRate);
    P25Phase2Decoder      p25p2(kAudioRate);
    AmbeDecoder           ambe;
    AmbeDecoder           ambe_p25p2;
    ImbeDecoder           imbe;
    VoicePlayout          playout;
    std::vector<DmrVoice> dmr_voice;
    std::vector<P25Voice> p25_voice;
    std::vector<P25Phase2Voice> p25p2_voice;
    std::vector<p25p2::System>  systems;  // announced by the control channels heard, latest last
    float                 speech[AmbeDecoder::kFrameSamples];
    float                 speech_power = 0.0f;  // of what is played in place of the FM audio
    VoiceOvers            overs;                // the speech, an over at a time, for the voice sink
    double                signal_time = 0.0;    // seconds of signal taken from the ring
    const auto            forget_dmr = [&] {
        dmr.reset();
        p25.reset();
        p25p2.reset();
        playout.reset();
        overs.flush();
    };

    Params   p;
    uint64_t seen_generation = ~uint64_t{0};

    bool   was_in_band  = false;
    bool   searching    = false;
    size_t search_seen  = 0;    // audio samples since the search began
    double search_sum   = 0.0;
    size_t search_count = 0;
    float  gain         = 0.0f;

    auto   load_wall = std::chrono::steady_clock::now();
    double load_cpu  = current_thread_cpu_seconds();

    while (!stop_.load(std::memory_order_acquire) && ring_.wait_readable(1)) {
        // 1. Settings.
        const uint64_t generation = generation_.load(std::memory_order_acquire);
        if (generation != seen_generation) {
            seen_generation = generation;
            Params next;
            {
                std::lock_guard<std::mutex> lock(params_mutex_);
                next = params_;
            }
            if (next.session != p.session) {
                // Listening has just been switched on: nothing from before counts.
                demod.reset();
                pacing.reset();
                classifier.reset();
                forget_dmr();
                fifo_.discard();
                limiter.reset();
                trim      = 0.0;
                gain      = 0.0f;
                searching = false;
            }
            if (next.tune_serial != p.tune_serial) {
                classifier.reset();  // another signal
                forget_dmr();
                searching    = next.search > 0.0;
                search_seen  = 0;
                search_sum   = 0.0;
                search_count = 0;
            }
            demod.set_squelch(next.squelch);
            demod.set_deemphasis(next.deemphasis);
            p25p2.set_slot(next.p25p2_slot);
            p = next;
        }

        overs.set_sink(voice_sink_.load(std::memory_order_acquire));

        // 2. Demodulate what has arrived.  The channel is given as an absolute
        //    frequency, so retuning the radio moves it within the stream - or
        //    out of it, and then there is nothing to hear.
        const size_t           n     = std::min(ring_.readable(), kMaxBlock);
        const IqRing::ReadSpans spans = ring_.peek(n);
        signal_time += static_cast<double>(n) / sample_rate_;

        const double bandwidth = usable_bandwidth(p.bandwidth);
        const FmMode mode      = FmDemodulator::mode_for(bandwidth);
        const double offset    = p.frequency - source_.frequency();
        const bool   in_band   = std::fabs(offset) + 0.5 * bandwidth <= 0.5 * sample_rate_;
        audio.clear();
        if (in_band) {
            const double width = searching ? bandwidth + kSearchWiden * p.search : bandwidth;
            demod.set_channel(offset, width, mode);
            // While looking for the carrier the channel is not yet what it will be.
            FmDemodulator::Analysis* look = searching ? nullptr : &analysis;
            demod.process(spans.first, spans.first_count, audio, look);
            if (spans.second_count > 0)
                demod.process(spans.second, spans.second_count, audio, look);
            classifier.set_channel(mode, bandwidth);
            classifier.process(analysis.frequency.data(), analysis.power.data(), analysis.frequency.size());
            // DMR and P25 are narrowband; their symbols come at the audio rate as trimmed.
            if (mode == FmMode::Narrow) {
                const double rate = kAudioRate * (1.0 + trim);
                dmr.process(analysis.frequency.data(), analysis.frequency.size(), rate, dmr_voice);
                p25.process(analysis.frequency.data(), analysis.frequency.size(), rate, p25_voice);
                p25p2.process(analysis.frequency.data(), analysis.frequency.size(), rate, p25p2_voice);
                // A control channel names its system, and a Phase 2 channel can
                // give its own away: remember them for the Phase 2 channels to come.
                const auto remember = [&](const p25p2::System& system) {
                    if (std::find(systems.begin(), systems.end(), system) != systems.end())
                        return;
                    if (systems.size() >= kMaxSystems)
                        systems.erase(systems.begin());
                    systems.push_back(system);
                    p25p2.set_systems(systems);
                    std::lock_guard<std::mutex> lock(signal_mutex_);
                    p25_systems_ = systems;
                };
                const P25Status& control = p25.status();
                if (control.control && control.site.wacn >= 0 && control.site.system >= 0 && control.nac >= 0)
                    remember({static_cast<uint32_t>(control.site.wacn), static_cast<uint16_t>(control.site.system),
                              static_cast<uint16_t>(control.nac)});
                if (p25p2.status().keyed && p25p2.status().recovered)
                    remember(p25p2.status().system);
            } else if (dmr.status().frames > 0 || p25.status().frames > 0 || p25p2.status().timeslots > 0) {
                forget_dmr();
            }
            analysis.frequency.clear();
            analysis.power.clear();
        } else if (was_in_band) {
            demod.reset();
            classifier.reset();
            forget_dmr();
            searching = false;
        }
        // Digital voice, if any, through its vocoder; and while DMR or P25 is
        // on the channel its speech (or silence) takes the place of the FM
        // audio.  With DMR only or P25 only, what is not asked for is silent.
        const bool dmr_here   = in_band && dmr.status().present;
        const bool p25_here   = in_band && p25.status().present;
        const bool p25p2_here = in_band && p25p2.status().present;
        const bool digital    = dmr_here || p25_here || p25p2_here;
        const bool filtered   = p.dmr_only || p.p25_only;
        const bool hear_dmr   = dmr_here && (!filtered || p.dmr_only);
        const bool hear_p25   = p25_here && (!filtered || p.p25_only);
        const bool hear_p25p2 = p25p2_here && (!filtered || p.p25_only);
        if (overs.sink()) {
            // Who speaks, for the overs.  (All ones is no one known.)
            overs.set_channel(p.frequency);
            const auto identify = [&](VoiceOvers::Stream stream, const auto& call, std::string destination) {
                if (call.active && call.ids_known)
                    overs.identify(stream, destination, call.source == 0xFFFFFF ? 0 : call.source);
            };
            if (hear_dmr) {
                identify(VoiceOvers::DmrSlot1, dmr.status().slot[0], dmr_destination(dmr.status().slot[0]));
                identify(VoiceOvers::DmrSlot2, dmr.status().slot[1], dmr_destination(dmr.status().slot[1]));
            }
            if (hear_p25)
                identify(VoiceOvers::P25Phase1, p25.status().call, p25_destination(p25.status().call));
            if (hear_p25p2) {
                identify(VoiceOvers::P25Phase2Slot1, p25p2.status().slot[0], p25_destination(p25p2.status().slot[0]));
                identify(VoiceOvers::P25Phase2Slot2, p25p2.status().slot[1], p25_destination(p25p2.status().slot[1]));
            }
        }
        for (const DmrVoice& v : dmr_voice) {
            if (!hear_dmr)
                break;
            if (v.first)
                ambe.reset();
            ambe.decode(v.frame, speech);
            playout.push(speech, AmbeDecoder::kFrameSamples);
            overs.push(v.slot == 0 ? VoiceOvers::DmrSlot1 : VoiceOvers::DmrSlot2, speech, AmbeDecoder::kFrameSamples,
                       v.first);
        }
        for (const P25Voice& v : p25_voice) {
            if (!hear_p25)
                break;
            if (v.first)
                imbe.reset();
            imbe.decode(v.frame, speech);
            playout.push(speech, ImbeDecoder::kFrameSamples);
            overs.push(VoiceOvers::P25Phase1, speech, ImbeDecoder::kFrameSamples, v.first);
        }
        for (const P25Phase2Voice& v : p25p2_voice) {
            if (!hear_p25p2)
                break;
            if (v.first)
                ambe_p25p2.reset();
            ambe_p25p2.decode(v.frame, speech);
            playout.push(speech, AmbeDecoder::kFrameSamples);
            overs.push(v.slot == 0 ? VoiceOvers::P25Phase2Slot1 : VoiceOvers::P25Phase2Slot2, speech,
                       AmbeDecoder::kFrameSamples, v.first);
        }
        dmr_voice.clear();
        p25_voice.clear();
        p25p2_voice.clear();
        overs.advance(signal_time);
        const bool heard = hear_dmr || hear_p25 || hear_p25p2;
        if (heard && !audio.empty()) {
            playout.pull(audio.data(), audio.size(), kAudioRate * (1.0 + trim));
            for (float v : audio)
                speech_power += kSpeechSmoothing * (v * v - speech_power);
        } else if (!heard) {
            speech_power = 0.0f;
            if (digital || filtered)
                std::fill(audio.begin(), audio.end(), 0.0f);  // nothing to hear
        }
        // Tell the user interface what the classifier makes of the channel:
        // after each window, and when it has started afresh.  DMR or P25 that
        // decodes settles the question, whatever the statistics say (a call
        // of nothing but silence frames, for one, is far from random).
        {
            std::lock_guard<std::mutex> lock(signal_mutex_);
            dmr_   = dmr.status();
            p25_   = p25.status();
            p25p2_ = p25p2.status();
            const int decoded = dmr_here ? 1 : p25_here ? 2 : p25p2_here ? 3 : 0;
            if (classifier.windows() != published_windows || classifier.result().kind != published_kind ||
                decoded != published_dmr) {
                published_windows = classifier.windows();
                published_kind    = classifier.result().kind;
                published_dmr     = decoded;
                signal_           = classifier.result();
                if (decoded == 1 || decoded == 2) {
                    signal_.kind         = SignalKind::Fsk4;
                    signal_.symbol_rate  = Fsk4Front::kSymbolRate;
                    signal_.deviation_hz = decoded == 1 ? dmr_.level_hz : p25_.level_hz;
                    signal_.protocol     = decoded == 1 ? "DMR" : "P25 Phase 1";
                    signal_.ctcss_hz     = 0.0;
                } else if (decoded == 3) {
                    signal_.kind         = SignalKind::Dqpsk;
                    signal_.symbol_rate  = p25p2::kSymbolRate;
                    signal_.deviation_hz = p25p2_.level_hz;
                    signal_.protocol     = "P25 Phase 2";
                    signal_.ctcss_hz     = 0.0;
                }
            }
        }
        was_in_band = in_band;
        ring_.consume(n);

        // 3. Searching for the carrier: the mean frequency of whatever
        //    dominates the widened channel is where it is.
        if (searching && in_band) {
            const FmDemodulator::FrequencyEstimate estimate = demod.take_frequency_estimate();
            if (static_cast<double>(search_seen) >= kSearchSettleSeconds * kAudioRate) {
                search_sum += estimate.mean_hz * static_cast<double>(estimate.samples);
                search_count += estimate.samples;
            }
            search_seen += audio.size();
            std::fill(audio.begin(), audio.end(), 0.0f);  // nothing worth hearing yet
            const double needed = (mode == FmMode::Wide ? kSearchWideSeconds : kSearchNarrowSeconds) *
                                  demod.discriminator_rate();
            if (static_cast<double>(search_count) >= needed) {
                p.frequency += std::clamp(search_sum / static_cast<double>(search_count), -p.search, p.search);
                searching = false;
                std::lock_guard<std::mutex> lock(params_mutex_);
                if (params_.tune_serial == p.tune_serial)  // unless it has been retuned meanwhile
                    params_.frequency = p.frequency;
            }
        }

        // 4. Volume (with any boost beyond it; past full scale, the limiter
        //    takes over), then - as played, for the plot too - into the FIFO
        //    at a rate that keeps its fill level steady.
        if (!audio.empty()) {
            const float target_gain = p.muted ? 0.0f : p.volume * p.volume * p.boost;
            for (float& v : audio) {
                gain += kGainSmoothing * (target_gain - gain);
                v = limiter.process(v * gain);
            }
            scope_.write(audio.data(), audio.size(), p.muted);

            const bool                       playing = fifo_.playing();
            const AudioRateControl::Decision pace    = pacing.update(
                static_cast<double>(n) / sample_rate_, fifo_.fill(), fifo_.largest_read(), playing);
            fifo_.set_start_level(pace.start_level);
            demod.set_rate_trim(pace.trim);
            trim = std::clamp(pace.trim, -0.02, 0.02);  // as the demodulator takes it
            if (!pace.discard)
                fifo_.write(audio.data(), audio.size());
            else if (playing)
                st_overruns_.fetch_add(1, std::memory_order_relaxed);
            st_audio_samples_.fetch_add(audio.size(), std::memory_order_relaxed);
        }
        fifo_.set_flowing(in_band && enabled_.load(std::memory_order_relaxed));

        // 5. Status.
        const FmDemodulator::Stats stats = demod.stats();
        st_in_band_.store(in_band, std::memory_order_relaxed);
        if (!searching)
            st_settled_serial_.store(p.tune_serial, std::memory_order_release);
        st_squelch_open_.store(in_band && !searching && stats.squelch_open, std::memory_order_relaxed);
        st_level_.store(in_band ? stats.level_dbfs : -200.0f, std::memory_order_relaxed);
        const float audio_rms = heard ? std::sqrt(speech_power) : digital || filtered ? 0.0f : stats.audio_rms;
        st_audio_rms_.store(in_band && !searching ? audio_rms : 0.0f, std::memory_order_relaxed);
        st_trim_.store(pacing.trim(), std::memory_order_relaxed);

        const auto   now  = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double>(now - load_wall).count();
        if (wall >= 0.25) {
            const double cpu = current_thread_cpu_seconds();
            st_load_.store(static_cast<float>((cpu - load_cpu) / wall), std::memory_order_relaxed);
            load_wall = now;
            load_cpu  = cpu;
        }
    }
    overs.flush();  // what was being said when listening stopped
}

}  // namespace spectro
