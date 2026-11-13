#include "audio/listener.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "audio/rate_control.hpp"
#include "util/thread_util.hpp"

namespace spectro {

namespace {

constexpr double kTapSeconds = 0.1;      // slack between the receive thread and the audio thread
constexpr size_t kMaxBlock   = 1 << 16;  // IQ samples demodulated per pass

constexpr float kGainSmoothing = 0.002f;      // volume changes take about 10 ms

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

void Listener::tune(double frequency_hz, double search_hz)
{
    change([&](Params& p) {
        p.frequency = frequency_hz;
        p.search    = std::max(0.0, search_hz);
        ++p.tune_serial;
    });
}

void Listener::set_bandwidth(double hz)
{
    // Between the two ranges, the nearer one.
    hz = hz < 0.5 * (kMaxNarrowBandwidth + kMinWideBandwidth) ? std::clamp(hz, kMinNarrowBandwidth, kMaxNarrowBandwidth)
                                                              : std::clamp(hz, kMinWideBandwidth, kMaxWideBandwidth);
    change([&](Params& p) { p.bandwidth = hz; });
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

ListenStatus Listener::status() const
{
    ListenStatus s;
    {
        std::lock_guard<std::mutex> lock(params_mutex_);
        s.enabled   = params_.enabled;
        s.frequency = params_.frequency;
        s.bandwidth = usable_bandwidth(params_.bandwidth);
    }
    s.mode          = FmDemodulator::mode_for(s.bandwidth);
    s.in_band       = st_in_band_.load(std::memory_order_relaxed);
    s.searching     = st_searching_.load(std::memory_order_relaxed);
    s.squelch_open  = st_squelch_open_.load(std::memory_order_relaxed);
    s.level_dbfs    = st_level_.load(std::memory_order_relaxed);
    s.audio_rms     = st_audio_rms_.load(std::memory_order_relaxed);
    s.load          = st_load_.load(std::memory_order_relaxed);
    s.rate_trim     = st_trim_.load(std::memory_order_relaxed);
    s.audio_samples = st_audio_samples_.load(std::memory_order_relaxed);
    s.overruns      = st_overruns_.load(std::memory_order_relaxed);
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
    AudioRateControl   pacing(kAudioRate);
    std::vector<float> audio;
    audio.reserve(16384);

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
                fifo_.discard();
                gain      = 0.0f;
                searching = false;
            }
            if (next.tune_serial != p.tune_serial) {
                searching    = next.search > 0.0;
                search_seen  = 0;
                search_sum   = 0.0;
                search_count = 0;
            }
            demod.set_squelch(next.squelch);
            demod.set_deemphasis(next.deemphasis);
            p = next;
        }

        // 2. Demodulate what has arrived.  The channel is given as an absolute
        //    frequency, so retuning the radio moves it within the stream - or
        //    out of it, and then there is nothing to hear.
        const size_t           n     = std::min(ring_.readable(), kMaxBlock);
        const IqRing::ReadSpans spans = ring_.peek(n);

        const double bandwidth = usable_bandwidth(p.bandwidth);
        const FmMode mode      = FmDemodulator::mode_for(bandwidth);
        const double offset    = p.frequency - source_.frequency();
        const bool   in_band   = std::fabs(offset) + 0.5 * bandwidth <= 0.5 * sample_rate_;
        audio.clear();
        if (in_band) {
            const double width = searching ? bandwidth + kSearchWiden * p.search : bandwidth;
            demod.set_channel(offset, width, mode);
            demod.process(spans.first, spans.first_count, audio);
            if (spans.second_count > 0)
                demod.process(spans.second, spans.second_count, audio);
        } else if (was_in_band) {
            demod.reset();
            searching = false;
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

        // 4. Volume, then into the FIFO at a rate that keeps its fill level steady.
        if (!audio.empty()) {
            const float target_gain = p.volume * p.volume;
            for (float& v : audio) {
                gain += kGainSmoothing * (target_gain - gain);
                v = std::clamp(v * gain, -1.0f, 1.0f);
            }

            const bool                       playing = fifo_.playing();
            const AudioRateControl::Decision pace    = pacing.update(
                static_cast<double>(n) / sample_rate_, fifo_.fill(), fifo_.largest_read(), playing);
            fifo_.set_start_level(pace.start_level);
            demod.set_rate_trim(pace.trim);
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
        st_searching_.store(searching, std::memory_order_relaxed);
        st_squelch_open_.store(in_band && !searching && stats.squelch_open, std::memory_order_relaxed);
        st_level_.store(in_band ? stats.level_dbfs : -200.0f, std::memory_order_relaxed);
        st_audio_rms_.store(in_band && !searching ? stats.audio_rms : 0.0f, std::memory_order_relaxed);
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
}

}  // namespace spectro
