#include "radio/sim_source.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numbers>
#include <random>
#include <stdexcept>
#include <thread>

#include <fftw3.h>

#include "dsp/ambe_vocoder.hpp"
#include "dsp/dmr_transmitter.hpp"
#include "dsp/p25_phase2_transmitter.hpp"
#include "dsp/p25_transmitter.hpp"
#include "dsp/p25_trunking.hpp"
#include "dsp/fftw_lock.hpp"

namespace spectro {

namespace {

constexpr double kPi        = std::numbers::pi;
constexpr double kTwoPi     = 2.0 * std::numbers::pi;
constexpr double kFullScale = 2048.0;

constexpr int    kMaxBlock = 16384;       // samples generated per read() at most
constexpr size_t kNoiseLen = 1u << 19;    // complex samples of white noise
constexpr size_t kTableLen = 1u << 17;    // complex samples per band-limited table

constexpr double kNoiseSigma    = 4.0;    // receiver noise, counts per component at reference gain
constexpr double kAdcNoiseSigma = 0.5;    // converter noise, independent of gain

// The scene.  Frequencies are fractions of the sample rate, relative to the
// initial centre frequency; levels are dBFS at the reference gain.
constexpr double kFmCenter    = -0.19;
constexpr double kFmDeviation = 0.015;
constexpr double kFmRateHz    = 0.8;
constexpr double kFmLevel     = -38.0;

// A two-way radio channel.  Unlike everything else in the scene its width is
// fixed in Hz, not a fraction of the sample rate.
constexpr double kVoiceCenter        = -0.28;
constexpr double kVoiceLevel         = -36.0;
constexpr double kVoiceDeviationHz   = 1500.0;  // of a 1 kHz tone; see pre_emphasis()
constexpr double kVoicePreEmphasisS  = 750e-6;
constexpr double kVoiceNoteSeconds   = 0.25;
constexpr double kVoiceNotesHz[]     = {523.25, 659.25, 783.99, 1046.50, 783.99, 659.25};  // C5 E5 G5 C6 G5 E5
constexpr double kVoiceHalfWidthHz   = 4e3;     // room it needs on either side of its carrier

// A broadcast station, in the gap between the wide-band bursts and the weak
// carrier near the band edge.  Its width too is fixed in Hz.
constexpr double kBroadcastCenter       = 0.38;
constexpr double kBroadcastLevel        = -30.0;
constexpr double kBroadcastDeviationHz  = 60e3;    // of the arpeggio, before pre-emphasis
constexpr double kBroadcastPilotHz      = 19e3;
constexpr double kBroadcastPilotDevHz   = 6.75e3;  // 9 % of 75 kHz
constexpr double kBroadcastPreEmphasisS = 75e-6;
constexpr double kBroadcastNoteSeconds  = 0.3;
constexpr double kBroadcastNotesHz[]    = {220.0, 261.63, 329.63, 440.0, 329.63, 261.63};  // A3 C4 E4 A4 E4 C4
constexpr double kBroadcastHalfWidthHz  = 100e3;

// Digital stations, above the voice channel and clear of the rest.
constexpr double kDmrCenter     = -0.24;
constexpr double kDmrLevel      = -40.0;
constexpr double kPagerCenter   = -0.15;
constexpr double kPagerLevel    = -42.0;
constexpr double kP25Center     = 0.14;
constexpr double kP25Level      = -41.0;
constexpr double kDigitalHalfWidthHz = 6.25e3;
// DMR's base station data sync, one symbol per character ('1' = +3, '3' = -3).
constexpr uint32_t    kPocsagSync = 0x7CD215D8;

constexpr double kChirpLo      = -0.47;
constexpr double kChirpHi      = 0.47;
constexpr double kChirpPeriodS = 3.0;
constexpr double kChirpLevel   = -48.0;

constexpr double kBurstCenter    = 0.25;
constexpr double kBurstBandwidth = 0.16;
constexpr double kBurstLevel     = -32.0;

constexpr double kHopFirst     = -0.12;
constexpr double kHopSpacing   = 0.012;
constexpr int    kHopChannels  = 20;
constexpr double kHopBandwidth = 0.008;
constexpr double kHopLevel     = -40.0;

constexpr float kRampSlope = 1.0f / 128.0f;  // burst edges: 128-sample linear ramps

double amplitude(double dbfs) { return kFullScale * std::pow(10.0, dbfs / 20.0); }

// True if a signal centred at f (cycles/sample) with the given half bandwidth
// lies inside the Nyquist band.  Anything else is muted rather than aliased.
bool in_band(double f, double half_bandwidth) { return std::fabs(f) + half_bandwidth < 0.495; }

// mix += amp * exp(j (phase + dphi k)), k = 0..n-1
void rotate_add(float* mix, int n, double phase, double dphi, double amp)
{
    double       zr = amp * std::cos(phase), zi = amp * std::sin(phase);
    const double wr = std::cos(dphi), wi = std::sin(dphi);
    for (int i = 0; i < n; ++i) {
        mix[2 * i]     += static_cast<float>(zr);
        mix[2 * i + 1] += static_cast<float>(zi);
        const double t = zr * wr - zi * wi;
        zi             = zr * wi + zi * wr;
        zr             = t;
    }
}

// Periodic complex noise with a flat spectrum `bandwidth` (fraction of the
// sample rate) wide, centred on zero, normalised to unit RMS.
std::vector<float> make_band_noise(size_t length, double bandwidth, std::mt19937& rng)
{
    fftwf_complex* spectrum = fftwf_alloc_complex(length);
    if (!spectrum)
        throw std::bad_alloc();
    for (size_t i = 0; i < length; ++i)
        spectrum[i][0] = spectrum[i][1] = 0.0f;

    const long half = std::max<long>(1, std::lround(bandwidth * static_cast<double>(length) / 2.0));
    std::uniform_real_distribution<float> random_phase(0.0f, static_cast<float>(kTwoPi));
    for (long k = -half; k <= half; ++k) {
        const size_t bin   = static_cast<size_t>(k < 0 ? k + static_cast<long>(length) : k);
        const float  phase = random_phase(rng);
        spectrum[bin][0]   = std::cos(phase);
        spectrum[bin][1]   = std::sin(phase);
    }

    fftwf_plan plan;
    {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        plan = fftwf_plan_dft_1d(static_cast<int>(length), spectrum, spectrum, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    if (!plan) {
        fftwf_free(spectrum);
        throw std::runtime_error("simulator: FFTW could not create a plan");
    }
    fftwf_execute(plan);
    {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        fftwf_destroy_plan(plan);
    }

    const float        scale = 1.0f / std::sqrt(static_cast<float>(2 * half + 1));
    std::vector<float> table(2 * length);
    for (size_t i = 0; i < length; ++i) {
        table[2 * i]     = spectrum[i][0] * scale;
        table[2 * i + 1] = spectrum[i][1] * scale;
    }
    fftwf_free(spectrum);
    return table;
}

uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

// Gain of an FM transmitter's pre-emphasis (+6 dB per octave), 1 at 1 kHz.
double pre_emphasis(double hz)
{
    const auto response = [](double f) { return std::hypot(1.0, kTwoPi * f * kVoicePreEmphasisS); };
    return response(hz) / response(1000.0);
}

// Something like speech, -1 ... 1, `count` samples from `first` at `rate`: a
// voice whose pitch wanders between 100 and 190 Hz, two formants, and
// syllables about four times a second.  The pitch is taken as steady over
// one call (20 ms or so) to keep it cheap.
void synthetic_speech(uint64_t first, size_t count, double rate, float* out)
{
    const auto pitch_at = [](double t) {
        return 145.0 + 45.0 * std::sin(2.0 * kPi * 0.7 * t) + 10.0 * std::sin(2.0 * kPi * 3.1 * t);
    };
    const auto cycles_at = [](double t) {  // the integral of the pitch
        return 145.0 * t - 45.0 / (2.0 * kPi * 0.7) * std::cos(2.0 * kPi * 0.7 * t) -
               10.0 / (2.0 * kPi * 3.1) * std::cos(2.0 * kPi * 3.1 * t);
    };
    const double middle = (static_cast<double>(first) + 0.5 * static_cast<double>(count)) / rate;
    const double pitch  = pitch_at(middle);
    double       gain[25];
    int          harmonics = 0;
    for (; harmonics < 25 && (harmonics + 1) * pitch <= 3500.0; ++harmonics) {
        const double f   = (harmonics + 1) * pitch;
        gain[harmonics] = std::exp(-std::pow((f - 500.0) / 250.0, 2.0)) +
                          0.6 * std::exp(-std::pow((f - 1500.0) / 400.0, 2.0)) + 0.05;
    }
    for (size_t i = 0; i < count; ++i) {
        const double t      = static_cast<double>(first + i) / rate;
        const double cycles = cycles_at(t);
        double       v      = 0.0;
        for (int h = 1; h <= harmonics; ++h)
            v += gain[h - 1] * std::sin(2.0 * kPi * h * cycles + 0.3 * h);
        const double syllable = std::max(0.0, std::sin(2.0 * kPi * 2.0 * t + 0.4 * std::sin(2.0 * kPi * 0.3 * t)));
        out[i]                = static_cast<float>(0.35 * v * syllable);
    }
}

}  // namespace

std::vector<SimSource::Carrier> SimSource::carriers(double sample_rate)
{
    return {{-0.36 * sample_rate, -30.0}, {0.43 * sample_rate, -55.0}};
}

SimSource::VoiceChannel SimSource::voice_channel(double sample_rate)
{
    return {kVoiceCenter * sample_rate, kVoiceLevel, kVoiceDeviationHz};
}

SimSource::BroadcastStation SimSource::broadcast_station(double sample_rate)
{
    return {kBroadcastCenter * sample_rate, kBroadcastLevel, kBroadcastDeviationHz, kBroadcastPilotDevHz,
            kBroadcastPreEmphasisS};
}

std::vector<SimSource::DigitalStation> SimSource::digital_stations(double sample_rate)
{
    return {{"DMR", kDmrCenter * sample_rate, kDmrLevel, 4800.0, 1944.0, 4},
            {"POCSAG", kPagerCenter * sample_rate, kPagerLevel, 1200.0, 4500.0, 2},
            {"P25", kP25Center * sample_rate, kP25Level, 4800.0, 1800.0, 4},
            {"P25 control", kP25Center * sample_rate + kP25ControlAboveHz, kP25Level, 4800.0, 1800.0, 4},
            {"P25 Phase 2", kP25Center * sample_rate - kP25Phase2BelowHz, kP25Level, 6000.0, 2250.0, 4}};
}

SimSource::SimSource(const SourceConfig& config)
    : sample_rate_(config.sample_rate),
      initial_frequency_(config.frequency),
      seed_(config.sim_seed),
      frequency_(config.frequency),
      gain_(config.gain)
{
    if (!(sample_rate_ >= 1000.0))
        throw std::runtime_error("simulator: the sample rate must be at least 1 kS/s");

    info_.kind            = "simulator";
    info_.description     = "Built-in signal simulator (no radio hardware in use)";
    info_.tunable         = true;
    info_.has_gain        = true;
    info_.frequency_range = {1e6, 6e9, 1.0};
    info_.gain_range      = {0.0, 60.0, 1.0};

    // DMR: 4800 Bd, levels +/-648 and +/-1944 Hz (values 0..3 = -3, -1, +1, +3),
    // from a repeater that makes a call now and then.
    dmr_         = std::make_shared<DmrTransmitter>(kDmrColorCode);
    dmr_encoder_ = std::make_shared<AmbeEncoder>();
    Fsk dmr{kDmrCenter, kDmrLevel, 4800.0, {-1944.0, -648.0, 648.0, 1944.0}, {}, 144, 66};
    dmr.source = [this] { return next_dmr_symbol(); };
    // POCSAG: 1200 bit/s at +/-4.5 kHz (a 1 is the lower frequency), batches of
    // 17 codewords starting with the frame sync.
    Fsk pager{kPagerCenter, kPagerLevel, 1200.0, {4500.0, -4500.0}, {}, 17 * 32, 0};
    for (int b = 31; b >= 0; --b)
        pager.sync.push_back(static_cast<int>((kPocsagSync >> b) & 1u));
    // P25: 4800 Bd, levels +/-600 and +/-1800 Hz, a conventional radio.
    p25_         = std::make_shared<P25Transmitter>(kP25Nac);
    p25_encoder_ = std::make_shared<ImbeEncoder>();
    Fsk p25{kP25Center, kP25Level, 4800.0, {-1800.0, -600.0, 600.0, 1800.0}, {}, 864, 0};
    p25.source = [this] { return next_p25_symbol(); };
    // Its control channel.
    p25_control_ = std::make_shared<P25Transmitter>(kP25Nac);
    p25_control_->control([this] { return next_p25_block(); });
    Fsk control{kP25Center + kP25ControlAboveHz / sample_rate_, kP25Level, 4800.0, {-1800.0, -600.0, 600.0, 1800.0}, {}, 360, 0};
    control.source = [this] { return next_p25_control_symbol(); };
    // A Phase 2 voice channel: 6000 Bd phase steps of +-45 and +-135 degrees,
    // sent as their frequencies, +-750 and +-2250 Hz - the scrambling key is
    // the system's, from the control channel.
    p25p2_         = std::make_shared<P25Phase2Transmitter>(p25p2::System{kP25Wacn, kP25System, kP25Nac});
    p25p2_encoder_ = std::make_shared<AmbeEncoder>();
    Fsk phase2{kP25Center - kP25Phase2BelowHz / sample_rate_, kP25Level, 6000.0, {-2250.0, -750.0, 750.0, 2250.0}, {}, 180, 0};
    phase2.source = [this] { return next_p25_phase2_symbol(); };
    fsk_ = {dmr, pager, p25, control, phase2};

    tones_ = {
        {-0.36, -30.0, 0.0, 0.0},   // strong steady carrier
        {0.06, -45.0, 0.25, 0.9},   // slowly fading carrier
        {0.43, -55.0, 0.0, 0.0},    // weak carrier near the band edge
    };
}

SimSource::~SimSource() = default;

uint32_t SimSource::next_random()
{
    // xoshiro128++
    uint32_t*      s      = rng_state_;
    const uint32_t result = rotl(s[0] + s[3], 7) + s[0];
    const uint32_t t      = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 11);
    return result;
}

double SimSource::random_uniform()
{
    return static_cast<double>(next_random()) * (1.0 / 4294967296.0);
}

void SimSource::build_tables()
{
    std::mt19937 rng(seed_ * 2654435761u + 12345u);

    noise_.resize(2 * kNoiseLen);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (float& v : noise_)
        v = gauss(rng);

    wide_table_   = make_band_noise(kTableLen, kBurstBandwidth, rng);
    narrow_table_ = make_band_noise(kTableLen, kHopBandwidth, rng);
}

void SimSource::start()
{
    if (running_.load())
        return;
    if (noise_.empty())
        build_tables();

    mix_.assign(2 * kMaxBlock, 0.0f);
    envelope_.assign(kMaxBlock, 0.0f);

    // Seed the event generator (splitmix-style expansion of the seed).
    uint32_t z = seed_ + 0x9E3779B9u;
    for (uint32_t& s : rng_state_) {
        z += 0x9E3779B9u;
        uint32_t v = z;
        v = (v ^ (v >> 16)) * 0x85EBCA6Bu;
        v = (v ^ (v >> 13)) * 0xC2B2AE35u;
        s = v ^ (v >> 16);
    }

    for (Tone& tone : tones_)
        tone.phase = 0.0;
    fm_phase_ = chirp_phase_ = wide_phase_ = hop_phase_ = 0.0;
    voice_phase_ = voice_audio_phase_ = 0.0;
    broadcast_phase_ = broadcast_audio_phase_ = broadcast_pilot_phase_ = 0.0;
    for (size_t i = 0; i < fsk_.size(); ++i) {
        Fsk& f      = fsk_[i];
        f.phase     = 0.0;
        f.in_symbol = 0.0;
        f.index     = 0;
        f.random    = seed_ * 2654435761u + 977u * static_cast<uint32_t>(i + 1);
        f.prev      = next_fsk_symbol(f);
        f.curr      = next_fsk_symbol(f);
        f.next      = next_fsk_symbol(f);
    }
    chirp_freq_      = kChirpLo;
    wide_pos_ = narrow_pos_ = 0;
    burst_on_ = hop_on_ = false;
    burst_remaining_ = hop_remaining_ = 0;
    burst_level_ = hop_level_ = 0.0f;
    time_s_   = 0.0;
    produced_ = 0;

    start_time_ = std::chrono::steady_clock::now();
    running_.store(true);
}

void SimSource::stop()
{
    running_.store(false);
}

bool SimSource::set_frequency(double hz)
{
    frequency_.store(std::clamp(hz, info_.frequency_range.min, info_.frequency_range.max));
    return true;
}

bool SimSource::set_gain(double db)
{
    gain_.store(std::clamp(db, info_.gain_range.min, info_.gain_range.max));
    return true;
}

void SimSource::add_noise(int n)
{
    // A random excerpt of the noise table; this also initialises the mix.
    const size_t span  = noise_.size() - 2 * static_cast<size_t>(kMaxBlock);
    const size_t pos   = (static_cast<size_t>(next_random()) * 2) % span;
    const float* src   = noise_.data() + pos;
    float*       mix   = mix_.data();
    const float  sigma = static_cast<float>(kNoiseSigma);
    for (int j = 0; j < 2 * n; ++j)
        mix[j] = src[j] * sigma;
}

void SimSource::add_tone(Tone& tone, int n, double shift)
{
    const double f    = tone.offset + shift;
    const double dphi = kTwoPi * f;
    if (in_band(f, 0.0)) {
        double amp = amplitude(tone.level_dbfs);
        if (tone.fade_hz > 0.0)
            amp *= 1.0 - tone.fade_depth * (0.5 + 0.5 * std::sin(kTwoPi * tone.fade_hz * time_s_));
        rotate_add(mix_.data(), n, tone.phase, dphi, amp);
    }
    tone.phase = std::fmod(tone.phase + dphi * n, kTwoPi);
}

void SimSource::add_fm(int n, double shift)
{
    // A carrier whose frequency wanders sinusoidally.  The modulation is slow, so
    // the frequency is simply held constant over short segments.
    const double amp = amplitude(kFmLevel);
    for (int done = 0; done < n;) {
        const int    seg  = std::min(64, n - done);
        const double t    = time_s_ + done / sample_rate_;
        const double f    = kFmCenter + shift + kFmDeviation * std::sin(kTwoPi * kFmRateHz * t);
        const double dphi = kTwoPi * f;
        if (in_band(f, 0.0))
            rotate_add(mix_.data() + 2 * done, seg, fm_phase_, dphi, amp);
        fm_phase_ = std::fmod(fm_phase_ + dphi * seg, kTwoPi);
        done += seg;
    }
}

void SimSource::add_voice(int n, double shift)
{
    // Narrowband FM: the carrier follows an audio tone, here one note of a tune
    // at a time.  As in add_fm() the frequency is held over short segments,
    // which are kept below 5 us so that the steps are far above the audio band.
    const double amp     = amplitude(kVoiceLevel);
    const int    max_seg = std::clamp(static_cast<int>(sample_rate_ / 200e3), 1, 64);
    const double center  = kVoiceCenter + shift;
    const bool   on_air  = in_band(center, kVoiceHalfWidthHz / sample_rate_);

    for (int done = 0; done < n;) {
        const int    seg   = std::min(max_seg, n - done);
        const double t     = time_s_ + done / sample_rate_;
        const double pitch = kVoiceNotesHz[static_cast<size_t>(t / kVoiceNoteSeconds) % std::size(kVoiceNotesHz)];
        const double audio = kTwoPi * pitch / sample_rate_;  // radians per sample
        const double tone  = std::sin(voice_audio_phase_ + 0.5 * audio * seg);
        const double f     = center + kVoiceDeviationHz * pre_emphasis(pitch) * tone / sample_rate_;
        const double dphi  = kTwoPi * f;
        if (on_air)
            rotate_add(mix_.data() + 2 * done, seg, voice_phase_, dphi, amp);
        voice_phase_       = std::fmod(voice_phase_ + dphi * seg, kTwoPi);
        voice_audio_phase_ = std::fmod(voice_audio_phase_ + audio * seg, kTwoPi);
        done += seg;
    }
}

void SimSource::add_broadcast(int n, double shift)
{
    // Wideband FM, built like add_voice(): the frequency is held over segments
    // short enough (5 us at most) that the steps lie far above the audio.  Only
    // on air where the stream is wide enough to hold it without overlapping
    // the rest of the scene.
    const double center = kBroadcastCenter + shift;
    if (sample_rate_ < kBroadcastMinRate || !in_band(center, kBroadcastHalfWidthHz / sample_rate_)) {
        broadcast_phase_ = std::fmod(broadcast_phase_ + kTwoPi * center * n, kTwoPi);
        return;
    }
    const double amp     = amplitude(kBroadcastLevel);
    const int    max_seg = std::clamp(static_cast<int>(sample_rate_ / 200e3), 1, 64);
    const double pilot   = kTwoPi * kBroadcastPilotHz / sample_rate_;  // radians per sample

    for (int done = 0; done < n;) {
        const int    seg   = std::min(max_seg, n - done);
        const double t     = time_s_ + done / sample_rate_;
        const double pitch = kBroadcastNotesHz[static_cast<size_t>(t / kBroadcastNoteSeconds) %
                                               std::size(kBroadcastNotesHz)];
        const double audio = kTwoPi * pitch / sample_rate_;
        // Pre-emphasis: the gain of 1 + j w tau, which the receiver's
        // de-emphasis (unity at low frequencies) undoes exactly.
        const double boost = std::hypot(1.0, kTwoPi * pitch * kBroadcastPreEmphasisS);
        const double dev   = kBroadcastDeviationHz * boost * std::sin(broadcast_audio_phase_ + 0.5 * audio * seg) +
                             kBroadcastPilotDevHz * std::sin(broadcast_pilot_phase_ + 0.5 * pilot * seg);
        const double dphi  = kTwoPi * (center + dev / sample_rate_);
        rotate_add(mix_.data() + 2 * done, seg, broadcast_phase_, dphi, amp);
        broadcast_phase_       = std::fmod(broadcast_phase_ + dphi * seg, kTwoPi);
        broadcast_audio_phase_ = std::fmod(broadcast_audio_phase_ + audio * seg, kTwoPi);
        broadcast_pilot_phase_ = std::fmod(broadcast_pilot_phase_ + pilot * seg, kTwoPi);
        done += seg;
    }
}

int SimSource::next_fsk_symbol(Fsk& fsk)
{
    if (fsk.source)
        return fsk.source();
    const size_t at    = fsk.index;
    fsk.index          = (fsk.index + 1) % fsk.frame;
    if (at >= fsk.sync_at && at < fsk.sync_at + fsk.sync.size())
        return fsk.sync[at - fsk.sync_at];
    // xorshift32: its own random numbers, so the rest of the scene stays as it was.
    fsk.random ^= fsk.random << 13;
    fsk.random ^= fsk.random >> 17;
    fsk.random ^= fsk.random << 5;
    return static_cast<int>((fsk.random >> 8) % fsk.levels_hz.size());
}

// The DMR repeater: its calls start on time, and say something.
int SimSource::next_dmr_symbol()
{
    constexpr double kFrameSeconds = 0.03;
    if (dmr_symbols_ % dmr::kFrameSymbols == 0) {
        const uint64_t frame     = dmr_symbols_ / dmr::kFrameSymbols;
        const auto     per_cycle = static_cast<uint64_t>(std::lround(kDmrCycleSeconds / kFrameSeconds));
        for (const DmrCall& c : kDmrCalls) {
            if (frame % per_cycle != static_cast<uint64_t>(std::lround(c.start / kFrameSeconds)))
                continue;
            dmr::LinkControl lc;
            lc.destination = c.talkgroup;
            lc.source      = c.source;
            dmr_encoder_->reset();
            const int superframes = std::max(1, static_cast<int>(std::lround(c.seconds / 0.36)));
            dmr_->start_call(c.slot, lc, superframes, [this] {
                // Without a vocoder there is no point in speaking: it sends silence.
                if (!AmbeEncoder::available())
                    return dmr::ambe_silence();
                float speech[AmbeEncoder::kFrameSamples];
                synthetic_speech(dmr_speech_, AmbeEncoder::kFrameSamples, 8000.0, speech);
                dmr_speech_ += AmbeEncoder::kFrameSamples;
                for (float& v : speech)
                    v *= 0.3f;
                return dmr_encoder_->encode(speech);
            });
        }
    }
    ++dmr_symbols_;
    return (dmr_->next_symbol() + 3) / 2;
}

// The P25 radio: on the air for its call.
int SimSource::next_p25_symbol()
{
    const auto per_cycle = static_cast<uint64_t>(std::llround(kDmrCycleSeconds * 4800.0));
    if (p25_symbols_ % per_cycle == static_cast<uint64_t>(std::llround(kP25Call.start * 4800.0))) {
        p25::LinkControl lc;
        lc.destination = kP25Call.talkgroup;
        lc.source      = kP25Call.source;
        p25_encoder_->reset();
        const int ldus = std::max(2, static_cast<int>(std::lround(kP25Call.seconds / 0.18)));
        p25_->start_call(lc, ldus, [this] {
            if (!ImbeEncoder::available())
                return p25::ImbeFrame{};
            float speech[ImbeEncoder::kFrameSamples];
            synthetic_speech(p25_speech_ + 40000, ImbeEncoder::kFrameSamples, 8000.0, speech);  // another voice
            p25_speech_ += ImbeEncoder::kFrameSamples;
            for (float& v : speech)
                v *= 0.3f;
            return p25_encoder_->encode(speech);
        });
    }
    ++p25_symbols_;
    int symbol = 0;
    return p25_->next_symbol(symbol) ? (symbol + 3) / 2 : -1;
}

int SimSource::next_p25_control_symbol()
{
    int symbol = 0;
    return p25_control_->next_symbol(symbol) ? (symbol + 3) / 2 : -1;
}

// The Phase 2 voice channel: on the air for its call.
int SimSource::next_p25_phase2_symbol()
{
    const auto per_cycle = static_cast<uint64_t>(std::llround(kDmrCycleSeconds * p25p2::kSymbolRate));
    if (p25p2_symbols_ % per_cycle == static_cast<uint64_t>(std::llround(kP25Phase2Call.start * p25p2::kSymbolRate))) {
        p25p2_encoder_->reset();
        const int superframes = std::max(1, static_cast<int>(std::lround(kP25Phase2Call.seconds / 0.36)));
        p25p2_->start_call(kP25Phase2Call.slot, static_cast<uint16_t>(kP25Phase2Call.talkgroup), kP25Phase2Call.source,
                           superframes, [this] {
                               if (!AmbeEncoder::available())
                                   return dmr::ambe_silence();
                               float speech[AmbeEncoder::kFrameSamples];
                               synthetic_speech(p25p2_speech_ + 80000, AmbeEncoder::kFrameSamples, 8000.0, speech);
                               p25p2_speech_ += AmbeEncoder::kFrameSamples;
                               for (float& v : speech)
                                   v *= 0.3f;
                               return p25p2_encoder_->encode(speech);
                           });
    }
    ++p25p2_symbols_;
    int symbol = 0;
    return p25p2_->next_symbol(symbol) ? (symbol + 3) / 2 : -1;
}

// What the control channel says: who the system is, the channel plans - an
// FDMA one that puts the P25 radio on channel 10, a TDMA one with the Phase 2
// channel - and while either talks, its grant.
p25::Tsbk SimSource::next_p25_block()
{
    const double   voice_hz        = initial_frequency_ + kP25Center * sample_rate_;
    const double   base_hz         = voice_hz - 10 * 12.5e3;
    const uint16_t voice_channel   = 0x1000 | 10;
    const uint16_t control_channel = static_cast<uint16_t>(0x1000 | (10 + std::lround(kP25ControlAboveHz / 12.5e3)));
    // Plan 2 counts two slots to a carrier.
    const auto     phase2_carrier  = static_cast<uint16_t>(10 - std::lround(kP25Phase2BelowHz / 12.5e3));
    const auto     phase2_channel  = static_cast<uint16_t>(0x2000 | (2 * phase2_carrier + kP25Phase2Call.slot));
    const uint64_t block           = p25_blocks_++ % 6;
    switch (block) {
    case 0: return tsbk::network_status(kP25Wacn, kP25System, control_channel);
    case 1: return tsbk::rfss_status(kP25System, 1, 2, control_channel);
    case 2: return tsbk::channel_plan(1, base_hz, 12.5e3, -45e6);
    case 3: return tsbk::channel_plan_tdma(2, 3, base_hz, 12.5e3, -45e6);
    default:
        if (p25_->on_air())
            return tsbk::group_grant(voice_channel, kP25Call.talkgroup, kP25Call.source);
        if (p25p2_->on_air())
            return tsbk::group_grant(phase2_channel, static_cast<uint16_t>(kP25Phase2Call.talkgroup), kP25Phase2Call.source);
        return block == 4 ? tsbk::channel_plan(1, base_hz, 12.5e3, -45e6)
                          : tsbk::channel_plan_tdma(2, 3, base_hz, 12.5e3, -45e6);
    }
}

void SimSource::add_fsk(Fsk& fsk, int n, double shift)
{
    // Like add_voice(): the frequency is held over segments of at most 5 us,
    // and never across a symbol boundary.  Between symbols it moves on a
    // raised-cosine curve over half a symbol, so the transmitter stays within
    // its channel.
    const double center = fsk.center + shift;
    const double step   = fsk.rate / sample_rate_;  // symbols per sample
    const bool   on_air = in_band(center, kDigitalHalfWidthHz / sample_rate_);
    const double amp    = amplitude(fsk.level_dbfs);
    const int    max_seg = std::clamp(static_cast<int>(sample_rate_ / 200e3), 1, 64);
    const auto   blend  = [](double a, double b, double w) { return a + (b - a) * (0.5 - 0.5 * std::cos(kPi * w)); };

    for (int done = 0; done < n;) {
        const int to_boundary = std::max(1, static_cast<int>(std::ceil((1.0 - fsk.in_symbol) / step)));
        const int seg         = std::min({max_seg, n - done, to_boundary});
        const double u        = std::min(fsk.in_symbol + 0.5 * seg * step, 0.9999);
        const auto   level = [&](int i) { return i < 0 ? 0.0 : fsk.levels_hz[static_cast<size_t>(i)]; };  // -1: off air
        const double prev  = level(fsk.prev);
        const double curr  = level(fsk.curr);
        const double next  = level(fsk.next);
        double       dev  = curr;
        if (u < 0.25)
            dev = blend(prev, curr, (u + 0.25) / 0.5);
        else if (u > 0.75)
            dev = blend(curr, next, (u - 0.75) / 0.5);
        const double dphi = kTwoPi * (center + dev / sample_rate_);
        if (on_air && fsk.curr >= 0)
            rotate_add(mix_.data() + 2 * done, seg, fsk.phase, dphi, amp);
        fsk.phase = std::fmod(fsk.phase + dphi * seg, kTwoPi);
        fsk.in_symbol += seg * step;
        while (fsk.in_symbol >= 1.0 - 1e-9) {
            fsk.in_symbol = std::max(0.0, fsk.in_symbol - 1.0);
            fsk.prev      = fsk.curr;
            fsk.curr      = fsk.next;
            fsk.next      = next_fsk_symbol(fsk);
        }
        done += seg;
    }
}

void SimSource::add_chirp(int n, double shift)
{
    // Linear sweep across most of the band, restarting at the bottom.
    const double amp = amplitude(kChirpLevel);
    const double df  = (kChirpHi - kChirpLo) / (kChirpPeriodS * sample_rate_);  // per sample

    for (int done = 0; done < n;) {
        const double to_wrap = std::ceil((kChirpHi - chirp_freq_) / df);
        const int    seg     = static_cast<int>(std::clamp(to_wrap, 1.0, static_cast<double>(n - done)));
        const double f0      = chirp_freq_ + shift;
        const double f1      = f0 + df * seg;

        if (in_band(f0, 0.0) && in_band(f1, 0.0)) {
            float* mix = mix_.data() + 2 * done;
            double zr = amp * std::cos(chirp_phase_), zi = amp * std::sin(chirp_phase_);
            double wr = std::cos(kTwoPi * f0), wi = std::sin(kTwoPi * f0);
            const double dr = std::cos(kTwoPi * df), di = std::sin(kTwoPi * df);
            for (int i = 0; i < seg; ++i) {
                mix[2 * i]     += static_cast<float>(zr);
                mix[2 * i + 1] += static_cast<float>(zi);
                const double tz = zr * wr - zi * wi;
                zi = zr * wi + zi * wr;
                zr = tz;
                const double tw = wr * dr - wi * di;
                wi = wr * di + wi * dr;
                wr = tw;
            }
        }
        // Phase advance: sum over k of 2 pi (f0 + k df).
        const double s = static_cast<double>(seg);
        chirp_phase_   = std::fmod(chirp_phase_ + kTwoPi * (s * f0 + df * s * (s - 1.0) / 2.0), kTwoPi);
        chirp_freq_ += df * s;
        if (chirp_freq_ >= kChirpHi)
            chirp_freq_ = kChirpLo;
        done += seg;
    }
}

// mix += amplitude * envelope[k] * table[k] * exp(j (phase + 2 pi freq k))
void SimSource::add_table_signal(const std::vector<float>& table, size_t& table_pos, double& phase,
                                 double freq, float amp, const float* envelope, float* mix, int n)
{
    const size_t mask = table.size() / 2 - 1;
    const double dphi = kTwoPi * freq;
    double       zr = std::cos(phase), zi = std::sin(phase);
    const double wr = std::cos(dphi), wi = std::sin(dphi);
    size_t       p  = table_pos;
    for (int i = 0; i < n; ++i) {
        const float a  = amp * envelope[i];
        const float tr = table[2 * p], ti = table[2 * p + 1];
        const float cr = static_cast<float>(zr), ci = static_cast<float>(zi);
        mix[2 * i]     += a * (tr * cr - ti * ci);
        mix[2 * i + 1] += a * (tr * ci + ti * cr);
        const double t = zr * wr - zi * wi;
        zi             = zr * wi + zi * wr;
        zr             = t;
        p              = (p + 1) & mask;
    }
    table_pos = p;
    phase     = std::fmod(phase + dphi * n, kTwoPi);
}

void SimSource::add_bursts(int n, double shift)
{
    // Wide-band packets of random length, like a busy Wi-Fi channel.
    float* env  = envelope_.data();
    float  peak = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (burst_remaining_ <= 0) {
            burst_on_ = !burst_on_;
            const double ms = burst_on_ ? 4.0 + 26.0 * random_uniform() : 10.0 + 110.0 * random_uniform();
            burst_remaining_ = std::max<int64_t>(256, std::llround(ms * 1e-3 * sample_rate_));
        }
        --burst_remaining_;
        burst_level_ = std::clamp(burst_level_ + (burst_on_ ? kRampSlope : -kRampSlope), 0.0f, 1.0f);
        env[i]       = burst_level_;
        peak         = std::max(peak, burst_level_);
    }

    const double f = kBurstCenter + shift;
    if (peak > 0.0f && in_band(f, kBurstBandwidth / 2.0)) {
        add_table_signal(wide_table_, wide_pos_, wide_phase_, f, static_cast<float>(amplitude(kBurstLevel)),
                         env, mix_.data(), n);
    }
}

void SimSource::add_hopper(int n, double shift)
{
    // Short narrow-band bursts on a pseudo-random channel, like Bluetooth.
    const float   amp    = static_cast<float>(amplitude(kHopLevel));
    const int64_t on_len = std::max<int64_t>(256, std::llround(5e-3 * sample_rate_));
    const int64_t off_len = std::max<int64_t>(256, std::llround(3e-3 * sample_rate_));

    for (int done = 0; done < n;) {
        if (hop_remaining_ <= 0) {
            hop_on_ = !hop_on_;
            if (hop_on_)
                hop_offset_ = kHopFirst + kHopSpacing * static_cast<double>(next_random() % kHopChannels);
            hop_remaining_ = hop_on_ ? on_len : off_len;
        }
        const int seg  = static_cast<int>(std::min<int64_t>(n - done, hop_remaining_));
        float*    env  = envelope_.data();
        float     peak = 0.0f;
        for (int i = 0; i < seg; ++i) {
            hop_level_ = std::clamp(hop_level_ + (hop_on_ ? kRampSlope : -kRampSlope), 0.0f, 1.0f);
            env[i]     = hop_level_;
            peak       = std::max(peak, hop_level_);
        }
        const double f = hop_offset_ + shift;
        if (peak > 0.0f && in_band(f, kHopBandwidth / 2.0))
            add_table_signal(narrow_table_, narrow_pos_, hop_phase_, f, amp, env, mix_.data() + 2 * done, seg);
        hop_remaining_ -= seg;
        done += seg;
    }
}

void SimSource::generate(int n)
{
    // Tuning the "radio" moves every signal the opposite way in baseband.
    const double shift = (initial_frequency_ - frequency_.load(std::memory_order_relaxed)) / sample_rate_;

    add_noise(n);
    for (Tone& tone : tones_)
        add_tone(tone, n, shift);
    add_fm(n, shift);
    add_voice(n, shift);
    add_broadcast(n, shift);
    for (Fsk& fsk : fsk_)
        add_fsk(fsk, n, shift);
    add_chirp(n, shift);
    add_bursts(n, shift);
    add_hopper(n, shift);

    time_s_ += n / sample_rate_;
}

int SimSource::read(int16_t* iq, int max_samples, ReadStatus& status)
{
    status = {};
    if (!running_.load(std::memory_order_relaxed) || max_samples <= 0)
        return -1;

    const int n = std::min(max_samples, kMaxBlock);
    generate(n);

    // Receiver gain, converter noise, and clipping at the converter's full scale.
    const float  gain = std::pow(10.0f, static_cast<float>(gain_.load(std::memory_order_relaxed) - kReferenceGainDb) / 20.0f);
    const size_t span = noise_.size() - 2 * static_cast<size_t>(kMaxBlock);
    const float* adc_noise = noise_.data() + (static_cast<size_t>(next_random()) * 2) % span;
    const float  adc_sigma = static_cast<float>(kAdcNoiseSigma);
    const float* mix       = mix_.data();
    for (int j = 0; j < 2 * n; ++j) {
        float v = mix[j] * gain + adc_noise[j] * adc_sigma;
        v       = std::clamp(v, -2047.0f, 2047.0f);
        iq[j]   = static_cast<int16_t>(static_cast<int>(std::floor(v + 0.5f)));
    }

    // Pace the stream to real time.
    produced_ += static_cast<uint64_t>(n);
    const auto due = start_time_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                       std::chrono::duration<double>(static_cast<double>(produced_) / sample_rate_));
    const auto now = std::chrono::steady_clock::now();
    if (due > now) {
        std::this_thread::sleep_until(due);
    } else if (now - due > std::chrono::milliseconds(250)) {
        // Fell far behind (slow machine, debugger...): do not try to catch up.
        start_time_ += now - due;
    }
    return n;
}

}  // namespace spectro
