#include "radio/sim_source.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>
#include <thread>

#include <fftw3.h>

#include "dsp/fftw_lock.hpp"

namespace spectro {

namespace {

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

}  // namespace

std::vector<SimSource::Carrier> SimSource::carriers(double sample_rate)
{
    return {{-0.36 * sample_rate, -30.0}, {0.43 * sample_rate, -55.0}};
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
