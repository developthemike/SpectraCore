#include "dsp/fm_demod.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>

namespace spectro {

namespace {

constexpr double kPi        = std::numbers::pi;
constexpr double kTwoPi     = 2.0 * std::numbers::pi;
constexpr float  kFullScale = 2048.0f;  // amplitude of a full-scale SC16 Q11 sample

// Stage 1 decimates by a whole number to the lowest rate that is not below
// this.  A fourth-order CIC window then rejects whatever would alias into the
// widest channel of the mode by at least 68 dB.
constexpr double kNarrowChannelRate = 100e3;
constexpr double kWideChannelRate   = 4.0 * FmDemodulator::kWideMaxBandwidth;
constexpr int    kShiftOrder        = 4;

// Stage 2: the channel filter.
constexpr int    kPhases         = 64;    // resampler phases; neighbouring phases are interpolated
constexpr double kStopbandDb     = 70.0;
constexpr double kTransition     = 0.2;   // transition width as a fraction of the bandwidth
constexpr int    kMaxChannelTaps = 4096;
constexpr size_t kBlock          = 4096;  // input samples handled per pass
// The channel, transition band included, must fit into the rates after stage 1
// and at the discriminator.
constexpr double kMaxBandwidthPerRate = 0.78;

// Stage 3: the discriminator runs at the audio rate for narrowband FM.
// Wideband FM needs room for +/-75 kHz of deviation, and its stereo
// subcarrier must be filtered out before the rate comes down to the audio
// rate, so there it runs at a multiple of it.
constexpr int kWideDiscFactor = 8;

// Narrowband full scale: a deviation of this fraction of the channel
// bandwidth.  It makes a +/-2.5 kHz signal in a 12.5 kHz channel and a
// +/-5 kHz signal in a 25 kHz channel come out equally loud.
constexpr double kNarrowFullScaleDeviation = 0.4;

// Stage 4, narrowband: the usual voice processing of a two-way radio.
constexpr double kVoiceDeemphasisS = 750e-6;  // -6 dB per octave across the voice band, 0 dB at 1 kHz
constexpr double kVoiceLowHz       = 300.0;   // also attenuates sub-audible signalling tones
constexpr double kVoiceHighHz      = 3400.0;

// Stage 4, wideband: mono audio (L + R) goes up to 15 kHz.  The stereo pilot
// at 19 kHz and the subcarrier above it must not reach the audio.
constexpr double kWideAudioHz = 15e3;
constexpr double kWidePilotHz = 19e3;
constexpr double kDcBlockHz   = 30.0;

// Squelch: the channel power is judged in blocks of this length.
constexpr double kLevelBlockSeconds = 0.005;
constexpr float  kSquelchHystDb     = 3.0f;           // closes this far below the opening threshold ...
constexpr int    kSquelchHold       = 30;             // ... after this many blocks (150 ms)
constexpr float  kGateAttack        = 1.0f / 240.0f;  // per audio sample at 48 kHz: opens in 5 ms
constexpr float  kGateRelease       = 1.0f / 1440.0f; // closes in 30 ms
constexpr float  kLevelSmoothing    = 0.2f;           // per block: about 25 ms
constexpr float  kAudioSmoothing    = 1.0f / 4800.0f; // per audio sample: 100 ms

double min_channel_rate(FmMode mode)
{
    return mode == FmMode::Wide ? kWideChannelRate : kNarrowChannelRate;
}

int decimation_for(double sample_rate, FmMode mode)
{
    return std::max(1, static_cast<int>(std::floor(sample_rate / min_channel_rate(mode))));
}

int disc_factor_for(FmMode mode)
{
    return mode == FmMode::Wide ? kWideDiscFactor : 1;
}

// ---------------------------------------------------------------------------
// Dot products.  This is where all the time goes, so they are written with
// explicit SIMD vectors (a compiler extension shared by GCC and Clang) and
// enough independent accumulators to keep every multiply-add unit busy.
// ---------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)

#if defined(__AVX__)
#define SPECTRO_VECTOR_BYTES 32
#else
#define SPECTRO_VECTOR_BYTES 16
#endif
typedef float    VecF __attribute__((vector_size(SPECTRO_VECTOR_BYTES)));
constexpr size_t kVecLen = SPECTRO_VECTOR_BYTES / sizeof(float);

inline VecF load(const float* p)
{
    VecF v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline float horizontal_sum(VecF v)
{
    float s[kVecLen];
    std::memcpy(s, &v, sizeof(s));
    float total = 0.0f;
    for (size_t k = 0; k < kVecLen; ++k)
        total += s[k];
    return total;
}

// a = x . ta, b = x . tb
void dot2(const float* x, const float* ta, const float* tb, size_t n, float& a, float& b)
{
    VecF   a0{}, a1{}, a2{}, a3{}, b0{}, b1{}, b2{}, b3{};
    size_t i = 0;
    for (; i + 4 * kVecLen <= n; i += 4 * kVecLen) {
        const VecF x0 = load(x + i), x1 = load(x + i + kVecLen);
        const VecF x2 = load(x + i + 2 * kVecLen), x3 = load(x + i + 3 * kVecLen);
        a0 += x0 * load(ta + i);
        a1 += x1 * load(ta + i + kVecLen);
        a2 += x2 * load(ta + i + 2 * kVecLen);
        a3 += x3 * load(ta + i + 3 * kVecLen);
        b0 += x0 * load(tb + i);
        b1 += x1 * load(tb + i + kVecLen);
        b2 += x2 * load(tb + i + 2 * kVecLen);
        b3 += x3 * load(tb + i + 3 * kVecLen);
    }
    float ra = horizontal_sum((a0 + a1) + (a2 + a3));
    float rb = horizontal_sum((b0 + b1) + (b2 + b3));
    for (; i < n; ++i) {
        ra += x[i] * ta[i];
        rb += x[i] * tb[i];
    }
    a = ra;
    b = rb;
}

float dot(const float* x, const float* t, size_t n)
{
    VecF   a0{}, a1{}, a2{}, a3{};
    size_t i = 0;
    for (; i + 4 * kVecLen <= n; i += 4 * kVecLen) {
        a0 += load(x + i) * load(t + i);
        a1 += load(x + i + kVecLen) * load(t + i + kVecLen);
        a2 += load(x + i + 2 * kVecLen) * load(t + i + 2 * kVecLen);
        a3 += load(x + i + 3 * kVecLen) * load(t + i + 3 * kVecLen);
    }
    float r = horizontal_sum((a0 + a1) + (a2 + a3));
    for (; i < n; ++i)
        r += x[i] * t[i];
    return r;
}

#else  // other compilers: plain loops

void dot2(const float* x, const float* ta, const float* tb, size_t n, float& a, float& b)
{
    float ra = 0.0f, rb = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        ra += x[i] * ta[i];
        rb += x[i] * tb[i];
    }
    a = ra;
    b = rb;
}

float dot(const float* x, const float* t, size_t n)
{
    float r = 0.0f;
    for (size_t i = 0; i < n; ++i)
        r += x[i] * t[i];
    return r;
}

#endif

void convert_to_float(const int16_t* src, float* dst, size_t count)
{
    for (size_t j = 0; j < count; ++j)
        dst[j] = static_cast<float>(src[j]);
}

// Modified Bessel function of the first kind, order zero (for the Kaiser window).
double bessel_i0(double x)
{
    double sum = 1.0, term = 1.0;
    const double q = 0.25 * x * x;
    for (int k = 1; k < 64; ++k) {
        term *= q / (static_cast<double>(k) * k);
        sum += term;
        if (term < 1e-17 * sum)
            break;
    }
    return sum;
}

double sinc(double x)
{
    return std::fabs(x) < 1e-12 ? 1.0 : std::sin(kPi * x) / (kPi * x);
}

double kaiser_beta(double stopband_db)
{
    return 0.1102 * (stopband_db - 8.7);
}

// Taps of a Kaiser-windowed sinc low-pass reaching `stopband_db` of
// attenuation over a transition `transition` wide (cycles per sample).
int kaiser_taps(double transition, double stopband_db)
{
    return static_cast<int>(std::ceil((stopband_db - 7.95) / (14.36 * transition)));
}

// One second-order section of a Butterworth low-pass or high-pass filter.
void design_biquad(bool highpass, double corner_hz, double q, double rate,
                   double& b0, double& b1, double& b2, double& a1, double& a2)
{
    const double w     = kTwoPi * corner_hz / rate;
    const double cs    = std::cos(w);
    const double alpha = std::sin(w) / (2.0 * q);
    const double a0    = 1.0 + alpha;
    if (highpass) {
        b0 = (1.0 + cs) / 2.0 / a0;
        b1 = -(1.0 + cs) / a0;
        b2 = b0;
    } else {
        b0 = (1.0 - cs) / 2.0 / a0;
        b1 = (1.0 - cs) / a0;
        b2 = b0;
    }
    a1 = -2.0 * cs / a0;
    a2 = (1.0 - alpha) / a0;
}

}  // namespace

float FmDemodulator::Biquad::run(float x)
{
    const double in = x;
    const double y  = b0 * in + z1;
    z1 = b1 * in - a1 * y + z2;
    z2 = b2 * in - a2 * y;
    // Without input the state decays towards zero for ever; stop it before it
    // reaches the denormal range, where arithmetic becomes very slow.
    if (std::fabs(z1) < 1e-30)
        z1 = 0.0;
    if (std::fabs(z2) < 1e-30)
        z2 = 0.0;
    return static_cast<float>(y);
}

double FmDemodulator::widest(double sample_rate, double audio_rate, FmMode mode)
{
    const double after_stage1  = sample_rate / decimation_for(sample_rate, mode);
    const double discriminator = audio_rate * disc_factor_for(mode);
    const double mode_max      = mode == FmMode::Wide ? kWideMaxBandwidth : kNarrowMaxBandwidth;
    return std::min({mode_max, kMaxBandwidthPerRate * after_stage1, kMaxBandwidthPerRate * discriminator});
}

bool FmDemodulator::supports(double sample_rate, double audio_rate, FmMode mode)
{
    const double mode_min = mode == FmMode::Wide ? kWideMinBandwidth : kNarrowMinBandwidth;
    return sample_rate >= kMinSampleRate && widest(sample_rate, audio_rate, mode) >= mode_min;
}

double FmDemodulator::usable_bandwidth(double sample_rate, double audio_rate, double bandwidth_hz)
{
    FmMode mode = mode_for(bandwidth_hz);
    if (!supports(sample_rate, audio_rate, mode))
        mode = FmMode::Narrow;
    const double widest_hz = widest(sample_rate, audio_rate, mode);
    const double mode_min  = mode == FmMode::Wide ? kWideMinBandwidth : kNarrowMinBandwidth;
    return std::clamp(bandwidth_hz, std::min(mode_min, widest_hz), widest_hz);
}

FmDemodulator::FmDemodulator(double sample_rate, double audio_rate)
    : sample_rate_(sample_rate), audio_rate_(audio_rate)
{
    if (!(sample_rate_ >= kMinSampleRate))
        throw std::invalid_argument("FM demodulator: the sample rate is too low to hold an FM channel");
    if (!(audio_rate_ >= 8000.0 && audio_rate_ <= 192000.0))
        throw std::invalid_argument("FM demodulator: unsupported audio rate");

    mode_         = FmMode::Narrow;
    bandwidth_hz_ = usable_bandwidth(sample_rate_, audio_rate_, kNarrowDefaultBandwidth);
    configure();
}

void FmDemodulator::set_channel(double offset_hz, double bandwidth_hz, FmMode mode)
{
    if (!supports(sample_rate_, audio_rate_, mode))
        mode = FmMode::Narrow;
    const double widest_hz = widest(sample_rate_, audio_rate_, mode);
    const double mode_min  = mode == FmMode::Wide ? kWideMinBandwidth : kNarrowMinBandwidth;
    bandwidth_hz = std::clamp(bandwidth_hz, std::min(mode_min, widest_hz), widest_hz);

    if (mode != mode_) {
        mode_         = mode;
        offset_hz_    = offset_hz;
        bandwidth_hz_ = bandwidth_hz;
        configure();
        return;
    }
    if (offset_hz != offset_hz_) {
        offset_hz_ = offset_hz;
        design_shift();
    }
    if (bandwidth_hz != bandwidth_hz_) {
        bandwidth_hz_ = bandwidth_hz;
        design_channel_filter();
        update_rates();
    }
}

void FmDemodulator::set_deemphasis(double seconds)
{
    seconds = std::clamp(seconds, 1e-6, 1e-3);
    if (seconds != deemphasis_s_) {
        deemphasis_s_ = seconds;
        design_deemphasis();
    }
}

void FmDemodulator::set_rate_trim(double trim)
{
    trim = std::clamp(trim, -0.02, 0.02);
    if (trim != rate_trim_) {
        rate_trim_ = trim;
        update_rates();
    }
}

// Everything that depends on the mode, followed by a fresh start.
void FmDemodulator::configure()
{
    // --- Stage 1 ---
    decimation_ = decimation_for(sample_rate_, mode_);
    // The window of a CIC decimator: a rectangle `decimation_` samples long,
    // convolved with itself.  Every convolution adds a zero at each multiple
    // of the output rate - precisely the frequencies that fold onto the channel.
    std::vector<double> window{1.0};
    if (decimation_ > 1) {
        for (int pass = 0; pass < kShiftOrder; ++pass) {
            std::vector<double> next(window.size() + static_cast<size_t>(decimation_) - 1, 0.0);
            double              running = 0.0;
            for (size_t n = 0; n < next.size(); ++n) {
                if (n < window.size())
                    running += window[n];
                if (n >= static_cast<size_t>(decimation_))
                    running -= window[n - static_cast<size_t>(decimation_)];
                next[n] = running;
            }
            window.swap(next);
        }
    }
    double window_sum = 0.0;
    for (const double v : window)
        window_sum += v;
    shift_len_ = window.size();
    window_.resize(shift_len_);
    for (size_t k = 0; k < shift_len_; ++k)
        window_[k] = static_cast<float>(window[k] / window_sum);  // unity gain at the channel centre
    shift_re_.assign(2 * shift_len_, 0.0f);
    shift_im_.assign(2 * shift_len_, 0.0f);
    in_.assign(2 * (shift_len_ + kBlock), 0.0f);
    design_shift();

    // --- Stage 2 ---
    // With no decimation a block of input becomes a block of stage 1 output.
    ch_re_.assign(static_cast<size_t>(kMaxChannelTaps) + kBlock + 16, 0.0f);
    ch_im_.assign(ch_re_.size(), 0.0f);
    disc_factor_ = disc_factor_for(mode_);
    design_channel_filter();
    update_rates();

    // --- Stage 4 ---
    design_deemphasis();
    if (mode_ == FmMode::Narrow) {
        // Fourth-order Butterworth filters, as two second-order sections each.
        const double q[2] = {0.5 / std::cos(kPi / 8.0), 0.5 / std::cos(3.0 * kPi / 8.0)};
        for (int s = 0; s < 2; ++s) {
            Biquad& hp = highpass_[s];
            Biquad& lp = lowpass_[s];
            design_biquad(true, kVoiceLowHz, q[s], audio_rate_, hp.b0, hp.b1, hp.b2, hp.a1, hp.a2);
            design_biquad(false, kVoiceHighHz, q[s], audio_rate_, lp.b0, lp.b1, lp.b2, lp.a1, lp.a2);
        }
        audio_taps_.clear();
        wide_buf_.clear();
    } else {
        // A linear-phase low-pass that passes 15 kHz and stops the pilot,
        // computed only for every disc_factor_-th sample.
        const double rate       = discriminator_rate();
        const double transition = (kWidePilotHz - kWideAudioHz) / rate;
        const double cutoff     = 0.5 * (kWidePilotHz + kWideAudioHz) / rate;
        int          taps       = kaiser_taps(transition, kStopbandDb);
        taps |= 1;  // odd: a whole-sample delay
        const double beta = kaiser_beta(kStopbandDb), i0_beta = bessel_i0(beta), half = 0.5 * (taps - 1);
        std::vector<double> h(static_cast<size_t>(taps));
        double              sum = 0.0;
        for (int j = 0; j < taps; ++j) {
            const double t = j - half, u = t / half;
            h[static_cast<size_t>(j)] =
                2.0 * cutoff * sinc(2.0 * cutoff * t) * bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - u * u))) / i0_beta;
            sum += h[static_cast<size_t>(j)];
        }
        audio_taps_.resize(static_cast<size_t>(taps));
        for (int j = 0; j < taps; ++j)
            audio_taps_[static_cast<size_t>(j)] = static_cast<float>(h[static_cast<size_t>(j)] / sum);
        wide_buf_.assign(audio_taps_.size() + 8192, 0.0f);

        design_biquad(true, kDcBlockHz, std::sqrt(0.5), audio_rate_, dc_block_.b0, dc_block_.b1, dc_block_.b2,
                      dc_block_.a1, dc_block_.a2);
    }

    // --- Squelch ---
    level_block_ = std::max(1, static_cast<int>(std::lround(kLevelBlockSeconds * discriminator_rate())));

    reset();
}

void FmDemodulator::update_rates()
{
    const double out_rate = discriminator_rate() * (1.0 + rate_trim_);
    step_          = channel_rate() / out_rate;
    hz_per_radian_ = static_cast<float>(out_rate / kTwoPi);
    audio_per_hz_  = static_cast<float>(mode_ == FmMode::Wide ? 1.0 / kWideDeviation
                                                              : 1.0 / (kNarrowFullScaleDeviation * bandwidth_hz_));
}

void FmDemodulator::design_deemphasis()
{
    if (mode_ == FmMode::Narrow) {
        // A single pole at the audio rate, with the gain set to 1 at 1 kHz.
        const double k  = 1.0 - std::exp(-1.0 / (kVoiceDeemphasisS * audio_rate_));
        const double w  = kTwoPi * 1000.0 / audio_rate_;
        const double re = 1.0 - (1.0 - k) * std::cos(w);
        const double im = (1.0 - k) * std::sin(w);
        deemph_coeff_   = static_cast<float>(k);
        deemph_gain_    = static_cast<float>(std::sqrt(re * re + im * im) / k);
    } else {
        // A single pole at the discriminator rate, unity gain at low frequencies.
        deemph_coeff_ = static_cast<float>(1.0 - std::exp(-1.0 / (deemphasis_s_ * discriminator_rate())));
        deemph_gain_  = 1.0f;
    }
}

// Taps of stage 1: the window times exp(-j w k), which moves the channel to
// 0 Hz.  Filtering x[m D + k] with these taps leaves a rotation of
// exp(-j w m D) per output sample m, which process() takes out again.
void FmDemodulator::design_shift()
{
    const double w = kTwoPi * offset_hz_ / sample_rate_;
    for (size_t k = 0; k < shift_len_; ++k) {
        const double angle = std::remainder(w * static_cast<double>(k), kTwoPi);
        const float  c     = window_[k] * static_cast<float>(std::cos(angle));
        const float  s     = window_[k] * static_cast<float>(std::sin(angle));
        // (i + j q) (c - j s) = (i c + q s) + j (q c - i s)
        shift_re_[2 * k]     = c;
        shift_re_[2 * k + 1] = s;
        shift_im_[2 * k]     = -s;
        shift_im_[2 * k + 1] = c;
    }
    phase_step_ = -std::remainder(w * decimation_, kTwoPi);
}

// A Kaiser-windowed sinc low-pass, sampled at kPhases fractional delays.  Row p
// holds the taps for an output sample p / kPhases of the way between two input
// samples; row kPhases is row 0 moved on by one sample, so that neighbouring
// rows can always be interpolated.
void FmDemodulator::design_channel_filter()
{
    const double rate       = channel_rate();
    const double cutoff     = 0.5 * bandwidth_hz_ / rate;  // cycles per sample, -6 dB
    const double transition = kTransition * bandwidth_hz_ / rate;
    int          taps       = kaiser_taps(transition, kStopbandDb);
    taps = std::clamp(taps + (taps & 1), 8, kMaxChannelTaps);

    const double beta    = kaiser_beta(kStopbandDb);
    const double i0_beta = bessel_i0(beta);
    const double half    = 0.5 * taps;

    phase_taps_.assign(static_cast<size_t>(kPhases + 1) * static_cast<size_t>(taps), 0.0f);
    std::vector<double> row(static_cast<size_t>(taps));
    for (int p = 0; p <= kPhases; ++p) {
        double sum = 0.0;
        for (int j = 0; j < taps; ++j) {
            const double tau    = static_cast<double>(p) / kPhases + half - 1.0 - j;
            const double u      = tau / half;
            const double kaiser = std::fabs(u) < 1.0 ? bessel_i0(beta * std::sqrt(1.0 - u * u)) / i0_beta : 0.0;
            row[static_cast<size_t>(j)] = 2.0 * cutoff * sinc(2.0 * cutoff * tau) * kaiser;
            sum += row[static_cast<size_t>(j)];
        }
        float* dst = phase_taps_.data() + static_cast<size_t>(p) * static_cast<size_t>(taps);
        for (int j = 0; j < taps; ++j)
            dst[j] = static_cast<float>(row[static_cast<size_t>(j)] / sum);  // unity gain at 0 Hz
    }
    channel_taps_ = taps;
}

void FmDemodulator::reset()
{
    in_count_ = 0;
    ch_count_ = 0;
    time_     = 0.0;
    phase_    = 0.0;
    prev_re_ = prev_im_ = 0.0f;
    freq_sum_   = 0.0;
    freq_count_ = 0;
    deemph_state_ = 0.0f;
    for (Biquad& section : highpass_)
        section.clear();
    for (Biquad& section : lowpass_)
        section.clear();
    dc_block_.clear();
    if (!audio_taps_.empty()) {
        std::fill(wide_buf_.begin(), wide_buf_.end(), 0.0f);
        wide_count_ = audio_taps_.size() - 1;  // as if preceded by silence
        wide_phase_ = 0;
    }
    block_power_  = 0.0f;
    block_count_  = 0;
    level_power_  = 0.0f;
    have_level_   = false;
    squelch_open_ = false;
    squelch_hold_ = 0;
    gate_         = 0.0f;
    audio_power_  = 0.0f;
}

FmDemodulator::Stats FmDemodulator::stats() const
{
    Stats s;
    if (have_level_)
        s.level_dbfs = 10.0f * std::log10(level_power_ / (kFullScale * kFullScale) + 1e-20f);
    s.squelch_open = squelch_open_;
    s.audio_rms    = std::sqrt(audio_power_);
    return s;
}

FmDemodulator::FrequencyEstimate FmDemodulator::take_frequency_estimate()
{
    FrequencyEstimate e;
    e.samples = freq_count_;
    e.mean_hz = freq_count_ > 0 ? freq_sum_ / static_cast<double>(freq_count_) : 0.0;
    freq_sum_   = 0.0;
    freq_count_ = 0;
    return e;
}

void FmDemodulator::process(const int16_t* iq, size_t samples, std::vector<float>& audio)
{
    const size_t len  = shift_len_;
    const size_t step = static_cast<size_t>(decimation_);

    while (samples > 0) {
        const size_t n = std::min(samples, kBlock);
        convert_to_float(iq, in_.data() + 2 * in_count_, 2 * n);
        in_count_ += n;
        iq += 2 * n;
        samples -= n;

        // Stage 1: one output sample for every `decimation_` input samples.
        size_t pos = 0;
        while (in_count_ - pos >= len) {
            float re = 0.0f, im = 0.0f;
            dot2(in_.data() + 2 * pos, shift_re_.data(), shift_im_.data(), 2 * len, re, im);
            const float c = static_cast<float>(std::cos(phase_));
            const float s = static_cast<float>(std::sin(phase_));
            ch_re_[ch_count_] = re * c - im * s;
            ch_im_[ch_count_] = re * s + im * c;
            ++ch_count_;
            phase_ += phase_step_;
            if (phase_ > kPi)
                phase_ -= kTwoPi;
            else if (phase_ < -kPi)
                phase_ += kTwoPi;
            pos += step;
        }
        if (pos > 0) {
            std::memmove(in_.data(), in_.data() + 2 * pos, 2 * (in_count_ - pos) * sizeof(float));
            in_count_ -= pos;
        }

        resample_and_demodulate(audio);
    }
}

// Stages 2 to 4, for everything stage 1 has delivered so far.
void FmDemodulator::resample_and_demodulate(std::vector<float>& audio)
{
    const size_t taps = static_cast<size_t>(channel_taps_);
    for (;;) {
        const size_t i = static_cast<size_t>(time_);
        if (i + taps > ch_count_)
            break;
        const double position = (time_ - static_cast<double>(i)) * kPhases;
        const int    p        = std::min(static_cast<int>(position), kPhases - 1);
        const float  blend    = static_cast<float>(position - p);
        const float* h0       = phase_taps_.data() + static_cast<size_t>(p) * taps;
        const float* h1       = h0 + taps;

        float re0 = 0.0f, re1 = 0.0f, im0 = 0.0f, im1 = 0.0f;
        dot2(ch_re_.data() + i, h0, h1, taps, re0, re1);
        dot2(ch_im_.data() + i, h0, h1, taps, im0, im1);
        discriminate(re0 + blend * (re1 - re0), im0 + blend * (im1 - im0), audio);
        time_ += step_;
    }

    // Drop the input that no output sample needs any more.
    const size_t done = std::min(static_cast<size_t>(time_), ch_count_);
    if (done > 0) {
        std::memmove(ch_re_.data(), ch_re_.data() + done, (ch_count_ - done) * sizeof(float));
        std::memmove(ch_im_.data(), ch_im_.data() + done, (ch_count_ - done) * sizeof(float));
        ch_count_ -= done;
        time_ -= static_cast<double>(done);
    }
}

// Stages 3 and 4 for one sample of the channel, at the discriminator rate.
void FmDemodulator::discriminate(float re, float im, std::vector<float>& audio)
{
    // Power in the channel, for the squelch and the level display.
    block_power_ += re * re + im * im;
    if (++block_count_ == level_block_) {
        update_squelch(block_power_ / static_cast<float>(level_block_));
        block_power_ = 0.0f;
        block_count_ = 0;
    }

    // Stage 3: the phase advance since the previous sample is the
    // instantaneous frequency.
    const float dot_product = re * prev_re_ + im * prev_im_;
    const float cross       = im * prev_re_ - re * prev_im_;
    prev_re_ = re;
    prev_im_ = im;
    const float hz = std::atan2(cross, dot_product) * hz_per_radian_;
    freq_sum_ += static_cast<double>(hz);
    ++freq_count_;

    // Stage 4.
    const float a = hz * audio_per_hz_;
    deemph_state_ += deemph_coeff_ * (a - deemph_state_);
    if (mode_ == FmMode::Narrow) {
        float v = deemph_state_ * deemph_gain_;
        for (Biquad& section : highpass_)
            v = section.run(v);
        for (Biquad& section : lowpass_)
            v = section.run(v);
        emit(v, audio);
        return;
    }

    // Wideband: low-pass, and keep every disc_factor_-th sample.
    if (wide_count_ == wide_buf_.size()) {
        const size_t keep = audio_taps_.size() - 1;
        std::memmove(wide_buf_.data(), wide_buf_.data() + wide_count_ - keep, keep * sizeof(float));
        wide_count_ = keep;
    }
    wide_buf_[wide_count_++] = deemph_state_;
    if (++wide_phase_ < disc_factor_)
        return;
    wide_phase_ = 0;
    const float v = dot(wide_buf_.data() + wide_count_ - audio_taps_.size(), audio_taps_.data(), audio_taps_.size());
    emit(dc_block_.run(v), audio);
}

// The squelch, with short ramps so that it does not click, then out.
void FmDemodulator::emit(float sample, std::vector<float>& audio)
{
    const float target = squelch_open_ ? 1.0f : 0.0f;
    if (gate_ < target)
        gate_ = std::min(target, gate_ + kGateAttack);
    else if (gate_ > target)
        gate_ = std::max(target, gate_ - kGateRelease);
    sample *= gate_;

    audio_power_ += kAudioSmoothing * (sample * sample - audio_power_);
    audio.push_back(sample);
}

void FmDemodulator::update_squelch(float mean_power)
{
    level_power_ = have_level_ ? level_power_ + kLevelSmoothing * (mean_power - level_power_) : mean_power;
    have_level_  = true;

    if (squelch_dbfs_ <= kSquelchOff) {
        squelch_open_ = true;
        return;
    }
    const float db = 10.0f * std::log10(mean_power / (kFullScale * kFullScale) + 1e-20f);
    if (db >= squelch_dbfs_) {
        squelch_open_ = true;
        squelch_hold_ = kSquelchHold;
    } else if (db < squelch_dbfs_ - kSquelchHystDb) {
        if (squelch_hold_ > 0)
            --squelch_hold_;
        else
            squelch_open_ = false;
    }
}

}  // namespace spectro
