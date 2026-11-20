#include "dsp/signal_classifier.hpp"

#include <algorithm>
#include <bit>
#include <functional>
#include <utility>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <mutex>
#include <numbers>

#include <fftw3.h>

#include "dsp/fftw_lock.hpp"

namespace spectro {

namespace {

constexpr double kPi  = std::numbers::pi;
constexpr int    kFft = 1 << 15;  // spectra: 1.46 Hz bins at 48 kHz

// A spectral line counts when it stands this far (power ratio) above the
// spectrum around it.  A symbol-rate line need not stand out as far: the
// symbols must then fall into clean levels as well, which noise and speech
// never do.
constexpr double kLineProminence   = 25.0;
constexpr double kSymbolProminence = 12.0;

// Symbol rates in common use, lowest first: a signal at one of them also
// shows lines at its multiples, so the lowest line is its rate.
constexpr double kSymbolRates[] = {512, 1200, 1600, 2400, 3200, 3600, 4800, 9600};

// Analog: less audio than this (RMS deviation, Hz) above the noise is no audio.
constexpr double kMinAudioHz = 150.0;
// CTCSS: the standard tones, and the least deviation (RMS, Hz) that counts.
constexpr double kCtcssTones[] = {67.0,  69.3,  71.9,  74.4,  77.0,  79.7,  82.5,  85.4,  88.5,  91.5,
                                  94.8,  97.4,  100.0, 103.5, 107.2, 110.9, 114.8, 118.8, 123.0, 127.3,
                                  131.8, 136.5, 141.3, 146.2, 151.4, 156.7, 159.8, 162.2, 165.5, 167.9,
                                  171.3, 173.8, 177.3, 179.9, 183.5, 186.2, 189.9, 192.8, 196.6, 199.5,
                                  203.5, 206.5, 210.7, 218.1, 225.7, 229.1, 233.6, 241.8, 250.3, 254.1};
constexpr double kMinCtcssHz   = 60.0;

// Sync patterns of 4FSK systems, one symbol per character: '1' is +3 (the
// highest frequency), '3' is -3.  Both use only the outermost levels.
struct SyncPattern {
    const char* system;
    const char* symbols;
};
constexpr SyncPattern kFsk4Syncs[] = {
    {"DMR", "131111333113313313113313"},          // base station, voice
    {"DMR", "313333111331131131331131"},          // base station, data
    {"DMR", "133313311131311113313331"},          // mobile, voice
    {"DMR", "311131133313133331131113"},          // mobile, data
    {"P25 Phase 1", "111113113311333313133333"},  // frame sync 0x5575F5FF77FF
};
constexpr uint32_t kPocsagSync = 0x7CD215D8;

bool near_rate(double rate, double standard) { return std::fabs(rate - standard) <= 0.01 * standard; }

double snap_rate(double rate)
{
    for (const double standard : kSymbolRates) {
        if (near_rate(rate, standard))
            return standard;
    }
    return std::round(rate);
}

// Modified Bessel function of the first kind, order zero (Kaiser window).
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

// A Kaiser-windowed low-pass, 60 dB down, unity gain at 0 Hz.
std::vector<float> lowpass(double cutoff, double transition)
{
    const int    taps = static_cast<int>(std::ceil((60.0 - 7.95) / (14.36 * transition))) | 1;
    const double beta = 0.1102 * (60.0 - 8.7), i0 = bessel_i0(beta), half = 0.5 * (taps - 1);
    std::vector<float> h(static_cast<size_t>(taps));
    double             sum = 0.0;
    for (int j = 0; j < taps; ++j) {
        const double t = j - half, u = t / half;
        const double s = std::fabs(t) < 1e-12 ? 1.0 : std::sin(2.0 * kPi * cutoff * t) / (2.0 * kPi * cutoff * t);
        h[static_cast<size_t>(j)] =
            static_cast<float>(2.0 * cutoff * s * bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - u * u))) / i0);
        sum += h[static_cast<size_t>(j)];
    }
    for (float& t : h)
        t = static_cast<float>(t / sum);
    return h;
}

// Linear interpolation of x at a fractional index.
float sample_at(const std::vector<float>& x, double t)
{
    const size_t i = static_cast<size_t>(t);
    if (i + 1 >= x.size())
        return x.back();
    const float f = static_cast<float>(t - static_cast<double>(i));
    return x[i] + f * (x[i + 1] - x[i]);
}

// One-dimensional k-means, starting from `centres`; returns the mean squared
// distance and leaves the centres sorted.
double kmeans(const std::vector<float>& values, std::vector<double>& centres, std::vector<size_t>& counts)
{
    const size_t k = centres.size();
    std::vector<double> sums(k);
    double              distortion = 0.0;
    for (int iteration = 0; iteration < 20; ++iteration) {
        std::fill(sums.begin(), sums.end(), 0.0);
        counts.assign(k, 0);
        distortion = 0.0;
        for (const float v : values) {
            size_t best = 0;
            for (size_t c = 1; c < k; ++c) {
                if (std::fabs(v - centres[c]) < std::fabs(v - centres[best]))
                    best = c;
            }
            sums[best] += v;
            ++counts[best];
            distortion += (v - centres[best]) * (v - centres[best]);
        }
        for (size_t c = 0; c < k; ++c) {
            if (counts[c] > 0)
                centres[c] = sums[c] / static_cast<double>(counts[c]);
        }
    }
    std::sort(centres.begin(), centres.end());
    return values.empty() ? 0.0 : distortion / static_cast<double>(values.size());
}

// Root-raised-cosine taps, `span` symbols either side, unity gain at 0 Hz: the
// matched filter of 4FSK systems such as DMR, and a fair low-pass for any FSK.
std::vector<float> rrc_taps(double sps, double alpha, int span)
{
    const int          half = static_cast<int>(std::ceil(span * sps));
    std::vector<float> taps(static_cast<size_t>(2 * half + 1));
    double             sum = 0.0;
    for (int i = -half; i <= half; ++i) {
        const double t = i / sps;  // in symbols
        double       h;
        if (std::fabs(t) < 1e-9) {
            h = 1.0 - alpha + 4.0 * alpha / kPi;
        } else if (std::fabs(std::fabs(t) - 1.0 / (4.0 * alpha)) < 1e-9) {
            h = alpha / std::sqrt(2.0) *
                ((1.0 + 2.0 / kPi) * std::sin(kPi / (4.0 * alpha)) + (1.0 - 2.0 / kPi) * std::cos(kPi / (4.0 * alpha)));
        } else {
            h = (std::sin(kPi * t * (1.0 - alpha)) + 4.0 * alpha * t * std::cos(kPi * t * (1.0 + alpha))) /
                (kPi * t * (1.0 - 16.0 * alpha * alpha * t * t));
        }
        taps[static_cast<size_t>(i + half)] = static_cast<float>(h);
        sum += h;
    }
    for (float& t : taps)
        t = static_cast<float>(t / sum);
    return taps;
}

std::string format_rate(double rate)
{
    char text[32];
    std::snprintf(text, sizeof(text), "%.0f Bd", rate);
    return text;
}

}  // namespace

// ---------------------------------------------------------------------------
// Descriptions
// ---------------------------------------------------------------------------

std::string describe(const SignalClass& s)
{
    char text[160];
    const auto ctcss = [&]() -> std::string {
        if (s.ctcss_hz <= 0.0)
            return {};
        char tone[32];
        std::snprintf(tone, sizeof(tone), ", CTCSS %.1f Hz", s.ctcss_hz);
        return tone;
    };
    switch (s.kind) {
    case SignalKind::Analysing: return "Analysing...";
    case SignalKind::Noise:     return "No signal (noise)";
    case SignalKind::Weak:      return "Weak signal: too weak to tell";
    case SignalKind::Carrier:   return "Carrier without audio" + ctcss();
    case SignalKind::Analog:    return "Analog voice or audio" + ctcss();
    case SignalKind::Broadcast: return s.stereo ? "Broadcast FM, stereo" : "Broadcast FM, mono";
    case SignalKind::Afsk:      return "Digital: AFSK, 1200 Bd (Bell 202 tones)";
    case SignalKind::Dqpsk:
        std::snprintf(text, sizeof(text), "Digital: DQPSK, %s%s%s", format_rate(s.symbol_rate).c_str(),
                      s.protocol.empty() ? "" : ", ", s.protocol.c_str());
        return text;
    case SignalKind::Fsk2:
    case SignalKind::Gmsk:
    case SignalKind::Fsk4: {
        const char* name = s.kind == SignalKind::Fsk4 ? "4FSK" : s.kind == SignalKind::Gmsk ? "GMSK" : "2FSK";
        std::snprintf(text, sizeof(text), "Digital: %s, %s, \xC2\xB1%.2f kHz%s%s", name,
                      format_rate(s.symbol_rate).c_str(), s.deviation_hz / 1e3, s.protocol.empty() ? "" : ", ",
                      s.protocol.c_str());
        return text;
    }
    }
    return {};
}

std::string short_name(const SignalClass& s)
{
    if (!s.protocol.empty())
        return s.protocol;
    switch (s.kind) {
    case SignalKind::Carrier:   return "carrier";
    case SignalKind::Analog:    return "voice";
    case SignalKind::Broadcast: return s.stereo ? "stereo" : "mono";
    case SignalKind::Afsk:      return "AFSK";
    case SignalKind::Fsk2:      return "2FSK";
    case SignalKind::Gmsk:      return "GMSK";
    case SignalKind::Fsk4:      return "4FSK";
    case SignalKind::Dqpsk:     return "DQPSK";
    default:                    return {};
    }
}

std::string typical_systems(const SignalClass& s)
{
    if (!s.protocol.empty())
        return {};
    const auto at = [&](double rate) { return near_rate(s.symbol_rate, rate); };
    switch (s.kind) {
    case SignalKind::Afsk:
        return "APRS and other AX.25 packet radio";
    case SignalKind::Fsk4:
        if (at(4800))
            return "DMR, P25 Phase 1, NXDN or System Fusion";
        if (at(2400))
            return "NXDN or dPMR";
        if (at(1600) || at(3200))
            return "FLEX paging";
        return {};
    case SignalKind::Fsk2:
        if ((at(512) || at(1200) || at(2400)) && s.deviation_hz > 3000.0)
            return "POCSAG paging";
        if (at(1600))
            return "FLEX paging";
        return {};
    case SignalKind::Gmsk:
        if (at(4800))
            return "D-STAR";
        if (at(9600))
            return "AIS or 9600 Bd packet radio";
        return {};
    default:
        return {};
    }
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

SignalClassifier::SignalClassifier(double sample_rate) : sample_rate_(sample_rate)
{
    window_ = static_cast<size_t>(std::lround(kWindowSeconds * sample_rate_));
    window_ = std::min<size_t>(window_, kFft);
    frequency_.reserve(window_);
    power_.reserve(window_);

    // A Hann window over the analysis window; spectra come out as variance
    // per bin (Hz^2), so that a band's sum is its share of the RMS deviation.
    fft_window_.resize(window_);
    double energy = 0.0;
    for (size_t i = 0; i < window_; ++i) {
        fft_window_[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * (static_cast<double>(i) + 0.5) / static_cast<double>(window_)));
        energy += static_cast<double>(fft_window_[i]) * fft_window_[i];
    }
    fft_scale_ = 2.0 / (static_cast<double>(kFft) * energy);

    fft_in_  = fftwf_alloc_real(kFft);
    fft_out_ = fftwf_alloc_complex(kFft / 2 + 1);
    {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        plan_ = fftwf_plan_dft_r2c_1d(kFft, fft_in_, static_cast<fftwf_complex*>(fft_out_), FFTW_ESTIMATE);
    }

    // Low-passes for the digital analysis, each about 0.7 x the highest
    // symbol rate it serves.
    const double fs = sample_rate_;
    bands_.push_back({1200.0, lowpass(1000.0 / fs, 600.0 / fs), {}});
    bands_.push_back({2400.0, lowpass(1800.0 / fs, 1000.0 / fs), {}});
    bands_.push_back({4800.0, lowpass(3400.0 / fs, 1500.0 / fs), {}});
    bands_.push_back({9600.0, lowpass(6000.0 / fs, 2000.0 / fs), {}});
    lowpass_taps_ = bands_.back().taps;
}

// out = in filtered with the symmetric `taps`, without delay (zero-padded at
// the ends).  Written as plain dot products so that it vectorises.
void SignalClassifier::filter(const std::vector<float>& taps, const std::vector<float>& in, std::vector<float>& out)
{
    const size_t n = in.size(), m = taps.size(), half = m / 2;
    padded_.assign(n + 2 * half, 0.0f);
    std::copy(in.begin(), in.end(), padded_.begin() + static_cast<long>(half));
    out.resize(n);
    const float* t = taps.data();
    for (size_t i = 0; i < n; ++i) {
        const float* x   = padded_.data() + i;
        float        acc = 0.0f;
        for (size_t j = 0; j < m; ++j)
            acc += t[j] * x[j];
        out[i] = acc;
    }
}

// Spectrum of the squared derivative: the symbol transitions.
void SignalClassifier::transition_spectrum(const std::vector<float>& lowpassed, std::vector<float>& out)
{
    const size_t n = lowpassed.size();
    transitions_.resize(n);
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float d   = i > 0 ? lowpassed[i] - lowpassed[i - 1] : 0.0f;
        transitions_[i] = d * d;
        mean += transitions_[i];
    }
    mean /= static_cast<double>(n);
    for (float& t : transitions_)
        t = static_cast<float>(t - mean);
    spectrum(transitions_, out);
}

// AFSK: two tones, 1200 and 2200 Hz, keyed at 1200 Bd.  At that rate each
// tone lasts too short to make a narrow line, so compare the energy near the
// two over one bit at a time: for AFSK one or the other clearly dominates,
// and which one changes at 1200 Bd.
bool SignalClassifier::afsk()
{
    // The audio of AFSK is its tones; plain FSK holds steady levels for whole
    // symbols, which puts its power low down instead.
    if (band(freq_spec_, 900.0, 2500.0) < 3.0 * band(freq_spec_, 50.0, 700.0))
        return false;

    const size_t        n   = centred_.size();
    const size_t        bit = static_cast<size_t>(std::lround(sample_rate_ / 1200.0));
    const double        w1 = 2.0 * kPi * 1200.0 / sample_rate_, w2 = 2.0 * kPi * 2200.0 / sample_rate_;
    double              r1 = 0, i1 = 0, r2 = 0, i2 = 0;  // running sums over one bit
    std::vector<double> m1r(n), m1i(n), m2r(n), m2i(n);
    contrast_.assign(n, 0.0f);
    double clear = 0.0;
    size_t marks = 0, spaces = 0;
    for (size_t k = 0; k < n; ++k) {
        const double x = centred_[k];
        const double t = static_cast<double>(k);
        m1r[k]         = x * std::cos(w1 * t);
        m1i[k]         = -x * std::sin(w1 * t);
        m2r[k]         = x * std::cos(w2 * t);
        m2i[k]         = -x * std::sin(w2 * t);
        r1 += m1r[k];
        i1 += m1i[k];
        r2 += m2r[k];
        i2 += m2i[k];
        if (k >= bit) {
            r1 -= m1r[k - bit];
            i1 -= m1i[k - bit];
            r2 -= m2r[k - bit];
            i2 -= m2i[k - bit];
        }
        const double e1 = r1 * r1 + i1 * i1, e2 = r2 * r2 + i2 * i2;
        contrast_[k]    = static_cast<float>(e1 + e2 > 0.0 ? (e1 - e2) / (e1 + e2) : 0.0);
        clear += std::fabs(contrast_[k]);
        marks += contrast_[k] > 0.3f;
        spaces += contrast_[k] < -0.3f;
    }
    clear /= static_cast<double>(n);
    // Both tones in use: plain FSK at 1200 Bd has a null at 1200 Hz and
    // would otherwise look like one tone throughout.
    if (clear < 0.45 || marks < n / 5 || spaces < n / 5)
        return false;
    transition_spectrum(contrast_, afsk_spec_);
    return line(afsk_spec_, 1200.0, 12.0).prominence >= kLineProminence;
}

SignalClassifier::~SignalClassifier()
{
    std::lock_guard<std::mutex> lock(fftw_planner_mutex());
    if (plan_)
        fftwf_destroy_plan(static_cast<fftwf_plan>(plan_));
    fftwf_free(fft_in_);
    fftwf_free(fft_out_);
}

void SignalClassifier::set_channel(FmMode mode, double bandwidth_hz)
{
    if (mode != mode_ || bandwidth_hz != bandwidth_) {
        mode_      = mode;
        bandwidth_ = bandwidth_hz;
        reset();
    }
}

void SignalClassifier::reset()
{
    frequency_.clear();
    power_.clear();
    result_          = {};
    previous_        = {};
    last_ctcss_      = 0.0;
    recent_protocol_.clear();
    protocol_age_    = 1000;
}

void SignalClassifier::process(const float* frequency, const float* power, size_t count)
{
    while (count > 0) {
        const size_t n = std::min(count, window_ - frequency_.size());
        frequency_.insert(frequency_.end(), frequency, frequency + n);
        power_.insert(power_.end(), power, power + n);
        frequency += n;
        power += n;
        count -= n;
        if (frequency_.size() == window_) {
            analyse();
            frequency_.clear();
            power_.clear();
        }
    }
}

// Power spectrum (Hz^2 per bin) of a window's worth of samples.
void SignalClassifier::spectrum(const std::vector<float>& x, std::vector<float>& out)
{
    for (size_t i = 0; i < window_; ++i)
        fft_in_[i] = x[i] * fft_window_[i];
    std::fill(fft_in_ + window_, fft_in_ + kFft, 0.0f);
    fftwf_execute(static_cast<fftwf_plan>(plan_));
    const fftwf_complex* X = static_cast<const fftwf_complex*>(fft_out_);
    out.resize(kFft / 2 + 1);
    for (size_t k = 0; k < out.size(); ++k)
        out[k] = static_cast<float>((static_cast<double>(X[k][0]) * X[k][0] + static_cast<double>(X[k][1]) * X[k][1]) *
                                    fft_scale_);
}

double SignalClassifier::band(const std::vector<float>& spec, double low, double high) const
{
    const double bin = sample_rate_ / kFft;
    const size_t a   = static_cast<size_t>(std::max(0.0, std::ceil(low / bin)));
    const size_t b   = std::min(spec.size() - 1, static_cast<size_t>(high / bin));
    double       sum = 0.0;
    for (size_t k = a; k <= b; ++k)
        sum += spec[k];
    return sum;
}

// The strongest line within `tolerance` of `frequency`, and how far it stands
// out from the spectrum within 300 Hz of it.
SignalClassifier::Line SignalClassifier::line(const std::vector<float>& spec, double frequency, double tolerance) const
{
    Line         result;
    const double bin = sample_rate_ / kFft;
    const long   last = static_cast<long>(spec.size()) - 2;
    const long   lo   = std::max(1L, static_cast<long>(std::floor((frequency - tolerance) / bin)));
    const long   hi   = std::min(last, static_cast<long>(std::ceil((frequency + tolerance) / bin)));
    if (lo > hi)
        return result;
    long peak = lo;
    for (long k = lo; k <= hi; ++k) {
        if (spec[static_cast<size_t>(k)] > spec[static_cast<size_t>(peak)])
            peak = k;
    }
    result.peak = peak > lo && peak < hi && spec[static_cast<size_t>(peak)] >= spec[static_cast<size_t>(peak - 1)] &&
                  spec[static_cast<size_t>(peak)] >= spec[static_cast<size_t>(peak + 1)];
    // The line itself: its main lobe, a few bins either side.
    for (long k = std::max(1L, peak - 4); k <= std::min(last, peak + 4); ++k)
        result.power += spec[static_cast<size_t>(k)];

    std::vector<float> around;
    const long reach = static_cast<long>(300.0 / bin), gap = static_cast<long>(30.0 / bin);
    for (long k = std::max(1L, peak - reach); k <= std::min(last, peak + reach); ++k) {
        if (std::labs(k - peak) > gap)
            around.push_back(spec[static_cast<size_t>(k)]);
    }
    if (around.empty())
        return result;
    std::nth_element(around.begin(), around.begin() + static_cast<long>(around.size() / 2), around.end());
    const double background = std::max(static_cast<double>(around[around.size() / 2]), 1e-30);
    result.prominence       = spec[static_cast<size_t>(peak)] / background;

    // Refine the frequency between bins.
    const double a = std::sqrt(spec[static_cast<size_t>(peak - 1)]), b = std::sqrt(spec[static_cast<size_t>(peak)]),
                 c = std::sqrt(spec[static_cast<size_t>(peak + 1)]);
    const double d = a - 2.0 * b + c;
    result.frequency = (static_cast<double>(peak) + (std::fabs(d) > 1e-30 ? 0.5 * (a - c) / d : 0.0)) * bin;
    return result;
}

// Whether the frequency itself holds a strong tone there - which would
// explain a line at twice that frequency among the transitions.
bool SignalClassifier::tone_at(double frequency) const
{
    const Line tone  = line(freq_spec_, frequency, std::max(3.0, 0.01 * frequency));
    const double all = band(freq_spec_, 100.0, 6000.0);
    return tone.prominence >= kLineProminence && tone.power >= 0.2 * all;
}

// The share of the frequency's power (100 Hz to 6 kHz) in its five strongest
// lines.  Data has a continuous spectrum; tones and tunes put nearly all
// their power into a few lines.
double SignalClassifier::tonal_share() const
{
    const double bin   = sample_rate_ / kFft;
    const size_t first = static_cast<size_t>(100.0 / bin), last = static_cast<size_t>(6000.0 / bin);
    std::vector<std::pair<float, size_t>> peaks;
    for (size_t k = first + 1; k < last; ++k) {
        if (freq_spec_[k] > freq_spec_[k - 1] && freq_spec_[k] >= freq_spec_[k + 1])
            peaks.emplace_back(freq_spec_[k], k);
    }
    const size_t top = std::min<size_t>(5, peaks.size());
    std::partial_sort(peaks.begin(), peaks.begin() + static_cast<long>(top), peaks.end(), std::greater<>());
    double in_lines = 0.0;
    for (size_t i = 0; i < top; ++i) {
        for (size_t k = peaks[i].second - std::min<size_t>(4, peaks[i].second); k <= peaks[i].second + 4; ++k)
            in_lines += freq_spec_[k];
    }
    const double all = band(freq_spec_, 100.0, 6000.0);
    return all > 0.0 ? in_lines / all : 0.0;
}

void SignalClassifier::analyse()
{
    ++windows_;
    SignalClass verdict = classify();

    // A protocol seen in one window still names the signal for a few more:
    // its sync pattern need not be in every window.
    if (!verdict.protocol.empty()) {
        recent_protocol_ = verdict.protocol;
        protocol_age_    = 0;
    } else if (++protocol_age_ <= 4 && verdict.kind == previous_.kind && verdict.digital()) {
        verdict.protocol = recent_protocol_;
    }

    // A sub-audible tone must be the same in two windows: speech has none.
    const double tone = verdict.ctcss_hz;
    if (std::fabs(tone - last_ctcss_) > 1.5)
        verdict.ctcss_hz = 0.0;
    last_ctcss_ = tone;

    // Two windows in a row must agree before the verdict changes.
    if (verdict.kind == previous_.kind)
        result_ = verdict;
    previous_ = verdict;
}

SignalClass SignalClassifier::classify()
{
    SignalClass r;
    const size_t n = window_;

    // Signal-to-noise ratio from the second and fourth moments of the
    // power: a constant envelope S in Gaussian noise N gives E[p] = S + N and
    // E[p^2] = S^2 + 4 S N + 2 N^2.
    double m2 = 0.0, m4 = 0.0;
    for (const float p : power_) {
        m2 += p;
        m4 += static_cast<double>(p) * p;
    }
    m2 /= static_cast<double>(n);
    m4 /= static_cast<double>(n);
    const double s     = std::sqrt(std::max(0.0, 2.0 * m2 * m2 - m4));
    const double noise = std::max(m2 - s, 1e-30 * m2 + 1e-30);
    r.snr_db           = static_cast<float>(std::clamp(10.0 * std::log10(std::max(s, 1e-30) / noise), -20.0, 60.0));
    if (r.snr_db < kMinSnrDb) {
        r.kind = SignalKind::Noise;
        return r;
    }

    // The frequency about its mean (the carrier offset).
    double mean = 0.0;
    for (const float f : frequency_)
        mean += f;
    mean /= static_cast<double>(n);
    centred_.resize(n);
    for (size_t i = 0; i < n; ++i)
        centred_[i] = static_cast<float>(frequency_[i] - mean);
    spectrum(centred_, freq_spec_);

    if (mode_ == FmMode::Wide) {
        r.kind   = SignalKind::Broadcast;
        r.stereo = line(freq_spec_, 19000.0, 30.0).prominence >= kLineProminence;
        return r;
    }

    const double audio = band(freq_spec_, 300.0, 3000.0);
    if (afsk()) {
        r.kind        = SignalKind::Afsk;
        r.symbol_rate = 1200.0;
        return r;
    }

    // Symbol transitions: the squared derivative of the frequency, low-passed
    // to suit the rate looked for (each band worked out when first needed).
    for (RateBand& b : bands_)
        b.ready = false;
    lowpassed_.clear();
    const auto transitions_for = [&](double rate) -> const std::vector<float>& {
        RateBand* b = &bands_.back();
        for (RateBand& candidate : bands_) {
            if (rate <= candidate.max_rate) {
                b = &candidate;
                break;
            }
        }
        if (!b->ready) {
            filter(b->taps, centred_, filtered_);
            transition_spectrum(filtered_, b->spectrum);
            b->ready = true;
        }
        return b->spectrum;
    };

    // The lowest common symbol rate with a clear line - unless a tone at half
    // that frequency explains it.
    const double bin   = sample_rate_ / kFft;
    const bool   tonal = tonal_share() > 0.5;
    for (const double rate : kSymbolRates) {
        if (rate > 0.4 * sample_rate_ || tonal)
            break;
        const Line l = line(transitions_for(rate), rate, std::max(0.01 * rate, 4.0 * bin));
        if (l.peak && l.prominence >= kSymbolProminence && !tone_at(0.5 * l.frequency) &&
            classify_digital(l.frequency, r))
            return r;
    }

    // Not data.  Too weak to tell analog audio from the noise?
    if (r.snr_db < kMinAnalogSnrDb) {
        r.kind = SignalKind::Weak;
        return r;
    }

    // Analog.  A sub-audible tone first.
    // Transmitters keep voice out of the range below 300 Hz to leave it to the
    // tone, so a real one dominates there.
    const Line ctcss = line(freq_spec_, 160.0, 100.0);
    if (ctcss.prominence >= 20.0 && std::sqrt(ctcss.power) >= kMinCtcssHz &&
        ctcss.power >= 0.5 * band(freq_spec_, 50.0, 300.0)) {
        r.ctcss_hz = ctcss.frequency;
        for (const double tone : kCtcssTones) {
            if (std::fabs(tone - ctcss.frequency) <= 1.5)
                r.ctcss_hz = tone;
        }
    }
    // Then audio: the voice band less the noise there.  The discriminator's
    // noise rises as the square of the frequency, so its level above the
    // voice band, where a transmitter puts no audio, gives it.
    double       noise_in_band = 0.0;
    const double high_lo = 3400.0, high_hi = std::min(0.45 * bandwidth_, 6000.0);
    if (high_hi - high_lo >= 500.0) {
        double ratio = 0.0;
        size_t count = 0;
        for (size_t k = static_cast<size_t>(high_lo / bin); k <= static_cast<size_t>(high_hi / bin); ++k) {
            const double f = static_cast<double>(k) * bin;
            ratio += freq_spec_[k] / (f * f);
            ++count;
        }
        ratio /= static_cast<double>(std::max<size_t>(count, 1));
        for (size_t k = static_cast<size_t>(300.0 / bin); k <= static_cast<size_t>(3000.0 / bin); ++k)
            noise_in_band += ratio * (static_cast<double>(k) * bin) * (static_cast<double>(k) * bin);
    }
    r.kind = audio - noise_in_band >= kMinAudioHz * kMinAudioHz ? SignalKind::Analog : SignalKind::Carrier;
    return r;
}

// Recovers the symbols at `rate` and tells the modulation from their levels.
// False if they do not fall into clear levels after all.
bool SignalClassifier::classify_digital(double rate, SignalClass& out)
{
    const double sps = sample_rate_ / rate;
    if (sps < 3.0)
        return false;

    // Matched filter, then the timing at which the symbols are strongest:
    // their centres.
    const std::vector<float> taps = rrc_taps(sps, 0.2, 6);
    const size_t             n = centred_.size(), half = taps.size() / 2;
    if (n < 4 * taps.size())
        return false;
    smoothed_.assign(n, 0.0f);
    for (size_t i = half; i + half < n; ++i) {
        float acc = 0.0f;
        for (size_t j = 0; j < taps.size(); ++j)
            acc += taps[j] * centred_[i + half - j];
        smoothed_[i] = acc;
    }
    const double first = static_cast<double>(half), last = static_cast<double>(n - half - 1);
    double       best_phase = 0.0, best_energy = -1.0;
    for (double phase = 0.0; phase < sps; phase += 0.25) {
        double energy = 0.0;
        size_t count  = 0;
        for (double t = first + phase; t < last; t += sps) {
            const float v = sample_at(smoothed_, t);
            energy += static_cast<double>(v) * v;
            ++count;
        }
        if (count > 0 && energy / static_cast<double>(count) > best_energy) {
            best_energy = energy / static_cast<double>(count);
            best_phase  = phase;
        }
    }
    symbols_.clear();
    for (double t = first + best_phase; t < last; t += sps)
        symbols_.push_back(sample_at(smoothed_, t));
    if (symbols_.size() < 32)
        return false;

    // Two levels or four?  The clusters start at the quantiles of the symbols,
    // so that an offset of the carrier does not matter.
    std::vector<float> sorted = symbols_;
    std::sort(sorted.begin(), sorted.end());
    const auto quantile = [&](double q) {
        return static_cast<double>(sorted[static_cast<size_t>(q * static_cast<double>(sorted.size() - 1))]);
    };
    std::vector<size_t> counts2, counts4;
    std::vector<double> two{quantile(0.25), quantile(0.75)};
    std::vector<double> four{quantile(0.125), quantile(0.375), quantile(0.625), quantile(0.875)};
    const double        d2 = kmeans(symbols_, two, counts2);
    const double        d4 = kmeans(symbols_, four, counts4);

    // Four levels: evenly spaced, each in use, and much tighter than two.
    const double g1 = four[1] - four[0], g2 = four[2] - four[1], g3 = four[3] - four[2];
    const double gap_min = std::min({g1, g2, g3}), gap_max = std::max({g1, g2, g3});
    const size_t least4  = *std::min_element(counts4.begin(), counts4.end());
    const bool   four_levels = gap_min > 0.0 && gap_max <= 1.8 * gap_min && d4 < 0.25 * d2 &&
                             least4 >= symbols_.size() / 16 && std::sqrt(d4) <= 0.3 * gap_min;
    // Two levels: well apart and both in use.
    const double spacing2   = two[1] - two[0];
    const size_t least2     = std::min(counts2[0], counts2[1]);
    const bool   two_levels = spacing2 > 0.0 && std::sqrt(d2) <= 0.3 * spacing2 && least2 >= symbols_.size() / 8;
    if (!four_levels && !two_levels)
        return false;

    out.symbol_rate = snap_rate(rate);
    if (four_levels) {
        out.kind         = SignalKind::Fsk4;
        out.deviation_hz = 0.5 * (four[3] - four[0]);
        // Sync patterns of 4FSK systems at 4800 Bd.
        if (near_rate(rate, 4800.0)) {
            std::vector<int> levels(symbols_.size());
            for (size_t i = 0; i < symbols_.size(); ++i) {
                int best = 0;
                for (int c = 1; c < 4; ++c) {
                    if (std::fabs(symbols_[i] - four[static_cast<size_t>(c)]) <
                        std::fabs(symbols_[i] - four[static_cast<size_t>(best)]))
                        best = c;
                }
                levels[i] = 2 * best - 3;  // -3, -1, +1, +3
            }
            for (const SyncPattern& sync : kFsk4Syncs) {
                const size_t length = std::char_traits<char>::length(sync.symbols);
                for (size_t start = 0; start + length <= levels.size() && out.protocol.empty(); ++start) {
                    int same = 0, inverted = 0;
                    for (size_t j = 0; j < length; ++j) {
                        const int expected = sync.symbols[j] == '1' ? 3 : -3;
                        same += levels[start + j] == expected;
                        inverted += levels[start + j] == -expected;
                    }
                    if (same >= static_cast<int>(length) - 2 || inverted >= static_cast<int>(length) - 2)
                        out.protocol = sync.system;
                }
                if (!out.protocol.empty())
                    break;
            }
        }
    } else {
        // The deviation from the lightly filtered frequency: the matched filter
        // suits 4FSK systems but rounds off the edges of plain FSK.
        if (lowpassed_.empty())
            filter(lowpass_taps_, centred_, lowpassed_);
        std::vector<float> light;
        for (double t = first + best_phase; t < last; t += sps)
            light.push_back(sample_at(lowpassed_, t));
        std::vector<double> light_levels{two[0], two[1]};
        std::vector<size_t> light_counts;
        kmeans(light, light_levels, light_counts);
        out.deviation_hz        = 0.5 * (light_levels[1] - light_levels[0]);
        const double modulation = 2.0 * out.deviation_hz / rate;
        out.kind                = modulation >= 0.3 && modulation <= 0.75 ? SignalKind::Gmsk : SignalKind::Fsk2;
        // POCSAG: its frame sync codeword, in either polarity.
        if (out.kind == SignalKind::Fsk2 && (near_rate(rate, 512) || near_rate(rate, 1200) || near_rate(rate, 2400))) {
            const double middle = 0.5 * (two[0] + two[1]);
            uint32_t     bits   = 0;
            for (size_t i = 0; i < symbols_.size(); ++i) {
                bits = (bits << 1) | (symbols_[i] < middle ? 1u : 0u);
                if (i >= 31) {
                    const int differ = std::popcount(bits ^ kPocsagSync);
                    if (differ <= 2 || differ >= 30) {
                        out.protocol = "POCSAG";
                        break;
                    }
                }
            }
        }
    }
    return true;
}

}  // namespace spectro
