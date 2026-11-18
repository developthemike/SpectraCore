#include "dsp/fsk4_front.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>

namespace spectro {

namespace {

constexpr double kPi          = std::numbers::pi;
constexpr double kRolloff     = 0.2;     // of the root-raised-cosine pulses
constexpr int    kFilterSpan  = 4;       // symbols either side
constexpr double kMinLevelHz  = 400.0;   // outer symbols: DMR's are at 1944 Hz, P25's at 1800 Hz
constexpr double kMaxLevelHz  = 4000.0;
constexpr double kMaxMisfit   = 0.06;    // of a sync pattern: residual power / outer level^2

// Root raised cosine, `sps` samples per symbol, unit gain at 0 Hz.
std::vector<float> root_raised_cosine(double sps)
{
    const int          half = static_cast<int>(std::ceil(kFilterSpan * sps));
    std::vector<float> h(static_cast<size_t>(2 * half + 1));
    double             sum = 0.0;
    const double       a   = kRolloff;
    for (int i = -half; i <= half; ++i) {
        const double t = i / sps;
        double       v = 0.0;
        if (std::fabs(t) < 1e-9)
            v = 1.0 - a + 4.0 * a / kPi;
        else if (std::fabs(std::fabs(t) - 1.0 / (4.0 * a)) < 1e-9)
            v = a / std::sqrt(2.0) *
                ((1.0 + 2.0 / kPi) * std::sin(kPi / (4.0 * a)) + (1.0 - 2.0 / kPi) * std::cos(kPi / (4.0 * a)));
        else
            v = (std::sin(kPi * t * (1.0 - a)) + 4.0 * a * t * std::cos(kPi * t * (1.0 + a))) /
                (kPi * t * (1.0 - 16.0 * a * a * t * t));
        h[static_cast<size_t>(i + half)] = static_cast<float>(v);
        sum += v;
    }
    for (float& v : h)
        v = static_cast<float>(v / sum);
    return h;
}

}  // namespace

Fsk4Front::Fsk4Front(double sample_rate, std::vector<uint64_t> patterns, double history_symbols)
    : Fsk4Front(sample_rate, std::move(patterns), history_symbols, Options{})
{
}

Fsk4Front::Fsk4Front(double sample_rate, std::vector<uint64_t> patterns, double history_symbols, Options options)
    : symbol_rate_(options.symbol_rate),
      length_(std::min<size_t>(options.pattern_symbols, 24)),
      nominal_sps_(sample_rate / options.symbol_rate),
      sps_(nominal_sps_)
{
    if (options.boxcar) {
        const auto n = static_cast<size_t>(std::max(1L, std::lround(sps_)));
        taps_.assign(n, 1.0f / static_cast<float>(n));
    } else {
        taps_ = root_raised_cosine(sps_);
    }
    fir_.assign(2 * taps_.size(), 0.0f);
    const size_t needed = static_cast<size_t>(std::ceil(history_symbols * sps_)) + taps_.size() + 64;
    history_.assign(std::bit_ceil(needed), 0.0f);
    mask_ = history_.size() - 1;

    // Symbols as sent: +3, +1, -1, -3 for the dibits 01, 00, 10, 11; the
    // patterns hold only the outer ones.
    for (const uint64_t bits : patterns) {
        std::array<int8_t, 24> p{};
        double                 sum = 0.0;
        for (size_t k = 0; k < length_; ++k) {
            const auto dibit = static_cast<unsigned>((bits >> (2 * (length_ - 1 - k))) & 3u);
            p[k]             = static_cast<int8_t>((dibit & 2u) ? -1 : 1);
            sum += p[k];
        }
        patterns_.push_back(p);
        pattern_mean_.push_back(sum / static_cast<double>(length_));
    }
    correlation_.assign(patterns_.size(), 0.0);
    reset();
}

void Fsk4Front::reset()
{
    sps_ = nominal_sps_;
    std::fill(fir_.begin(), fir_.end(), 0.0f);
    fir_pos_ = 0;
    std::fill(history_.begin(), history_.end(), 0.0f);
    written_ = 0;
    last_correlation_.assign(patterns_.size(), 0.0);
    candidate_ = {};
}

double Fsk4Front::sample(double t) const
{
    t                = std::max(t, 0.0);
    const double fl  = std::floor(t);
    const auto   i   = static_cast<uint64_t>(fl);
    const double f   = t - fl;
    const double a   = history_[i & mask_];
    const double b   = history_[(i + 1) & mask_];
    return a + f * (b - a);
}

bool Fsk4Front::push(float frequency, Sync& sync)
{
    const size_t taps = taps_.size();
    fir_[fir_pos_] = fir_[fir_pos_ + taps] = frequency;
    fir_pos_                               = fir_pos_ + 1 == taps ? 0 : fir_pos_ + 1;
    const float* x                         = fir_.data() + fir_pos_;
    float        acc                       = 0.0f;
    for (size_t j = 0; j < taps; ++j)
        acc += taps_[j] * x[j];
    history_[written_ & mask_] = acc;
    ++written_;
    return search(sync);
}

bool Fsk4Front::search(Sync& sync)
{
    const double last = static_cast<double>(length_ - 1);
    const double n    = static_cast<double>(length_);
    const auto   span = static_cast<uint64_t>(std::ceil(last * sps_)) + taps_.size();
    if (written_ <= span + 2)
        return false;
    const uint64_t newest = written_ - 1;

    double v[24];
    double sum = 0.0, sum_sq = 0.0;
    for (size_t k = 0; k < length_; ++k) {
        const auto back = static_cast<uint64_t>(std::llround((last - static_cast<double>(k)) * sps_));
        v[k]            = history_[(newest - back) & mask_];
        sum += v[k];
        sum_sq += v[k] * v[k];
    }
    const double mean = sum / n;
    const double sxx  = sum_sq - n * mean * mean;

    double best_misfit = kMaxMisfit;
    size_t best        = patterns_.size();
    double best_level = 0.0, best_dc = 0.0;
    for (size_t p = 0; p < patterns_.size(); ++p) {
        double sv = 0.0;
        for (size_t k = 0; k < length_; ++k)
            sv += patterns_[p][k] * v[k];
        const double ms  = pattern_mean_[p];
        const double ssx = sv - n * ms * mean;
        correlation_[p]  = ssx;
        const double level = ssx / (n * (1.0 - ms * ms));
        if (level < kMinLevelHz || level > kMaxLevelHz)
            continue;
        const double misfit = (sxx - level * ssx) / (n * level * level);
        if (misfit < best_misfit) {
            best_misfit = misfit;
            best        = p;
            best_level  = level;
            best_dc     = mean - level * ms;
        }
    }

    bool found = false;
    if (candidate_.active && !candidate_.have_after && newest == candidate_.index + 1) {
        candidate_.after      = correlation_[candidate_.pattern];
        candidate_.have_after = true;
    }
    if (candidate_.active && static_cast<double>(newest - candidate_.index) >= 0.5 * sps_ && candidate_.have_after) {
        // The correlation either side of the best sample places the pattern
        // between samples.
        const double curve = candidate_.before - 2.0 * candidate_.at + candidate_.after;
        double       shift = 0.0;
        if (curve < 0.0)
            shift = std::clamp(0.5 * (candidate_.before - candidate_.after) / curve, -0.5, 0.5);
        sync.pattern      = candidate_.pattern;
        sync.time         = static_cast<double>(candidate_.index) + shift - last * sps_;
        sync.level        = candidate_.level;
        sync.dc           = candidate_.dc;
        candidate_.active = false;
        found             = true;
    }
    if (best < patterns_.size() && (!candidate_.active || best_misfit < candidate_.misfit)) {
        candidate_.active     = true;
        candidate_.index      = newest;
        candidate_.misfit     = best_misfit;
        candidate_.pattern    = best;
        candidate_.level      = best_level;
        candidate_.dc         = best_dc;
        candidate_.before     = last_correlation_[best];
        candidate_.at         = correlation_[best];
        candidate_.have_after = false;
    }
    std::swap(last_correlation_, correlation_);
    return found;
}

}  // namespace spectro
