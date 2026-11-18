#include "audio/voice_playout.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace spectro {

namespace {

constexpr double kPi     = std::numbers::pi;
constexpr double kCutoff = 3600.0 / VoicePlayout::kInputRate;  // cycles per input sample: the voice band

double bessel_i0(double x)
{
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 30; ++k) {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
    }
    return sum;
}

}  // namespace

VoicePlayout::VoicePlayout()
{
    // A Kaiser-windowed sinc for each fractional delay: taps j = 0 .. 2H-1
    // weigh the input samples floor(t) - H + 1 + j.
    const int    taps = 2 * kHalfTaps;
    const double beta = 6.0, norm = bessel_i0(beta);
    table_.resize(static_cast<size_t>((kPhases + 1) * taps));
    for (int p = 0; p <= kPhases; ++p) {
        const double frac = static_cast<double>(p) / kPhases;
        double       sum  = 0.0;
        float*       row  = &table_[static_cast<size_t>(p * taps)];
        for (int j = 0; j < taps; ++j) {
            const double x = static_cast<double>(j - kHalfTaps + 1) - frac;  // distance from t
            const double u = x / kHalfTaps;
            const double w = std::fabs(u) < 1.0 ? bessel_i0(beta * std::sqrt(1.0 - u * u)) / norm : 0.0;
            const double s = std::fabs(x) < 1e-12 ? 1.0 : std::sin(2.0 * kPi * kCutoff * x) / (2.0 * kPi * kCutoff * x);
            row[j]         = static_cast<float>(2.0 * kCutoff * s * w);
            sum += row[j];
        }
        for (int j = 0; j < taps; ++j)
            row[j] = static_cast<float>(row[j] / sum);
    }
    reset();
}

void VoicePlayout::reset()
{
    buffer_.clear();
    read_     = 0;
    position_ = 0.0;
    playing_  = false;
}

double VoicePlayout::buffered() const
{
    const double waiting = static_cast<double>(buffer_.size() - read_) - position_;
    return std::max(0.0, waiting) / kInputRate;
}

void VoicePlayout::push(const float* samples, size_t count)
{
    if (buffer_.empty()) {
        // Start with the history the filter looks back on.
        buffer_.assign(kHalfTaps, 0.0f);
        read_     = kHalfTaps;
        position_ = 0.0;
    }
    buffer_.insert(buffer_.end(), samples, samples + count);
    // Too much waiting: the clocks have drifted apart, or the output stalled.
    const auto most = static_cast<size_t>(kMaxSeconds * kInputRate);
    if (buffer_.size() - read_ > most) {
        read_     = buffer_.size() - static_cast<size_t>(kPrefillSeconds * kInputRate);
        position_ = 0.0;
    }
}

void VoicePlayout::pull(float* out, size_t count, double rate)
{
    const double step = kInputRate / rate;
    const int    taps = 2 * kHalfTaps;
    if (!playing_ && buffered() >= kPrefillSeconds)
        playing_ = true;
    for (size_t i = 0; i < count; ++i) {
        // Interpolation needs kHalfTaps samples ahead of the position.
        if (!playing_ || read_ + static_cast<size_t>(position_) + kHalfTaps >= buffer_.size()) {
            playing_ = false;
            out[i]   = 0.0f;
            continue;
        }
        const auto   whole = static_cast<size_t>(position_);
        const double frac  = position_ - static_cast<double>(whole);
        const double phase = frac * kPhases;
        const int    p     = std::min(static_cast<int>(phase), kPhases - 1);
        const auto   blend = static_cast<float>(phase - p);
        const float* h0    = &table_[static_cast<size_t>(p * taps)];
        const float* h1    = h0 + taps;
        const float* x     = &buffer_[read_ + whole - kHalfTaps + 1];
        float        a = 0.0f, b = 0.0f;
        for (int j = 0; j < taps; ++j) {
            a += h0[j] * x[j];
            b += h1[j] * x[j];
        }
        out[i] = a + blend * (b - a);
        position_ += step;
    }
    // Drop what the filter no longer needs.
    const auto done = static_cast<size_t>(position_);
    if (done > 0) {
        read_ += done;
        position_ -= static_cast<double>(done);
    }
    if (read_ > 4096) {
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<long>(read_ - kHalfTaps));
        read_ = kHalfTaps;
    }
}

}  // namespace spectro
