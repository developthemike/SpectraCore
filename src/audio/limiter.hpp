// Keeps the audio played within full scale without clipping it.
//
// The volume control can boost the audio beyond unity, and an overdeviated FM
// signal can go past full scale even without; cutting off what goes past +/-1
// would square off the waveform and distort it badly.  This peak limiter
// instead turns the level down the moment a sample would go past +/-1 - by
// just as much as needed - and back up over about kReleaseSeconds once it no
// longer would.  Audio that stays within full scale passes untouched, and the
// output never leaves +/-1.
#pragma once

#include <algorithm>
#include <cmath>

namespace spectro {

class PeakLimiter {
public:
    static constexpr double kReleaseSeconds = 0.2;

    explicit PeakLimiter(double sample_rate)
        : release_(static_cast<float>(std::exp(-1.0 / (kReleaseSeconds * sample_rate))))
    {
    }

    void reset() { peak_ = 0.0f; }

    float process(float v)
    {
        // The peak level, taken at once and let go of slowly - but never
        // below this sample's, or the sample would end up past full scale.
        peak_ = std::max(std::fabs(v), peak_ * release_);
        return peak_ > 1.0f ? v / peak_ : v;
    }

    // How far the level is turned down now, in dB (0 when it is not).
    float reduction_db() const { return peak_ > 1.0f ? 20.0f * std::log10(peak_) : 0.0f; }

private:
    float release_;
    float peak_ = 0.0f;
};

}  // namespace spectro
