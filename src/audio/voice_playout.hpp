// Plays decoded digital voice in place of the FM audio.
//
// A vocoder gives 8 kHz speech in bursts - DMR, three 20 ms frames every
// 60 ms - while the audio path wants a steady stream at its own rate.  This
// buffers enough speech to ride out the gaps, then resamples it to whatever
// rate is asked for, so that the stream keeps the exact length the FM audio
// would have had (and the audio pacing goes on working).  When it runs dry it
// gives silence and waits until it has enough again.
#pragma once

#include <cstddef>
#include <vector>

namespace spectro {

class VoicePlayout {
public:
    static constexpr double kInputRate       = 8000.0;
    static constexpr double kPrefillSeconds  = 0.1;   // buffered before playing starts
    static constexpr double kMaxSeconds      = 0.5;   // more is thrown away

    VoicePlayout();

    void reset();

    // Speech at kInputRate.
    void push(const float* samples, size_t count);

    // `count` samples at `rate` per second: speech, or silence while there is
    // not enough.
    void pull(float* out, size_t count, double rate);

    // Whether speech is coming out.
    bool playing() const { return playing_; }
    // Seconds of speech waiting.
    double buffered() const;

private:
    static constexpr int kHalfTaps = 8;    // zero crossings either side
    static constexpr int kPhases   = 128;  // of the interpolation filter

    std::vector<float> buffer_;      // speech, from read_ on
    size_t             read_ = 0;    // index of the sample at position_ 0
    double             position_ = 0.0;  // fractional, from read_
    bool               playing_  = false;
    std::vector<float> table_;       // (kPhases + 1) x 2 kHalfTaps filter taps
};

}  // namespace spectro
