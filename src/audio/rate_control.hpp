// Keeps audio production in step with a sound card.
//
// The radio and the sound card each have a clock of their own, so audio is
// never produced at exactly the rate at which it is played: left alone, the
// FIFO between the two slowly runs dry or overflows.  This is the control loop
// that prevents it.  Each time a block of audio is about to be written, it
// looks at how much is still waiting in the FIFO, and returns
//
//   - a small correction of the production rate (at most half a percent,
//     far too little to hear) that steers that amount towards a target, and
//   - the target itself, as the level at which playback should (re)start: a
//     fixed reserve plus an allowance for the size of the blocks in which
//     samples arrive from the radio and are fetched by the sound card.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace spectro {

class AudioRateControl {
public:
    static constexpr double kReserveSeconds     = 0.04;   // audio still waiting when a new block arrives
    static constexpr double kBurstMemorySeconds = 5.0;    // how long a large block is remembered
    static constexpr double kSmoothingSeconds   = 1.0;    // of the measured amount
    static constexpr double kLoopGain           = 0.1;    // 10 ms too much waiting: produce 0.1 % less
    static constexpr double kMaxTrim            = 0.005;
    static constexpr double kMaxExcessSeconds   = 0.25;   // more surplus than this is thrown away

    explicit AudioRateControl(double audio_rate) : audio_rate_(audio_rate) {}

    void reset()
    {
        waiting_s_ = 0.0;
        burst_s_   = 0.0;
        trim_      = 0.0;
    }

    struct Decision {
        double trim        = 0.0;    // produce audio at (1 + trim) times the nominal rate
        size_t start_level = 0;      // samples that should be waiting before playback starts
        bool   discard     = false;  // the sound card has stalled: drop this block
    };

    // block_seconds: signal time covered by the block about to be written.
    // fill:          samples waiting in the FIFO right now.
    // period:        the most samples the sound card fetches at once.
    // playing:       whether it is fetching audio at the moment.
    Decision update(double block_seconds, size_t fill, size_t period, bool playing)
    {
        burst_s_ = std::max(block_seconds, burst_s_ * std::exp(-block_seconds / kBurstMemorySeconds));
        const double period_s = static_cast<double>(period) / audio_rate_;
        const double fill_s   = static_cast<double>(fill) / audio_rate_;
        target_s_ = std::clamp(kReserveSeconds + 0.5 * burst_s_ + period_s, 0.05, 0.6);

        if (playing) {
            waiting_s_ += (1.0 - std::exp(-block_seconds / kSmoothingSeconds)) * (fill_s - waiting_s_);
            trim_ = std::clamp(-kLoopGain * (waiting_s_ - target_s_), -kMaxTrim, kMaxTrim);
        } else {
            waiting_s_ = target_s_;  // starting up: there is nothing to measure yet
        }

        Decision d;
        d.trim        = trim_;
        d.start_level = static_cast<size_t>((target_s_ + 0.5 * period_s) * audio_rate_);
        if (fill_s > target_s_ + kMaxExcessSeconds) {
            d.discard  = true;
            waiting_s_ = target_s_;
        }
        return d;
    }

    double trim() const { return trim_; }
    double target_seconds() const { return target_s_; }

private:
    const double audio_rate_;
    double       waiting_s_ = 0.0;  // audio waiting when a block arrives, smoothed
    double       burst_s_   = 0.0;  // duration of the largest recent block
    double       target_s_  = 0.05;
    double       trim_      = 0.0;
};

}  // namespace spectro
