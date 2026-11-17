// The last few seconds of the audio being played, for a time series plot.
//
// The audio thread writes; one other thread reads either the most recent
// samples or, for long stretches, the minimum and maximum of every block of
// kBlock samples - all it takes to draw them.  The buffer holds twice as much
// as can be read in one call, so the writer cannot catch up with a reader
// that is still copying.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace spectro {

class AudioScope {
public:
    static constexpr size_t kCapacity = size_t{1} << 19;  // samples: 10.9 s at 48 kHz
    static constexpr size_t kBlock    = 64;
    static constexpr size_t kBlocks   = kCapacity / kBlock;
    static constexpr size_t kMaxRead  = kCapacity / 2;    // samples per call

    AudioScope() : samples_(kCapacity, 0.0f), mins_(kBlocks, 0.0f), maxs_(kBlocks, 0.0f) {}

    // ---- Writer -----------------------------------------------------------------

    // Appends audio; with `silent`, silence of the same length instead.
    void write(const float* samples, size_t count, bool silent)
    {
        uint64_t pos = written_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < count; ++i) {
            const float v = silent ? 0.0f : samples[i];
            samples_[pos & (kCapacity - 1)] = v;
            block_min_ = std::min(block_min_, v);
            block_max_ = std::max(block_max_, v);
            if (++pos % kBlock == 0) {
                const size_t block = (pos / kBlock - 1) & (kBlocks - 1);
                mins_[block]       = block_min_;
                maxs_[block]       = block_max_;
                block_min_         = std::numeric_limits<float>::infinity();
                block_max_         = -std::numeric_limits<float>::infinity();
            }
        }
        written_.store(pos, std::memory_order_release);
    }

    // ---- Reader -----------------------------------------------------------------

    uint64_t written() const { return written_.load(std::memory_order_acquire); }

    // Copies the latest `count` samples, oldest first.  Returns how many there
    // were (fewer just after the start).
    size_t latest(float* out, size_t count) const
    {
        const uint64_t end = written();
        count = static_cast<size_t>(std::min<uint64_t>({count, end, kMaxRead}));
        for (size_t i = 0; i < count; ++i)
            out[i] = samples_[(end - count + i) & (kCapacity - 1)];
        return count;
    }

    // The minimum and maximum of each of the latest `count` complete blocks,
    // oldest first.  Returns how many blocks there were.
    size_t latest_blocks(float* mins, float* maxs, size_t count) const
    {
        const uint64_t blocks = written() / kBlock;
        count = static_cast<size_t>(std::min<uint64_t>({count, blocks, kMaxRead / kBlock}));
        for (size_t i = 0; i < count; ++i) {
            const size_t block = (blocks - count + i) & (kBlocks - 1);
            mins[i]            = mins_[block];
            maxs[i]            = maxs_[block];
        }
        return count;
    }

private:
    std::vector<float>    samples_;
    std::vector<float>    mins_, maxs_;
    std::atomic<uint64_t> written_{0};
    float                 block_min_ = std::numeric_limits<float>::infinity();   // writer only
    float                 block_max_ = -std::numeric_limits<float>::infinity();
};

}  // namespace spectro
