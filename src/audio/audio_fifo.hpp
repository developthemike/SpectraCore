// Lock-free FIFO of audio samples between the thread that produces them and
// the sound card's callback - and a jitter buffer: playback starts only once
// enough audio is waiting, and starts over the same way after running dry, so
// that a late block of samples costs one short gap rather than a crackle.
//
// One producer thread, one consumer thread; the status calls are for anyone.
#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace spectro {

class AudioFifo {
public:
    // The capacity is rounded up to a power of two (in samples).
    explicit AudioFifo(size_t min_capacity)
        : data_(std::bit_ceil(std::max<size_t>(min_capacity, 64)), 0.0f), mask_(data_.size() - 1)
    {
    }

    size_t capacity() const { return data_.size(); }

    // ---- Producer -------------------------------------------------------------

    // Appends samples; returns how many fitted.
    size_t write(const float* samples, size_t count)
    {
        const uint64_t w    = write_pos_.load(std::memory_order_relaxed);
        const uint64_t r    = read_pos_.load(std::memory_order_acquire);
        const size_t   free = data_.size() - static_cast<size_t>(w - r);
        const size_t   n    = std::min(count, free);
        const size_t   idx  = static_cast<size_t>(w) & mask_;
        const size_t   n1   = std::min(n, data_.size() - idx);
        std::memcpy(data_.data() + idx, samples, n1 * sizeof(float));
        std::memcpy(data_.data(), samples + n1, (n - n1) * sizeof(float));
        write_pos_.store(w + n, std::memory_order_release);
        return n;
    }

    // Samples waiting to be played.
    size_t fill() const
    {
        return static_cast<size_t>(write_pos_.load(std::memory_order_acquire) -
                                   read_pos_.load(std::memory_order_acquire));
    }

    // Playback starts (and restarts) once this many samples are waiting.
    void set_start_level(size_t samples)
    {
        start_level_.store(std::min(samples, data_.size() / 2), std::memory_order_relaxed);
    }

    // Tell the FIFO whether audio is being produced.  Running dry while it is
    // not is the end of the programme rather than an underrun.
    void set_flowing(bool flowing) { flowing_.store(flowing, std::memory_order_relaxed); }

    // Makes the consumer throw away, at its next read, everything written so far.
    void discard() { discard_.store(true, std::memory_order_release); }

    // ---- Consumer -------------------------------------------------------------

    // Always delivers `count` samples: silence stands in for what is missing.
    void read(float* out, size_t count)
    {
        if (count > largest_read_.load(std::memory_order_relaxed))
            largest_read_.store(count, std::memory_order_relaxed);

        uint64_t       r = read_pos_.load(std::memory_order_relaxed);
        const uint64_t w = write_pos_.load(std::memory_order_acquire);
        if (discard_.exchange(false, std::memory_order_acq_rel)) {
            r = w;
            read_pos_.store(r, std::memory_order_release);
            playing_.store(false, std::memory_order_relaxed);
        }
        const size_t available = static_cast<size_t>(w - r);

        if (!playing_.load(std::memory_order_relaxed)) {
            if (available < std::max(start_level_.load(std::memory_order_relaxed), count)) {
                std::fill_n(out, count, 0.0f);
                return;
            }
            playing_.store(true, std::memory_order_relaxed);
        }

        const size_t n   = std::min(available, count);
        const size_t idx = static_cast<size_t>(r) & mask_;
        const size_t n1  = std::min(n, data_.size() - idx);
        std::memcpy(out, data_.data() + idx, n1 * sizeof(float));
        std::memcpy(out + n1, data_.data(), (n - n1) * sizeof(float));
        read_pos_.store(r + n, std::memory_order_release);

        if (n < count) {
            std::fill_n(out + n, count - n, 0.0f);
            playing_.store(false, std::memory_order_relaxed);
            if (flowing_.load(std::memory_order_relaxed))
                underruns_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // ---- Status ---------------------------------------------------------------

    bool     playing() const { return playing_.load(std::memory_order_relaxed); }
    uint64_t underruns() const { return underruns_.load(std::memory_order_relaxed); }
    // The most samples the consumer has asked for at once: its period.
    size_t   largest_read() const { return largest_read_.load(std::memory_order_relaxed); }

private:
    std::vector<float> data_;
    const size_t       mask_;

    alignas(64) std::atomic<uint64_t> write_pos_{0};
    alignas(64) std::atomic<uint64_t> read_pos_{0};

    std::atomic<size_t>   start_level_{0};
    std::atomic<size_t>   largest_read_{0};
    std::atomic<uint64_t> underruns_{0};
    std::atomic<bool>     playing_{false};
    std::atomic<bool>     flowing_{false};
    std::atomic<bool>     discard_{false};
};

}  // namespace spectro
