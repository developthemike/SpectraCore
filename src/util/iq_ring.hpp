// Lock-free single-producer / single-consumer ring buffer for IQ samples.
//
// One "sample" is a complex value stored as two interleaved int16_t (I, Q).
// The producer (radio thread) writes straight into the ring's memory, the
// consumer (DSP thread) reads overlapping FFT frames straight out of it, so the
// 80 MB/s sample stream is never copied on its way through.
#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

namespace spectro {

class IqRing {
public:
    // Capacity is rounded up to a power of two (in samples).
    explicit IqRing(size_t min_capacity_samples)
        : capacity_(std::bit_ceil(std::max<size_t>(min_capacity_samples, 1024))),
          mask_(capacity_ - 1)
    {
        const size_t bytes = capacity_ * 2 * sizeof(int16_t);
        data_ = static_cast<int16_t*>(std::aligned_alloc(64, bytes));
        if (!data_)
            throw std::bad_alloc();
        std::memset(data_, 0, bytes);  // also pre-faults the pages
    }

    ~IqRing() { std::free(data_); }

    IqRing(const IqRing&)            = delete;
    IqRing& operator=(const IqRing&) = delete;

    size_t capacity() const { return capacity_; }

    // Number of samples currently buffered (exact for the consumer, a lower
    // bound for anyone else).
    size_t readable() const
    {
        return static_cast<size_t>(write_pos_.load(std::memory_order_acquire) -
                                   read_pos_.load(std::memory_order_relaxed));
    }

    // ---- Producer side ------------------------------------------------------

    struct WriteSpan {
        int16_t* data    = nullptr;  // room for `samples` IQ pairs
        size_t   samples = 0;
    };

    // The largest contiguous writable region, limited to max_samples.  Returns an
    // empty span when the ring is full.
    WriteSpan write_span(size_t max_samples)
    {
        const uint64_t w    = write_pos_.load(std::memory_order_relaxed);
        const uint64_t r    = read_pos_.load(std::memory_order_acquire);
        const size_t   free = capacity_ - static_cast<size_t>(w - r);
        const size_t   idx  = static_cast<size_t>(w) & mask_;
        const size_t   n    = std::min({free, capacity_ - idx, max_samples});
        return {data_ + 2 * idx, n};
    }

    // Publishes `samples` IQ pairs previously written into write_span().
    void commit(size_t samples)
    {
        write_pos_.store(write_pos_.load(std::memory_order_relaxed) + samples,
                         std::memory_order_release);
        signal();
    }

    // Wakes the consumer and makes wait_readable() return false once the
    // remaining data is insufficient.  May be called from any thread.
    void close()
    {
        closed_.store(true, std::memory_order_release);
        signal();
    }

    bool closed() const { return closed_.load(std::memory_order_acquire); }

    // ---- Consumer side ------------------------------------------------------

    // Blocks until at least `samples` are buffered.  Returns false if the ring
    // was closed before that happened.
    bool wait_readable(size_t samples)
    {
        for (;;) {
            // Load the event counter *before* testing the condition: a commit that
            // lands in between changes the counter and makes wait() return at once.
            const uint32_t seen = event_.load(std::memory_order_acquire);
            if (readable() >= samples)
                return true;
            if (closed_.load(std::memory_order_acquire))
                return false;
            event_.wait(seen, std::memory_order_acquire);
        }
    }

    struct ReadSpans {
        const int16_t* first        = nullptr;
        size_t         first_count  = 0;  // samples in `first`
        const int16_t* second       = nullptr;
        size_t         second_count = 0;  // samples in `second` (after wrap-around)
    };

    // The oldest `samples` buffered IQ pairs, as one or two contiguous pieces.
    // Requires readable() >= samples.  Does not consume them.
    ReadSpans peek(size_t samples) const
    {
        const size_t idx = static_cast<size_t>(read_pos_.load(std::memory_order_relaxed)) & mask_;
        const size_t n1  = std::min(samples, capacity_ - idx);
        return {data_ + 2 * idx, n1, data_, samples - n1};
    }

    // Discards the oldest `samples` IQ pairs.  Requires readable() >= samples.
    void consume(size_t samples)
    {
        read_pos_.store(read_pos_.load(std::memory_order_relaxed) + samples,
                        std::memory_order_release);
    }

private:
    void signal()
    {
        event_.fetch_add(1, std::memory_order_release);
        event_.notify_all();
    }

    int16_t*     data_ = nullptr;
    const size_t capacity_;
    const size_t mask_;

    // Monotonic sample counters; the buffer index is (counter & mask_).
    alignas(64) std::atomic<uint64_t> write_pos_{0};
    alignas(64) std::atomic<uint64_t> read_pos_{0};
    alignas(64) std::atomic<uint32_t> event_{0};
    std::atomic<bool>                 closed_{false};
};

}  // namespace spectro
