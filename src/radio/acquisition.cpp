#include "radio/acquisition.hpp"

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "util/log.hpp"
#include "util/thread_util.hpp"

namespace spectro {

Acquisition::Acquisition(IqSource& source, IqRing& ring) : source_(source), ring_(ring) {}

Acquisition::~Acquisition()
{
    stop();
}

void Acquisition::start()
{
    if (thread_.joinable())
        return;
    stop_.store(false);
    failed_.store(false);
    running_.store(true);
    thread_ = std::thread([this] { run(); });
}

void Acquisition::stop()
{
    if (!thread_.joinable())
        return;
    stop_.store(true, std::memory_order_release);
    thread_.join();
}

AcquisitionStats Acquisition::stats() const
{
    AcquisitionStats s;
    s.samples         = samples_.load(std::memory_order_relaxed);
    s.discontinuities = discontinuities_.load(std::memory_order_relaxed);
    s.lost_samples    = lost_samples_.load(std::memory_order_relaxed);
    s.ring_dropped    = ring_dropped_.load(std::memory_order_relaxed);
    s.measured_rate   = measured_rate_.load(std::memory_order_relaxed);
    s.ring_fill       = static_cast<double>(ring_.readable()) / static_cast<double>(ring_.capacity());
    s.running         = running_.load(std::memory_order_relaxed);
    s.failed          = failed_.load(std::memory_order_relaxed);
    if (s.failed) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        s.error = error_;
    }
    return s;
}

void Acquisition::run()
{
    set_current_thread_name("spectro-rx");

    const int chunk = std::max(1024, source_.preferred_chunk());
    // Where samples go when the ring is full.  Reading must never stop, or the
    // device would overflow and we could not even count what was lost.
    std::vector<int16_t> overflow(2 * static_cast<size_t>(chunk));

    auto     rate_time    = std::chrono::steady_clock::now();
    uint64_t rate_samples = 0;
    bool     warned_full  = false;

    while (!stop_.load(std::memory_order_acquire)) {
        IqRing::WriteSpan span    = ring_.write_span(static_cast<size_t>(chunk));
        const bool        discard = span.samples == 0;
        int16_t*          dest    = discard ? overflow.data() : span.data;
        const int         wanted  = discard ? chunk : static_cast<int>(span.samples);

        ReadStatus status;
        const int  n = source_.read(dest, wanted, status);
        if (n < 0) {
            std::string message = source_.last_error();
            if (message.empty())
                message = "The sample source stopped unexpectedly";
            {
                std::lock_guard<std::mutex> lock(error_mutex_);
                error_ = message;
            }
            // Publish "no longer running" before "failed", so that nobody can
            // observe a failed stream that still claims to be running.
            running_.store(false, std::memory_order_release);
            failed_.store(true, std::memory_order_release);
            LOG_ERROR("%s", message.c_str());
            break;
        }

        if (status.discontinuity) {
            discontinuities_.fetch_add(1, std::memory_order_relaxed);
            lost_samples_.fetch_add(status.lost_samples, std::memory_order_relaxed);
        }
        if (n == 0)
            continue;  // timeout: try again

        if (discard) {
            ring_dropped_.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
            if (!warned_full) {
                LOG_WARN("The DSP thread is not keeping up; samples are being discarded");
                warned_full = true;
            }
        } else {
            ring_.commit(static_cast<size_t>(n));
            samples_.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
            warned_full = false;
        }

        // Samples arrive in blocks, so a short measurement is only accurate to one
        // block; average a few half-second windows for a steady reading.
        rate_samples += static_cast<uint64_t>(n);
        const auto   now     = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - rate_time).count();
        if (elapsed >= 0.5) {
            const double rate     = static_cast<double>(rate_samples) / elapsed;
            const double previous = measured_rate_.load(std::memory_order_relaxed);
            measured_rate_.store(previous > 0.0 ? previous + 0.25 * (rate - previous) : rate,
                                 std::memory_order_relaxed);
            rate_time    = now;
            rate_samples = 0;
        }
    }

    running_.store(false, std::memory_order_release);
}

}  // namespace spectro
