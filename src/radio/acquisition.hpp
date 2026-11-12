// Receive thread: moves samples from an IqSource into the IqRing and keeps
// count of everything that happens to the stream on the way.
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "radio/iq_source.hpp"
#include "util/iq_ring.hpp"

namespace spectro {

struct AcquisitionStats {
    uint64_t samples         = 0;    // samples delivered to the DSP thread
    uint64_t discontinuities = 0;    // times the source reported a break in the stream
    uint64_t lost_samples    = 0;    // samples the source lost (device / USB overruns)
    uint64_t ring_dropped    = 0;    // samples discarded because the DSP thread fell behind
    double   measured_rate   = 0.0;  // samples per second actually arriving
    double   ring_fill       = 0.0;  // 0..1
    bool     running         = false;
    bool     failed          = false;
    std::string error;               // set when failed
};

class Acquisition {
public:
    Acquisition(IqSource& source, IqRing& ring);
    ~Acquisition();

    Acquisition(const Acquisition&)            = delete;
    Acquisition& operator=(const Acquisition&) = delete;

    // The source must already be started.
    void start();
    // Stops and joins the thread.  The source may be stopped afterwards.
    void stop();

    AcquisitionStats stats() const;

private:
    void run();

    IqSource& source_;
    IqRing&   ring_;

    std::thread       thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> failed_{false};

    std::atomic<uint64_t> samples_{0};
    std::atomic<uint64_t> discontinuities_{0};
    std::atomic<uint64_t> lost_samples_{0};
    std::atomic<uint64_t> ring_dropped_{0};
    std::atomic<double>   measured_rate_{0.0};

    mutable std::mutex error_mutex_;
    std::string        error_;
};

}  // namespace spectro
