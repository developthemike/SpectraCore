// RTL-SDR (RTL2832U-based DVB-T dongles) receive source, built on librtlsdr's
// asynchronous streaming interface.
//
// librtlsdr delivers samples through a callback on a thread of its own.  That
// thread only copies each USB buffer into a FIFO; read() takes the samples out
// and converts them from the dongle's unsigned 8-bit format to SC16 Q11.
//
// Unlike the bladeRF, the RTL2832 does not timestamp its samples, so samples
// lost inside the dongle or on the USB bus cannot be detected.  Samples that
// this program fails to take over in time are counted and reported.
#pragma once

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "radio/iq_source.hpp"

struct rtlsdr_dev;

namespace spectro {

// The RTL2832 resamples to 225001...300000 S/s or 900001...3200000 S/s only.
inline bool rtlsdr_rate_supported(double rate)
{
    return (rate > 225000.0 && rate <= 300000.0) || (rate > 900000.0 && rate <= 3200000.0);
}

// RTL-SDR tuners offer a fixed, unevenly spaced list of gains.  Returns the
// index of the entry of `gains_db` (ascending, not empty) to use for a request
// of `requested_db` while the tuner is at `current_db`: the nearest entry,
// except that a request at least half a dB away from the current gain always
// moves at least one entry in its direction.  Without that, stepping the gain
// by 1 dB would never cross a 3 dB gap in the list.
inline size_t rtlsdr_pick_gain(const std::vector<double>& gains_db, double current_db, double requested_db)
{
    size_t nearest = 0, current = 0;
    for (size_t i = 1; i < gains_db.size(); ++i) {
        if (std::fabs(gains_db[i] - requested_db) < std::fabs(gains_db[nearest] - requested_db))
            nearest = i;
        if (std::fabs(gains_db[i] - current_db) < std::fabs(gains_db[current] - current_db))
            current = i;
    }
    if (nearest == current) {
        if (requested_db >= current_db + 0.5 && current + 1 < gains_db.size())
            return current + 1;
        if (requested_db <= current_db - 0.5 && current > 0)
            return current - 1;
    }
    return nearest;
}

class RtlSdrSource final : public IqSource {
public:
    explicit RtlSdrSource(const SourceConfig& config);
    ~RtlSdrSource() override;

    void start() override;
    void stop() override;
    int  read(int16_t* iq, int max_samples, ReadStatus& status) override;

    const SourceInfo& info() const override { return info_; }
    double sample_rate() const override { return sample_rate_; }
    double frequency() const override { return frequency_.load(std::memory_order_relaxed); }
    double bandwidth() const override { return bandwidth_; }
    double gain() const override { return gain_.load(std::memory_order_relaxed); }

    bool set_frequency(double hz) override;
    bool set_gain(double db) override;

    std::string last_error() const override;
    int         preferred_chunk() const override { return chunk_samples_; }

private:
    static void usb_callback(unsigned char* data, uint32_t length, void* context);
    void        on_usb_data(const unsigned char* data, size_t length);
    void        stream_thread(uint32_t buffer_count, uint32_t buffer_bytes);
    void        stop_stream();
    void        set_error(const std::string& message);

    SourceConfig config_;
    SourceInfo   info_;

    rtlsdr_dev* dev_ = nullptr;

    double              sample_rate_ = 0.0;
    double              bandwidth_   = 0.0;
    std::atomic<double> frequency_{0.0};
    std::atomic<double> gain_{0.0};
    std::vector<double> gains_db_;  // what the tuner offers, ascending
    int                 chunk_samples_ = 16384;

    // Serialises device control calls coming from the UI thread.
    std::mutex control_mutex_;

    // FIFO of raw I/Q bytes between librtlsdr's thread and read().
    std::thread             thread_;
    std::mutex              fifo_mutex_;
    std::condition_variable fifo_cv_;
    std::vector<uint8_t>    fifo_;
    size_t                  fifo_head_ = 0;      // next byte to read
    size_t                  fifo_size_ = 0;      // bytes stored
    size_t                  skip_bytes_ = 0;     // start-up transient still to discard
    uint64_t                dropped_samples_ = 0;  // FIFO was full; not yet reported by read()
    bool                    stream_ended_ = true;
    int                     stream_result_ = 0;

    int consecutive_timeouts_ = 0;  // reader thread only

    mutable std::mutex error_mutex_;
    std::string        last_error_;
};

}  // namespace spectro
