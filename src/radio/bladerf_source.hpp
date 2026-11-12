// Nuand bladeRF (x40/x115 and 2.0 micro) receive source, built on libbladeRF's
// synchronous streaming interface with per-buffer metadata, which lets us detect
// and count every dropped sample.
#pragma once

#include <atomic>
#include <mutex>

#include "radio/iq_source.hpp"

struct bladerf;

namespace spectro {

class BladeRfSource final : public IqSource {
public:
    explicit BladeRfSource(const SourceConfig& config);
    ~BladeRfSource() override;

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
    void fail(const std::string& what, int status);
    void set_error(const std::string& message);

    SourceConfig config_;
    SourceInfo   info_;

    bladerf* dev_     = nullptr;
    int      channel_ = 0;  // bladerf_channel
    bool     streaming_ = false;

    double              sample_rate_ = 0.0;
    double              bandwidth_   = 0.0;
    std::atomic<double> frequency_{0.0};
    std::atomic<double> gain_{0.0};
    int                 chunk_samples_ = 16384;

    // Serialises device control calls coming from the UI thread.
    std::mutex control_mutex_;

    // Stream continuity tracking (reader thread only).
    bool     have_timestamp_ = false;
    uint64_t next_timestamp_ = 0;
    int      consecutive_timeouts_ = 0;

    mutable std::mutex error_mutex_;
    std::string        last_error_;
};

}  // namespace spectro
