// Plays back a raw IQ recording at its original sample rate.
//
// Supported sample formats (all little-endian, interleaved I/Q):
//   sc16  int16, Q11 (full scale +/-2048): what "bladeRF-cli rx" writes with format=bin
//   cs16  int16, full scale +/-32768
//   cf32  float32, full scale +/-1.0 (GNU Radio "complex" files)
//   cs8   int8
//   cu8   uint8 with a 127.5 offset (rtl_sdr)
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <vector>

#include "radio/iq_source.hpp"

namespace spectro {

class FileSource final : public IqSource {
public:
    explicit FileSource(const SourceConfig& config);
    ~FileSource() override;

    void start() override;
    void stop() override;
    int  read(int16_t* iq, int max_samples, ReadStatus& status) override;

    const SourceInfo& info() const override { return info_; }
    double sample_rate() const override { return config_.sample_rate; }
    double frequency() const override { return config_.frequency; }
    double bandwidth() const override { return config_.sample_rate; }
    double gain() const override { return 0.0; }

    std::string last_error() const override;

    enum class Format { SC16Q11, CS16, CF32, CS8, CU8 };
    // Resolves a format name ("sc16", "cf32", ...) or, for "auto", the file extension.
    static bool resolve_format(const std::string& name, const std::string& path, Format& format);
    static size_t bytes_per_sample(Format format);

private:
    void set_error(const std::string& message);

    SourceConfig config_;
    SourceInfo   info_;
    Format       format_ = Format::SC16Q11;

    std::FILE*           file_ = nullptr;
    std::vector<uint8_t> raw_;
    std::atomic<bool>    running_{false};

    std::chrono::steady_clock::time_point start_time_;
    uint64_t                              produced_ = 0;

    mutable std::mutex error_mutex_;
    std::string        last_error_;
};

}  // namespace spectro
