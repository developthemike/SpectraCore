// Synthetic IQ source for running without hardware.
//
// Produces a lively but fully deterministic RF scene at the configured sample
// rate, paced to real time: a noise floor, steady and fading carriers, an FM
// signal, a sweeping chirp, wide-band bursts and a frequency hopper.  The
// signals live at fixed "RF" frequencies around the initial centre frequency,
// so tuning and gain behave as they would on a real receiver.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>

#include "radio/iq_source.hpp"

namespace spectro {

class SimSource final : public IqSource {
public:
    explicit SimSource(const SourceConfig& config);
    ~SimSource() override;

    void start() override;
    void stop() override;
    int  read(int16_t* iq, int max_samples, ReadStatus& status) override;

    const SourceInfo& info() const override { return info_; }
    double sample_rate() const override { return sample_rate_; }
    double frequency() const override { return frequency_.load(std::memory_order_relaxed); }
    double bandwidth() const override { return sample_rate_; }
    double gain() const override { return gain_.load(std::memory_order_relaxed); }

    bool set_frequency(double hz) override;
    bool set_gain(double db) override;

    // Positions and levels of the steady test carriers, for tests and docs.
    struct Carrier {
        double offset_hz;   // from the initial centre frequency
        double level_dbfs;  // at the reference gain
    };
    static std::vector<Carrier> carriers(double sample_rate);
    static constexpr double kReferenceGainDb = 30.0;

private:
    struct Tone {
        double offset;       // fraction of the sample rate, relative to the initial centre
        double level_dbfs;
        double fade_hz;      // amplitude fading rate, 0 = steady
        double fade_depth;   // 0..1
        double phase = 0.0;  // radians
    };

    void build_tables();
    void generate(int n);
    void add_noise(int n);
    void add_tone(Tone& tone, int n, double shift);
    void add_fm(int n, double shift);
    void add_chirp(int n, double shift);
    void add_bursts(int n, double shift);
    void add_hopper(int n, double shift);
    static void add_table_signal(const std::vector<float>& table, size_t& table_pos, double& phase,
                                 double freq, float amplitude, const float* envelope, float* mix, int n);
    uint32_t next_random();
    double   random_uniform();

    SourceInfo info_;
    double     sample_rate_;
    double     initial_frequency_;
    unsigned   seed_;

    std::atomic<double> frequency_;
    std::atomic<double> gain_;
    std::atomic<bool>   running_{false};

    // Pacing
    std::chrono::steady_clock::time_point start_time_;
    uint64_t                              produced_ = 0;
    double                                time_s_   = 0.0;  // simulated time

    // Precomputed material
    std::vector<float> noise_;        // white Gaussian noise, unit variance per component
    std::vector<float> wide_table_;   // band-limited noise for the wide-band bursts
    std::vector<float> narrow_table_; // band-limited noise for the hopper

    // Signal state
    std::vector<Tone>  tones_;
    double             fm_phase_ = 0.0;
    double             chirp_phase_ = 0.0;
    double             chirp_freq_ = 0.0;   // cycles/sample relative to the initial centre
    size_t             wide_pos_ = 0;
    double             wide_phase_ = 0.0;
    bool               burst_on_ = false;
    int64_t            burst_remaining_ = 0;
    float              burst_level_ = 0.0f;
    size_t             narrow_pos_ = 0;
    double             hop_phase_ = 0.0;
    double             hop_offset_ = 0.0;
    bool               hop_on_ = false;
    int64_t            hop_remaining_ = 0;
    float              hop_level_ = 0.0f;
    uint32_t           rng_state_[4] = {};

    std::vector<float> mix_;       // interleaved I/Q accumulator for one block
    std::vector<float> envelope_;  // per-sample burst envelope
};

}  // namespace spectro
