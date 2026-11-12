// Turns the raw IQ stream into spectrogram rows.
//
// A dedicated thread pulls overlapping, windowed frames out of the IqRing,
// runs an FFT on every one of them (no samples are skipped) and combines the
// resulting power spectra into rows: one row summarises a fixed amount of
// signal time, so the waterfall scrolls at a constant, exact rate.  Finished
// rows are handed to the renderer through a lock-free queue.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "dsp/window.hpp"
#include "util/iq_ring.hpp"

namespace spectro {

enum class Detector : int {
    Average = 0,  // mean power of all frames in the row (true RMS)
    Peak,         // maximum power seen in the row (catches short bursts)
    Count
};
const char* detector_name(Detector detector);

struct SpectrumParams {
    int        fft_size     = 4096;  // power of two
    WindowType window       = WindowType::Hann;
    int        overlap_pct  = 50;    // 0, 50 or 75
    Detector   detector     = Detector::Average;
    double     rows_per_sec = 50.0;  // requested waterfall speed
    bool       dc_block     = true;  // cancel the receiver's DC offset / LO leakage

    bool operator==(const SpectrumParams&) const = default;
};

struct RowInfo {
    uint64_t index    = 0;      // running row counter
    double   period_s = 0.0;    // signal time covered by the row
    int      fft_size = 0;
    int      frames   = 0;      // FFT frames combined into the row
    bool     gap      = false;  // marks a discontinuity; the row carries no data
};

struct EngineStats {
    uint64_t rows_emitted = 0;
    uint64_t rows_dropped = 0;  // rows discarded because the renderer fell behind
    uint64_t frames       = 0;  // FFTs computed
    float    cpu_load     = 0;  // fraction of one core used by the DSP thread
    int      fft_size     = 0;  // FFT size currently in use
    double   row_period_s = 0;  // actual signal time per row
    double   rbw_hz       = 0;  // resolution (equivalent noise) bandwidth
};

class SpectrumEngine {
public:
    static constexpr int kMinFftSize = 256;
    static constexpr int kMaxFftSize = 1 << 18;

    // Level stored in rows that carry no data.
    static constexpr float kFloorDb = -200.0f;

    // `columns` is the width of every output row and must be a power of two.
    // Column c is centred on the frequency offset column_frequency(c).
    SpectrumEngine(IqRing& ring, double sample_rate, int columns);
    ~SpectrumEngine();

    SpectrumEngine(const SpectrumEngine&)            = delete;
    SpectrumEngine& operator=(const SpectrumEngine&) = delete;

    void start();
    void stop();  // the ring must be closed first (or still be receiving data)

    // May be called from any thread; takes effect at the next frame boundary.
    void           set_params(const SpectrumParams& params);
    SpectrumParams params() const;

    // Inserts a gap row and restarts the row in progress.  Use after anything
    // that invalidates the signal's continuity, e.g. retuning the radio.
    void mark_gap();

    // Copies up to max_rows finished rows (oldest first) into dst, which must
    // hold max_rows * columns() floats (dBFS).  `infos` may be null, otherwise it
    // must hold max_rows entries.  Returns the number of rows copied.  Must only
    // be called from one thread.
    int pop_rows(float* dst, int max_rows, RowInfo* infos = nullptr);

    int         columns() const { return columns_; }
    double      sample_rate() const { return sample_rate_; }
    EngineStats stats() const;

    // Frequency offset from the tuned centre, in Hz, of output column c for the
    // given FFT size.  Fractional columns are allowed.
    static double column_frequency(double c, int columns, int fft_size, double sample_rate);

private:
    struct FftSetup;

    void run();
    void planner_run();
    void request_setup(int fft_size, WindowType window);
    void retire_setup(std::shared_ptr<FftSetup> setup);
    void configure_row();
    void process_frame();
    void emit_row(bool gap);

    IqRing&      ring_;
    const double sample_rate_;
    const int    columns_;

    // --- parameters (written by any thread, read by the DSP thread) ---
    mutable std::mutex    params_mutex_;
    SpectrumParams        requested_;
    std::atomic<uint64_t> params_generation_{0};
    std::atomic<bool>     gap_requested_{false};

    // --- FFT plan construction (FFTW planning can take seconds for big sizes) ---
    std::thread                 planner_thread_;
    std::mutex                  planner_mutex_;
    std::condition_variable     planner_cv_;
    bool                        planner_stop_ = false;
    bool                        planner_has_request_ = false;
    bool                        planner_want_quick_ = true;
    int                         planner_fft_size_ = 0;
    WindowType                  planner_window_ = WindowType::Hann;
    std::shared_ptr<FftSetup>   planner_result_;
    std::atomic<bool>           planner_result_ready_{false};
    // Setups the DSP thread no longer uses.  Destroying an FFTW plan needs the
    // planner lock, which may be held for seconds while a big plan is being
    // measured - so the planner thread does it, never the DSP thread.
    std::vector<std::shared_ptr<FftSetup>> planner_retired_;

    // --- DSP thread state ---
    std::thread               thread_;
    std::atomic<bool>         stop_{false};
    std::shared_ptr<FftSetup> fft_;
    SpectrumParams            active_;
    int                       hop_ = 0;
    int                       frames_per_row_ = 1;
    int                       frames_in_row_ = 0;
    std::vector<float>        accum_;      // per-bin accumulated power, FFT order
    std::vector<float>        scratch_db_; // used when upsampling small FFTs
    float                     dc_re_ = 0.0f, dc_im_ = 0.0f;  // DC offset estimate, raw sample units
    float                     dc_loop_gain_ = 1.0f;          // starts high to acquire quickly
    float                     dc_loop_gain_steady_ = 0.05f;
    uint64_t                  row_index_ = 0;

    // --- output queue (single producer, single consumer) ---
    static constexpr uint32_t kQueueRows = 512;  // power of two
    std::vector<float>    queue_data_;
    std::vector<RowInfo>  queue_info_;
    alignas(64) std::atomic<uint32_t> queue_head_{0};  // next slot to write
    alignas(64) std::atomic<uint32_t> queue_tail_{0};  // next slot to read

    // --- statistics ---
    std::atomic<uint64_t> stat_rows_emitted_{0};
    std::atomic<uint64_t> stat_rows_dropped_{0};
    std::atomic<uint64_t> stat_frames_{0};
    std::atomic<float>    stat_load_{0.0f};
    std::atomic<int>      stat_fft_size_{0};
    std::atomic<double>   stat_row_period_{0.0};
    std::atomic<double>   stat_rbw_{0.0};
};

}  // namespace spectro
