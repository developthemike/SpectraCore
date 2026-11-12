#include "dsp/spectrum_engine.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>
#include <utility>

#include <fftw3.h>

#include "dsp/fftw_lock.hpp"
#include "util/log.hpp"
#include "util/thread_util.hpp"

namespace spectro {

std::mutex& fftw_planner_mutex()
{
    static std::mutex mutex;
    return mutex;
}

namespace {

// Full-scale amplitude of the SC16 Q11 sample format.
constexpr float kFullScale = 2048.0f;

// Upper bound on the time FFTW may spend tuning one plan.  This also bounds how
// long stop() can wait for the planner thread.
constexpr double kPlanningTimeLimitSeconds = 2.0;

// dst[j] = float(src[j]) * window2[j] over interleaved I/Q values.  Written as a
// flat element-wise loop so the compiler turns it into wide SIMD.
void convert_and_window(const int16_t* src, const float* window2, float* dst, size_t count)
{
    for (size_t j = 0; j < count; ++j)
        dst[j] = static_cast<float>(src[j]) * window2[j];
}

void accumulate_sum(const float* spectrum, float* accum, size_t bins)
{
    for (size_t k = 0; k < bins; ++k) {
        const float re = spectrum[2 * k];
        const float im = spectrum[2 * k + 1];
        accum[k] += re * re + im * im;
    }
}

void accumulate_max(const float* spectrum, float* accum, size_t bins)
{
    for (size_t k = 0; k < bins; ++k) {
        const float re = spectrum[2 * k];
        const float im = spectrum[2 * k + 1];
        const float p  = re * re + im * im;
        accum[k]       = p > accum[k] ? p : accum[k];
    }
}

int hop_for_overlap(int fft_size, int overlap_pct)
{
    if (overlap_pct >= 63)
        return fft_size / 4;  // 75 %
    if (overlap_pct >= 25)
        return fft_size / 2;  // 50 %
    return fft_size;          // none
}

}  // namespace

const char* detector_name(Detector detector)
{
    switch (detector) {
    case Detector::Average: return "Average (RMS)";
    case Detector::Peak:    return "Peak";
    default:                return "?";
    }
}

// ---------------------------------------------------------------------------
// FftSetup: everything that depends on the FFT size and window.
// ---------------------------------------------------------------------------
struct SpectrumEngine::FftSetup {
    const int        n;
    const WindowType window;
    const bool       tuned;  // planned with FFTW_MEASURE rather than FFTW_ESTIMATE

    fftwf_complex* in      = nullptr;
    fftwf_complex* out     = nullptr;
    float*         window2 = nullptr;  // window, duplicated for interleaved I/Q
    fftwf_plan     plan    = nullptr;

    // Converts |X[k]|^2 of raw samples to power relative to a full-scale sine.
    float power_scale = 1.0f;
    // Spectrum of the window itself at bins 0..4 (it is real and symmetric).  A
    // DC offset d in the samples shows up as d * window_dft[|k|] in bin k.
    float  window_dft[kWindowSpectrumHalfWidth + 1] = {};
    double enbw_bins = 1.0;

    FftSetup(int fft_size, WindowType window_type, bool tune)
        : n(fft_size), window(window_type), tuned(tune)
    {
        in      = fftwf_alloc_complex(static_cast<size_t>(n));
        out     = fftwf_alloc_complex(static_cast<size_t>(n));
        window2 = fftwf_alloc_real(2 * static_cast<size_t>(n));
        if (!in || !out || !window2) {
            release();
            throw std::bad_alloc();
        }

        std::vector<float> w(static_cast<size_t>(n));
        make_window(window, n, w.data());

        double sum = 0.0, sum_sq = 0.0;
        double dft[kWindowSpectrumHalfWidth + 1] = {};
        const double step = 2.0 * std::numbers::pi / n;
        for (int i = 0; i < n; ++i) {
            const double v = w[static_cast<size_t>(i)];
            sum    += v;
            sum_sq += v * v;
            for (int k = 0; k <= kWindowSpectrumHalfWidth; ++k)
                dft[k] += v * std::cos(step * k * i);
            window2[2 * i]     = w[static_cast<size_t>(i)];
            window2[2 * i + 1] = w[static_cast<size_t>(i)];
        }
        for (int k = 0; k <= kWindowSpectrumHalfWidth; ++k)
            window_dft[k] = static_cast<float>(dft[k]);

        enbw_bins = n * sum_sq / (sum * sum);
        const double full_scale_peak = static_cast<double>(kFullScale) * sum;
        power_scale = static_cast<float>(1.0 / (full_scale_peak * full_scale_peak));

        {
            std::lock_guard<std::mutex> lock(fftw_planner_mutex());
            // Measuring the largest transforms can take many seconds; settle for
            // the best plan found within a couple (never worse than an estimate).
            fftwf_set_timelimit(kPlanningTimeLimitSeconds);
            plan = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, tuned ? FFTW_MEASURE : FFTW_ESTIMATE);
        }
        if (!plan) {
            release();
            throw std::runtime_error("FFTW failed to create a plan");
        }
        // FFTW_MEASURE scribbles over the buffers while timing candidate plans.
        std::memset(in, 0, sizeof(fftwf_complex) * static_cast<size_t>(n));
        std::memset(out, 0, sizeof(fftwf_complex) * static_cast<size_t>(n));
    }

    ~FftSetup() { release(); }

    FftSetup(const FftSetup&)            = delete;
    FftSetup& operator=(const FftSetup&) = delete;

private:
    void release()
    {
        if (plan) {
            std::lock_guard<std::mutex> lock(fftw_planner_mutex());
            fftwf_destroy_plan(plan);
            plan = nullptr;
        }
        fftwf_free(in);
        fftwf_free(out);
        fftwf_free(window2);
        in = out = nullptr;
        window2  = nullptr;
    }
};

// ---------------------------------------------------------------------------

SpectrumEngine::SpectrumEngine(IqRing& ring, double sample_rate, int columns)
    : ring_(ring), sample_rate_(sample_rate), columns_(columns)
{
    if (columns < 2 || !std::has_single_bit(static_cast<unsigned>(columns)))
        throw std::invalid_argument("SpectrumEngine: columns must be a power of two");
    if (!(sample_rate > 0.0))
        throw std::invalid_argument("SpectrumEngine: sample rate must be positive");
    if (ring.capacity() < static_cast<size_t>(kMaxFftSize) * 2)
        throw std::invalid_argument("SpectrumEngine: IQ ring is too small");

    queue_data_.resize(static_cast<size_t>(kQueueRows) * static_cast<size_t>(columns_));
    queue_info_.resize(kQueueRows);
}

SpectrumEngine::~SpectrumEngine()
{
    stop();
}

double SpectrumEngine::column_frequency(double c, int columns, int fft_size, double sample_rate)
{
    // With more FFT bins than columns, a column shows the peak of `ratio`
    // adjacent bins and sits at their mean frequency.
    const double ratio = std::max(1.0, static_cast<double>(fft_size) / columns);
    return ((c + 0.5 - 0.5 / ratio) / columns - 0.5) * sample_rate;
}

void SpectrumEngine::set_params(const SpectrumParams& params)
{
    SpectrumParams p = params;
    p.fft_size = std::clamp(static_cast<int>(std::bit_ceil(static_cast<unsigned>(std::max(p.fft_size, 1)))),
                            kMinFftSize, kMaxFftSize);
    p.overlap_pct  = std::clamp(p.overlap_pct, 0, 75);
    p.rows_per_sec = std::clamp(p.rows_per_sec, 0.1, 5000.0);
    {
        std::lock_guard<std::mutex> lock(params_mutex_);
        if (p == requested_)
            return;
        requested_ = p;
    }
    params_generation_.fetch_add(1, std::memory_order_release);
}

SpectrumParams SpectrumEngine::params() const
{
    std::lock_guard<std::mutex> lock(params_mutex_);
    return requested_;
}

void SpectrumEngine::mark_gap()
{
    gap_requested_.store(true, std::memory_order_release);
}

EngineStats SpectrumEngine::stats() const
{
    EngineStats s;
    s.rows_emitted = stat_rows_emitted_.load(std::memory_order_relaxed);
    s.rows_dropped = stat_rows_dropped_.load(std::memory_order_relaxed);
    s.frames       = stat_frames_.load(std::memory_order_relaxed);
    s.cpu_load     = stat_load_.load(std::memory_order_relaxed);
    s.fft_size     = stat_fft_size_.load(std::memory_order_relaxed);
    s.row_period_s = stat_row_period_.load(std::memory_order_relaxed);
    s.rbw_hz       = stat_rbw_.load(std::memory_order_relaxed);
    return s;
}

void SpectrumEngine::start()
{
    if (thread_.joinable())
        return;

    {
        std::lock_guard<std::mutex> lock(params_mutex_);
        active_ = requested_;
    }
    // Start immediately with a quickly-made plan; the planner thread replaces it
    // with a tuned one in the background.
    fft_ = std::make_shared<FftSetup>(active_.fft_size, active_.window, false);
    configure_row();

    stop_.store(false);
    {
        // Ask for the tuned version of the plan we start with.  This is posted
        // before the DSP thread exists, so a request of its own always comes later.
        std::lock_guard<std::mutex> lock(planner_mutex_);
        planner_stop_        = false;
        planner_fft_size_    = active_.fft_size;
        planner_window_      = active_.window;
        planner_want_quick_  = false;
        planner_has_request_ = true;
    }
    planner_thread_ = std::thread([this] { planner_run(); });
    thread_         = std::thread([this] { run(); });
}

void SpectrumEngine::stop()
{
    if (thread_.joinable()) {
        stop_.store(true, std::memory_order_release);
        ring_.close();  // wakes the DSP thread if it is waiting for samples
        thread_.join();
    }
    if (planner_thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(planner_mutex_);
            planner_stop_ = true;
        }
        planner_cv_.notify_one();
        planner_thread_.join();
    }
}

void SpectrumEngine::request_setup(int fft_size, WindowType window)
{
    {
        std::lock_guard<std::mutex> lock(planner_mutex_);
        planner_fft_size_    = fft_size;
        planner_window_      = window;
        planner_want_quick_  = true;
        planner_has_request_ = true;
    }
    planner_cv_.notify_one();
}

void SpectrumEngine::retire_setup(std::shared_ptr<FftSetup> setup)
{
    if (!setup)
        return;
    {
        std::lock_guard<std::mutex> lock(planner_mutex_);
        planner_retired_.push_back(std::move(setup));
    }
    planner_cv_.notify_one();
}

// Builds FFT setups off the DSP thread, so the sample stream keeps being
// processed with the old setup while a new plan is prepared.  Each request
// yields a quick (FFTW_ESTIMATE) setup right away, followed by a tuned one.
void SpectrumEngine::planner_run()
{
    set_current_thread_name("spectro-plan");

    std::unique_lock<std::mutex> lock(planner_mutex_);
    for (;;) {
        planner_cv_.wait(lock, [this] {
            return planner_stop_ || planner_has_request_ || !planner_retired_.empty();
        });

        // Dispose of setups the DSP thread is done with (outside our lock: the
        // destructor takes FFTW's).
        if (!planner_retired_.empty()) {
            std::vector<std::shared_ptr<FftSetup>> retired;
            retired.swap(planner_retired_);
            lock.unlock();
            retired.clear();
            lock.lock();
        }
        if (planner_stop_)
            return;
        if (!planner_has_request_)
            continue;

        const int        fft_size = planner_fft_size_;
        const WindowType window   = planner_window_;
        const bool       quick    = planner_want_quick_;
        planner_has_request_      = false;
        lock.unlock();

        for (int stage = quick ? 0 : 1; stage < 2; ++stage) {
            std::shared_ptr<FftSetup> setup;
            try {
                setup = std::make_shared<FftSetup>(fft_size, window, stage == 1);
            } catch (const std::exception& e) {
                LOG_ERROR("Could not prepare a %d-point FFT: %s", fft_size, e.what());
                break;
            }

            std::shared_ptr<FftSetup> replaced;
            {
                std::lock_guard<std::mutex> publish(planner_mutex_);
                if (planner_has_request_ || planner_stop_)
                    break;  // superseded by a newer request; `setup` is dropped here, on this thread
                replaced        = std::move(planner_result_);
                planner_result_ = std::move(setup);
                planner_result_ready_.store(true, std::memory_order_release);
            }
            // An unclaimed earlier result (if any) is destroyed here, outside the lock.
        }
        lock.lock();
    }
}

// Derives the framing from active_ and fft_, and restarts the current row.
void SpectrumEngine::configure_row()
{
    const int n = fft_->n;
    hop_        = hop_for_overlap(n, active_.overlap_pct);

    const double frames_per_sec = sample_rate_ / hop_;
    const double rows_per_sec   = std::clamp(active_.rows_per_sec, 0.1, 5000.0);
    frames_per_row_ = static_cast<int>(
        std::clamp<long long>(std::llround(frames_per_sec / rows_per_sec), 1, 1LL << 24));

    accum_.assign(static_cast<size_t>(n), 0.0f);
    frames_in_row_ = 0;

    // DC tracking loop: ~5 ms time constant, bounded so that one frame's noise
    // never moves the estimate much.
    const double hop_time = hop_ / sample_rate_;
    dc_loop_gain_steady_  = static_cast<float>(std::clamp(1.0 - std::exp(-hop_time / 0.005), 0.01, 0.25));

    stat_fft_size_.store(n, std::memory_order_relaxed);
    stat_row_period_.store(static_cast<double>(frames_per_row_) * hop_ / sample_rate_,
                           std::memory_order_relaxed);
    stat_rbw_.store(fft_->enbw_bins * sample_rate_ / n, std::memory_order_relaxed);
}

void SpectrumEngine::process_frame()
{
    FftSetup&    fft = *fft_;
    const size_t n   = static_cast<size_t>(fft.n);

    // Window the frame straight out of the ring (it may wrap around the end).
    const IqRing::ReadSpans spans = ring_.peek(n);
    float*                  in    = reinterpret_cast<float*>(fft.in);
    convert_and_window(spans.first, fft.window2, in, 2 * spans.first_count);
    if (spans.second_count > 0) {
        convert_and_window(spans.second, fft.window2 + 2 * spans.first_count,
                           in + 2 * spans.first_count, 2 * spans.second_count);
    }

    fftwf_execute(fft.plan);
    float* out = reinterpret_cast<float*>(fft.out);

    if (active_.dc_block) {
        // A constant offset d in the samples contributes d * W[k] to bin k, where
        // W is the window's own spectrum: nine bins around DC and nothing else.
        // Subtracting it there is exact and costs nothing compared with touching
        // every sample.  The estimate d is steered by a slow feedback loop that
        // drives the DC bin to zero; it is far too slow (tens of Hz) to affect
        // real signals near the centre frequency.
        const float w0 = fft.window_dft[0];
        dc_re_ += dc_loop_gain_ * (out[0] / w0 - dc_re_);
        dc_im_ += dc_loop_gain_ * (out[1] / w0 - dc_im_);
        if (dc_loop_gain_ > dc_loop_gain_steady_) {
            // Acquire in a few frames, then settle to the steady-state bandwidth.
            dc_loop_gain_ = std::max(dc_loop_gain_steady_, dc_loop_gain_ * 0.5f);
        }
        for (int k = -kWindowSpectrumHalfWidth; k <= kWindowSpectrumHalfWidth; ++k) {
            const size_t bin  = static_cast<size_t>(k < 0 ? static_cast<int>(n) + k : k);
            const float  gain = fft.window_dft[k < 0 ? -k : k];
            out[2 * bin]     -= dc_re_ * gain;
            out[2 * bin + 1] -= dc_im_ * gain;
        }
    }

    if (active_.detector == Detector::Peak)
        accumulate_max(out, accum_.data(), n);
    else
        accumulate_sum(out, accum_.data(), n);
}

void SpectrumEngine::emit_row(bool gap)
{
    const uint32_t head = queue_head_.load(std::memory_order_relaxed);
    const uint32_t tail = queue_tail_.load(std::memory_order_acquire);

    if (head - tail >= kQueueRows) {
        stat_rows_dropped_.fetch_add(1, std::memory_order_relaxed);
    } else {
        const size_t slot = head % kQueueRows;
        float*       row  = queue_data_.data() + slot * static_cast<size_t>(columns_);

        if (gap) {
            std::fill_n(row, columns_, kFloorDb);
        } else {
            const int    n     = fft_->n;
            const float* accum = accum_.data();
            const float  scale = fft_->power_scale *
                (active_.detector == Detector::Average ? 1.0f / static_cast<float>(frames_in_row_) : 1.0f);

            if (n >= columns_) {
                // Peak-reduce `ratio` adjacent bins into one column so that a
                // narrow carrier can never fall between two columns.  The halves
                // are swapped to put negative frequencies on the left.
                const int ratio = n / columns_;
                const int half  = columns_ / 2;
                for (int c = 0; c < columns_; ++c) {
                    const float* src = accum + (c < half ? n / 2 + c * ratio : (c - half) * ratio);
                    float        m   = src[0];
                    for (int j = 1; j < ratio; ++j)
                        m = src[j] > m ? src[j] : m;
                    row[c] = std::max(10.0f * std::log10(m * scale + 1e-30f), kFloorDb);
                }
            } else {
                // Fewer bins than columns: interpolate in dB.
                scratch_db_.resize(static_cast<size_t>(n));
                for (int b = 0; b < n; ++b) {
                    const float p = accum[(b + n / 2) % n];
                    scratch_db_[static_cast<size_t>(b)] =
                        std::max(10.0f * std::log10(p * scale + 1e-30f), kFloorDb);
                }
                const float step = static_cast<float>(n) / static_cast<float>(columns_);
                for (int c = 0; c < columns_; ++c) {
                    const float pos  = static_cast<float>(c) * step;
                    const int   i0   = static_cast<int>(pos);
                    const int   i1   = std::min(i0 + 1, n - 1);
                    const float frac = pos - static_cast<float>(i0);
                    row[c] = scratch_db_[static_cast<size_t>(i0)] * (1.0f - frac) +
                             scratch_db_[static_cast<size_t>(i1)] * frac;
                }
            }
        }

        RowInfo& info = queue_info_[slot];
        info.index    = row_index_;
        info.period_s = gap ? 0.0 : static_cast<double>(frames_in_row_) * hop_ / sample_rate_;
        info.fft_size = fft_->n;
        info.frames   = gap ? 0 : frames_in_row_;
        info.gap      = gap;

        queue_head_.store(head + 1, std::memory_order_release);
        stat_rows_emitted_.fetch_add(1, std::memory_order_relaxed);
    }

    ++row_index_;
    std::fill(accum_.begin(), accum_.end(), 0.0f);
    frames_in_row_ = 0;
}

void SpectrumEngine::run()
{
    set_current_thread_name("spectro-dsp");

    uint64_t       seen_generation = params_generation_.load(std::memory_order_acquire);
    SpectrumParams target          = active_;
    {
        // Parameters may have changed between start() and here.
        std::lock_guard<std::mutex> lock(params_mutex_);
        target = requested_;
    }
    bool params_dirty = !(target == active_);

    auto   load_wall = std::chrono::steady_clock::now();
    double load_cpu  = current_thread_cpu_seconds();

    while (!stop_.load(std::memory_order_acquire)) {
        // 1. Parameter changes.
        const uint64_t generation = params_generation_.load(std::memory_order_acquire);
        if (generation != seen_generation || params_dirty) {
            seen_generation = generation;
            params_dirty    = false;
            {
                std::lock_guard<std::mutex> lock(params_mutex_);
                target = requested_;
            }
            if (target.fft_size != fft_->n || target.window != fft_->window)
                request_setup(target.fft_size, target.window);

            // Everything except the FFT shape applies immediately; the shape
            // follows once the planner delivers.
            SpectrumParams effective = target;
            effective.fft_size       = fft_->n;
            effective.window         = fft_->window;
            if (!(effective == active_)) {
                active_ = effective;
                configure_row();
            }
        }

        // 2. A new FFT setup has been prepared.
        if (planner_result_ready_.load(std::memory_order_acquire)) {
            std::shared_ptr<FftSetup> next;
            {
                std::lock_guard<std::mutex> lock(planner_mutex_);
                next = std::move(planner_result_);
                planner_result_ready_.store(false, std::memory_order_release);
            }
            if (next && next->n == target.fft_size && next->window == target.window) {
                const bool same_shape = next->n == fft_->n && next->window == fft_->window;
                retire_setup(std::exchange(fft_, std::move(next)));
                active_.fft_size = fft_->n;
                active_.window   = fft_->window;
                if (!same_shape)
                    configure_row();  // otherwise: seamless switch to the tuned plan
            } else {
                retire_setup(std::move(next));  // prepared for settings that have changed again
            }
        }

        // 3. Discontinuity requested by the application.
        if (gap_requested_.exchange(false, std::memory_order_acq_rel)) {
            emit_row(true);
            dc_loop_gain_ = 1.0f;  // re-acquire: the offset depends on tuning and gain
        }

        // 4. One FFT frame.
        if (!ring_.wait_readable(static_cast<size_t>(fft_->n)))
            break;  // ring closed and drained
        process_frame();
        ring_.consume(static_cast<size_t>(hop_));
        stat_frames_.fetch_add(1, std::memory_order_relaxed);

        if (++frames_in_row_ >= frames_per_row_) {
            emit_row(false);

            const auto   now  = std::chrono::steady_clock::now();
            const double wall = std::chrono::duration<double>(now - load_wall).count();
            if (wall >= 0.25) {
                const double cpu = current_thread_cpu_seconds();
                stat_load_.store(static_cast<float>((cpu - load_cpu) / wall), std::memory_order_relaxed);
                load_wall = now;
                load_cpu  = cpu;
            }
        }
    }
}

int SpectrumEngine::pop_rows(float* dst, int max_rows, RowInfo* infos)
{
    const uint32_t tail  = queue_tail_.load(std::memory_order_relaxed);
    const uint32_t head  = queue_head_.load(std::memory_order_acquire);
    const uint32_t avail = head - tail;
    const int      count = static_cast<int>(std::min<uint32_t>(avail, static_cast<uint32_t>(std::max(max_rows, 0))));

    const size_t row_bytes = static_cast<size_t>(columns_) * sizeof(float);
    for (int i = 0; i < count; ++i) {
        const size_t slot = (tail + static_cast<uint32_t>(i)) % kQueueRows;
        std::memcpy(dst + static_cast<size_t>(i) * static_cast<size_t>(columns_),
                    queue_data_.data() + slot * static_cast<size_t>(columns_), row_bytes);
        if (infos)
            infos[i] = queue_info_[slot];
    }
    queue_tail_.store(tail + static_cast<uint32_t>(count), std::memory_order_release);
    return count;
}

}  // namespace spectro
