#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numbers>
#include <random>
#include <thread>
#include <vector>

#include "dsp/spectrum_engine.hpp"
#include "dsp/window.hpp"
#include "test_framework.hpp"
#include "util/iq_ring.hpp"

using namespace spectro;

namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

// Feeds samples to a running SpectrumEngine and collects its rows.
struct Harness {
    IqRing         ring;
    SpectrumEngine engine;
    int            columns;

    Harness(double sample_rate, int columns_, const SpectrumParams& params)
        : ring(size_t{1} << 21), engine(ring, sample_rate, columns_), columns(columns_)
    {
        engine.set_params(params);
        engine.start();
    }

    void feed(const std::vector<int16_t>& iq)
    {
        const size_t total = iq.size() / 2;
        size_t       done  = 0;
        while (done < total) {
            auto span = ring.write_span(total - done);
            if (span.samples == 0) {
                std::this_thread::yield();
                continue;
            }
            std::memcpy(span.data, iq.data() + 2 * done, span.samples * 2 * sizeof(int16_t));
            ring.commit(span.samples);
            done += span.samples;
        }
    }

    // Waits until `count` rows have been produced (or the timeout expires).
    std::vector<std::vector<float>> collect(int count, std::vector<RowInfo>* infos = nullptr,
                                            double timeout_s = 20.0)
    {
        std::vector<std::vector<float>> rows;
        std::vector<float>              row(static_cast<size_t>(columns));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
        while (static_cast<int>(rows.size()) < count && std::chrono::steady_clock::now() < deadline) {
            RowInfo info;
            if (engine.pop_rows(row.data(), 1, &info) == 1) {
                rows.push_back(row);
                if (infos)
                    infos->push_back(info);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        return rows;
    }
};

// amplitude * exp(j 2 pi f t) + dc, quantised to int16.
std::vector<int16_t> make_tone(double sample_rate, double freq, double amplitude, size_t samples,
                               double dc_i = 0.0, double dc_q = 0.0)
{
    std::vector<int16_t> iq(2 * samples);
    for (size_t n = 0; n < samples; ++n) {
        const double phase = kTwoPi * freq * static_cast<double>(n) / sample_rate;
        iq[2 * n]     = static_cast<int16_t>(std::lround(amplitude * std::cos(phase) + dc_i));
        iq[2 * n + 1] = static_cast<int16_t>(std::lround(amplitude * std::sin(phase) + dc_q));
    }
    return iq;
}

void add_noise(std::vector<int16_t>& iq, double sigma, unsigned seed)
{
    std::mt19937                     rng(seed);
    std::normal_distribution<double> gauss(0.0, sigma);
    for (int16_t& v : iq)
        v = static_cast<int16_t>(std::clamp<long>(std::lround(v + gauss(rng)), -32768, 32767));
}

int peak_column(const std::vector<float>& row)
{
    return static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin());
}

float median(std::vector<float> values)
{
    std::nth_element(values.begin(), values.begin() + static_cast<long>(values.size() / 2), values.end());
    return values[values.size() / 2];
}

SpectrumParams basic_params(int fft_size)
{
    SpectrumParams p;
    p.fft_size     = fft_size;
    p.window       = WindowType::Hann;
    p.overlap_pct  = 50;
    p.detector     = Detector::Average;
    p.rows_per_sec = 100.0;
    p.dc_block     = false;
    return p;
}

}  // namespace

TEST(window_properties)
{
    CHECK_NEAR(window_enbw_bins(WindowType::Rectangular), 1.0, 1e-9);
    CHECK_NEAR(window_enbw_bins(WindowType::Hann), 1.5, 1e-9);
    CHECK_NEAR(window_enbw_bins(WindowType::Hamming), 1.3628, 1e-3);
    CHECK_NEAR(window_enbw_bins(WindowType::BlackmanHarris), 2.0044, 1e-3);
    CHECK_NEAR(window_enbw_bins(WindowType::FlatTop), 3.7702, 1e-3);

    // Periodic Hann: zero at n = 0, one at n = N/2, symmetric about N/2.
    std::vector<float> w(64);
    make_window(WindowType::Hann, 64, w.data());
    CHECK_NEAR(w[0], 0.0, 1e-7);
    CHECK_NEAR(w[32], 1.0, 1e-7);
    CHECK_NEAR(w[10], w[54], 1e-6);

    WindowType type;
    CHECK(window_from_name("hann", type) && type == WindowType::Hann);
    CHECK(window_from_name("Blackman-Harris", type) && type == WindowType::BlackmanHarris);
    CHECK(!window_from_name("kaiser", type));
}

TEST(column_frequency_mapping)
{
    const double fs = 20e6;
    // One bin per column: column W/2 is DC, column 0 is -fs/2.
    CHECK_NEAR(SpectrumEngine::column_frequency(1024, 2048, 2048, fs), 0.0, 1e-6);
    CHECK_NEAR(SpectrumEngine::column_frequency(0, 2048, 2048, fs), -10e6, 1e-6);
    CHECK_NEAR(SpectrumEngine::column_frequency(2047, 2048, 2048, fs), 10e6 - fs / 2048, 1e-6);
    // Two bins per column: the column sits between its two bins.
    CHECK_NEAR(SpectrumEngine::column_frequency(1024, 2048, 4096, fs), 0.5 * fs / 4096, 1e-6);
    // Fewer bins than columns: same as one bin per column.
    CHECK_NEAR(SpectrumEngine::column_frequency(1024, 2048, 1024, fs), 0.0, 1e-6);
}

TEST(tone_level_and_frequency_are_exact)
{
    // A bin-centred tone must read its true amplitude in dBFS and land in the
    // column whose centre frequency matches, on either side of DC.
    const double fs      = 20e6;
    const int    n       = 2048;
    const int    columns = 2048;
    for (const int bin : {-700, -1, 3, 512, 1000}) {
        const double freq      = bin * fs / n;
        const double amplitude = 2048.0 * std::pow(10.0, -20.0 / 20.0);  // -20 dBFS
        Harness      h(fs, columns, basic_params(n));
        h.feed(make_tone(fs, freq, amplitude, 700'000));
        const auto rows = h.collect(3);
        CHECK(rows.size() == 3);
        if (rows.size() < 3)
            return;
        const auto& row  = rows[1];
        const int   peak = peak_column(row);
        CHECK_EQ(peak, columns / 2 + bin);
        CHECK_NEAR(SpectrumEngine::column_frequency(peak, columns, n, fs), freq, 1.0);
        CHECK_NEAR(row[static_cast<size_t>(peak)], -20.0, 0.05);
        // Hann leaks into the two neighbours at -6 dB and essentially nowhere else.
        CHECK_NEAR(row[static_cast<size_t>(peak + 1)], -26.02, 0.1);
        CHECK(row[static_cast<size_t>((peak + 300) % columns)] < -90.0f);
    }
}

TEST(full_scale_tone_reads_zero_dbfs)
{
    const double fs = 1e6;
    Harness      h(fs, 1024, basic_params(1024));
    h.feed(make_tone(fs, 100 * fs / 1024, 2047.0, 100'000));
    const auto rows = h.collect(2);
    CHECK(rows.size() == 2);
    if (rows.size() < 2)
        return;
    CHECK_NEAR(*std::max_element(rows[1].begin(), rows[1].end()), 0.0, 0.05);
}

TEST(noise_floor_matches_theory)
{
    // White noise of variance sigma^2 per component puts
    //   2 sigma^2 / FS^2 * ENBW / N
    // into every bin, whatever the window.
    const double fs    = 20e6;
    const int    n     = 4096;
    const double sigma = 20.0;
    for (const WindowType window : {WindowType::Hann, WindowType::BlackmanHarris, WindowType::Rectangular}) {
        SpectrumParams p = basic_params(n);
        p.window         = window;
        p.rows_per_sec   = 20.0;  // ~490 frames per row: a tight average
        Harness h(fs, n, p);
        std::vector<int16_t> iq(2 * 3'000'000, 0);
        add_noise(iq, sigma, 7);
        h.feed(iq);
        const auto rows = h.collect(2);
        CHECK(rows.size() == 2);
        if (rows.size() < 2)
            return;
        const double expected = 10.0 * std::log10(2.0 * sigma * sigma / (2048.0 * 2048.0) *
                                                  window_enbw_bins(window) / n);
        // The median of a chi-square average sits a touch below its mean.
        CHECK_NEAR(median(rows[1]), expected, 0.3);
    }
}

TEST(peak_detector_holds_short_bursts)
{
    // A burst occupying a small part of one row: the average detector dilutes
    // it, the peak detector reports its full level.
    const double fs = 20e6;
    const int    n  = 1024;
    std::vector<int16_t> iq(2 * 1'200'000, 0);
    add_noise(iq, 3.0, 11);
    const double freq = 200 * fs / n;
    const double amp  = 2048.0 * std::pow(10.0, -30.0 / 20.0);
    for (size_t k = 300'000; k < 300'000 + 4 * 1024; ++k) {  // ~0.2 ms long
        const double phase = kTwoPi * freq * static_cast<double>(k) / fs;
        iq[2 * k]     = static_cast<int16_t>(iq[2 * k] + std::lround(amp * std::cos(phase)));
        iq[2 * k + 1] = static_cast<int16_t>(iq[2 * k + 1] + std::lround(amp * std::sin(phase)));
    }

    float levels[2] = {};
    for (int d = 0; d < 2; ++d) {
        SpectrumParams p = basic_params(n);
        p.detector       = d == 0 ? Detector::Average : Detector::Peak;
        p.rows_per_sec   = 50.0;  // 20 ms rows: the burst is 1 % of a row
        Harness h(fs, n, p);
        h.feed(iq);
        const auto rows = h.collect(2);
        CHECK(rows.size() == 2);
        if (rows.size() < 2)
            return;
        // The burst starts 15 ms in, i.e. inside the first row.
        levels[d] = rows[0][static_cast<size_t>(n / 2 + 200)];
    }
    CHECK_NEAR(levels[1], -30.0, 0.3);  // peak: full level
    CHECK(levels[0] < -45.0f);          // average: diluted by ~20 dB
}

TEST(dc_offset_is_removed)
{
    const double fs = 20e6;
    const int    n  = 4096;
    // A large DC offset (-24 dBFS) under a weak noise floor, plus a tone to
    // prove real signals are untouched.
    std::vector<int16_t> iq = make_tone(fs, 300 * fs / n, 64.0, 2'000'000, 130.0, -90.0);
    add_noise(iq, 4.0, 3);

    for (const bool block : {false, true}) {
        SpectrumParams p = basic_params(n);
        p.dc_block       = block;
        p.rows_per_sec   = 50.0;
        Harness h(fs, n, p);
        h.feed(iq);
        const auto rows = h.collect(4);
        CHECK(rows.size() == 4);
        if (rows.size() < 4)
            return;
        const auto& row    = rows[3];
        const float floor  = median(row);
        const float centre = *std::max_element(row.begin() + n / 2 - 4, row.begin() + n / 2 + 5);
        if (block) {
            // Nothing left at DC beyond the noise floor (allowing for noise ripple).
            CHECK(centre < floor + 4.0f);
        } else {
            const double dc_dbfs = 20.0 * std::log10(std::hypot(130.0, -90.0) / 2048.0);
            CHECK_NEAR(row[static_cast<size_t>(n / 2)], dc_dbfs, 0.1);
        }
        CHECK_NEAR(row[static_cast<size_t>(n / 2 + 300)], 20.0 * std::log10(64.0 / 2048.0), 0.2);
    }
}

TEST(dc_removal_works_for_every_window)
{
    const double fs = 20e6;
    const int    n  = 2048;
    std::vector<int16_t> iq(2 * 2'000'000);
    for (size_t k = 0; k < iq.size(); k += 2) {
        iq[k]     = 200;
        iq[k + 1] = -150;
    }
    add_noise(iq, 4.0, 5);
    for (const WindowType window : {WindowType::Rectangular, WindowType::Hann, WindowType::Hamming,
                                    WindowType::BlackmanHarris, WindowType::FlatTop}) {
        SpectrumParams p = basic_params(n);
        p.window         = window;
        p.dc_block       = true;
        p.rows_per_sec   = 50.0;
        Harness h(fs, n, p);
        h.feed(iq);
        const auto rows = h.collect(4);
        CHECK(rows.size() == 4);
        if (rows.size() < 4)
            return;
        const auto& row    = rows[3];
        const float centre = *std::max_element(row.begin() + n / 2 - 6, row.begin() + n / 2 + 7);
        CHECK(centre < median(row) + 4.0f);
    }
}

TEST(narrow_carrier_survives_bin_reduction)
{
    // 8 FFT bins per column: a single-bin carrier must still show at full level.
    const double fs      = 20e6;
    const int    n       = 16384;
    const int    columns = 2048;
    const int    bin     = 3333;  // FFT bin (not column), positive frequency
    const double freq    = bin * fs / n;
    Harness      h(fs, columns, basic_params(n));
    h.feed(make_tone(fs, freq, 2048.0 * std::pow(10.0, -40.0 / 20.0), 1'000'000));
    const auto rows = h.collect(3);
    CHECK(rows.size() == 3);
    if (rows.size() < 3)
        return;
    const int peak = peak_column(rows[1]);
    CHECK_EQ(peak, columns / 2 + bin / 8);
    CHECK_NEAR(rows[1][static_cast<size_t>(peak)], -40.0, 0.1);
    // The column's nominal frequency is within half a column of the carrier.
    CHECK_NEAR(SpectrumEngine::column_frequency(peak, columns, n, fs), freq, 0.5 * fs / columns);
}

TEST(small_fft_is_interpolated_across_columns)
{
    const double fs      = 20e6;
    const int    n       = 512;
    const int    columns = 2048;
    const int    bin     = 100;
    Harness      h(fs, columns, basic_params(n));
    h.feed(make_tone(fs, bin * fs / n, 204.8, 700'000));
    const auto rows = h.collect(3);
    CHECK(rows.size() == 3);
    if (rows.size() < 3)
        return;
    const int peak = peak_column(rows[1]);
    CHECK_EQ(peak, columns / 2 + bin * 4);
    CHECK_NEAR(rows[1][static_cast<size_t>(peak)], -20.0, 0.1);
    CHECK_NEAR(SpectrumEngine::column_frequency(peak, columns, n, fs), bin * fs / n, 1.0);
}

TEST(rows_cover_exact_signal_time)
{
    const double   fs = 20e6;
    SpectrumParams p  = basic_params(4096);
    p.rows_per_sec    = 50.0;
    Harness h(fs, 2048, p);

    const size_t samples = 5'000'000;
    std::vector<int16_t> iq(2 * samples, 0);
    add_noise(iq, 5.0, 1);
    h.feed(iq);

    // hop = 2048 -> 9765.6 frames/s -> 195 frames per row -> 399360 samples per row.
    const size_t frames        = (samples - 4096) / 2048 + 1;
    const int    expected_rows = static_cast<int>(frames / 195);
    std::vector<RowInfo> infos;
    const auto rows = h.collect(expected_rows, &infos);
    CHECK_EQ(static_cast<int>(rows.size()), expected_rows);
    if (infos.empty())
        return;
    CHECK_EQ(infos.front().frames, 195);
    CHECK_NEAR(infos.front().period_s, 195.0 * 2048.0 / fs, 1e-12);
    CHECK_EQ(infos.front().index, uint64_t{0});
    CHECK_EQ(infos.back().index, static_cast<uint64_t>(expected_rows - 1));
    CHECK_NEAR(h.engine.stats().row_period_s, 195.0 * 2048.0 / fs, 1e-12);
    CHECK_NEAR(h.engine.stats().rbw_hz, 1.5 * fs / 4096, 1e-3);

    // No further row may appear: the remaining samples do not fill one.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::vector<float> extra(2048);
    CHECK_EQ(h.engine.pop_rows(extra.data(), 1), 0);
}

TEST(parameters_change_while_running)
{
    const double   fs = 20e6;
    SpectrumParams p  = basic_params(1024);
    Harness        h(fs, 1024, p);

    const double amplitude = 204.8;  // -20 dBFS
    const double freq      = 2e6;    // bin-centred for neither size: use a flat-top-free tolerance
    const auto   block     = make_tone(fs, freq, amplitude, 1'000'000);

    h.feed(block);
    std::vector<RowInfo> infos;
    auto rows = h.collect(2, &infos);
    CHECK(rows.size() == 2);
    if (rows.size() < 2)
        return;
    CHECK_EQ(infos[0].fft_size, 1024);

    // Switch to a larger FFT and a different window; the engine keeps running
    // with the old setup until the new plan is ready.
    p.fft_size = 8192;
    p.window   = WindowType::BlackmanHarris;
    h.engine.set_params(p);

    bool switched = false;
    for (int attempt = 0; attempt < 100 && !switched; ++attempt) {
        h.feed(block);
        infos.clear();
        rows = h.collect(1, &infos, 2.0);
        while (true) {  // drain
            std::vector<RowInfo> more;
            auto extra = h.collect(1, &more, 0.02);
            if (extra.empty())
                break;
            rows  = extra;
            infos = more;
        }
        switched = !infos.empty() && infos.back().fft_size == 8192;
    }
    CHECK(switched);
    CHECK_EQ(h.engine.stats().fft_size, 8192);
    CHECK_NEAR(h.engine.stats().rbw_hz, window_enbw_bins(WindowType::BlackmanHarris) * fs / 8192, 1.0);
    if (!rows.empty()) {
        // 2 MHz with 8192 bins over 1024 columns: column = 512 + 2e6/fs*1024.
        const int peak = peak_column(rows.back());
        CHECK(std::abs(peak - (512 + static_cast<int>(std::lround(freq / fs * 1024)))) <= 1);
        CHECK_NEAR(rows.back()[static_cast<size_t>(peak)], -20.0, 1.0);  // scalloping of an off-bin tone
    }
}

TEST(fft_size_change_does_not_stall_the_stream)
{
    // Switching to a big FFT makes FFTW plan for a long time in the background.
    // The DSP thread must keep processing samples throughout.  In particular it
    // must not get stuck behind the planner when it lets go of its old plan,
    // which is what happens if it picks up the quickly-made new plan while the
    // planner is already busy measuring the tuned one.  To force exactly that
    // order, the stream is paused around the switch so that the DSP thread is
    // asleep when the quick plan arrives.
    const double fs = 20e6;
    Harness      h(fs, 1024, basic_params(1024));

    std::vector<int16_t> block(2 * 65'536);
    add_noise(block, 20.0, 4);

    std::atomic<bool> running{true};
    std::atomic<bool> feeding{true};
    std::thread       feeder([&] {
        while (running.load()) {
            if (feeding.load())
                h.feed(block);
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    std::thread drain([&] {
        std::vector<float> row(1024);
        while (running.load()) {
            while (h.engine.pop_rows(row.data(), 1) == 1) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Run for a moment, then let the DSP thread run dry.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    feeding.store(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    while (h.ring.readable() >= 1024)
        std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    h.engine.set_params(basic_params(SpectrumEngine::kMaxFftSize));
    // The change is noticed with the next frame; then give the planner time to
    // deliver the quick plan and get busy measuring the tuned one.
    h.feed(std::vector<int16_t>(block.begin(), block.begin() + 2 * 2048));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    // Resume the stream and watch the frame counter from this thread.
    feeding.store(true);
    const auto resumed       = std::chrono::steady_clock::now();
    auto       last_progress = resumed;
    uint64_t   last_frames   = h.engine.stats().frames;
    double     longest_stall = 0.0;
    bool       switched      = false;
    while (std::chrono::steady_clock::now() - resumed < std::chrono::milliseconds(3200)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        const auto     now    = std::chrono::steady_clock::now();
        const uint64_t frames = h.engine.stats().frames;
        if (frames != last_frames) {
            last_frames   = frames;
            last_progress = now;
        }
        longest_stall = std::max(longest_stall, std::chrono::duration<double>(now - last_progress).count());
        switched      = switched || h.engine.stats().fft_size == SpectrumEngine::kMaxFftSize;
    }
    running.store(false);
    feeder.join();
    drain.join();

    std::printf("    longest pause in processing while re-planning: %.0f ms\n", longest_stall * 1e3);
    CHECK(switched);
    CHECK(longest_stall < 0.2);
}

TEST(gap_rows_mark_discontinuities)
{
    const double fs = 20e6;
    Harness      h(fs, 1024, basic_params(1024));
    std::vector<int16_t> iq(2 * 500'000, 0);
    add_noise(iq, 5.0, 2);
    h.feed(iq);
    std::vector<RowInfo> infos;
    h.collect(2, &infos);

    h.engine.mark_gap();
    h.feed(iq);
    bool found = false;
    for (int i = 0; i < 20 && !found; ++i) {
        std::vector<RowInfo> more;
        const auto rows = h.collect(1, &more, 1.0);
        if (!rows.empty() && more[0].gap) {
            found = true;
            CHECK_NEAR(rows[0][0], SpectrumEngine::kFloorDb, 1e-6);
            CHECK_NEAR(rows[0][512], SpectrumEngine::kFloorDb, 1e-6);
        }
    }
    CHECK(found);
}

TEST(engine_stops_cleanly_while_idle)
{
    // No samples at all: the DSP thread is blocked waiting; stop() must not hang.
    IqRing         ring(size_t{1} << 20);
    SpectrumEngine engine(ring, 20e6, 1024);
    engine.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    engine.stop();
    CHECK(true);
}

TEST(dsp_throughput_exceeds_real_time)
{
    // The whole point: one thread must comfortably process 20 MS/s with the
    // default settings (4096-point FFT, 50 % overlap, every sample used).
    const double   fs = 20e6;
    SpectrumParams p;  // defaults
    Harness        h(fs, 2048, p);

    std::vector<int16_t> block(2 * 262'144);
    add_noise(block, 30.0, 9);

    // Warm up (also gives the planner time to deliver the tuned plan).
    for (int i = 0; i < 20; ++i)
        h.feed(block);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    const uint64_t frames_before = h.engine.stats().frames;
    const auto     t0            = std::chrono::steady_clock::now();
    const int      blocks        = 300;  // ~78.6 M samples = 3.9 s of signal at 20 MS/s
    for (int i = 0; i < blocks; ++i)
        h.feed(block);
    while (h.ring.readable() >= 4096)
        std::this_thread::yield();
    const double   elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const uint64_t frames  = h.engine.stats().frames - frames_before;
    const double   msps    = static_cast<double>(frames) * 2048.0 / elapsed / 1e6;

    std::printf("    DSP throughput: %.0f MS/s (%.1fx real time at 20 MS/s), %.0f FFTs/s\n", msps,
                msps / 20.0, static_cast<double>(frames) / elapsed);
    // Only meaningful for an optimised, uninstrumented build.
#if defined(NDEBUG) && !defined(__SANITIZE_THREAD__) && !defined(__SANITIZE_ADDRESS__)
    CHECK(msps > 30.0);
#endif
}
