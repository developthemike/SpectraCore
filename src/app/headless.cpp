#include "app/headless.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "app/pipeline.hpp"
#include "app/signals.hpp"
#include "util/log.hpp"
#include "util/units.hpp"

namespace spectro {

int run_headless(const Options& options)
{
    if (!std::isnan(options.listen_frequency))
        LOG_WARN("--listen plays audio only in the window, not in --headless mode");

    Pipeline pipeline(options);
    try {
        pipeline.start();
    } catch (const std::exception& e) {
        LOG_ERROR("%s", e.what());
        return 1;
    }

    IqSource&       source = pipeline.source();
    SpectrumEngine& engine = pipeline.engine();
    const int       width  = pipeline.columns();
    const double    rate   = source.sample_rate();

    std::printf("Source : %s\n", source.info().description.c_str());
    std::printf("Tuning : %s, %s, bandwidth %s\n", format_frequency(source.frequency()).c_str(),
                format_si(rate, "S/s").c_str(), format_si(source.bandwidth(), "Hz").c_str());
    std::fflush(stdout);

    constexpr int        kBatch = 64;
    std::vector<float>   rows(static_cast<size_t>(kBatch) * static_cast<size_t>(width));
    std::vector<RowInfo> infos(kBatch);
    std::vector<float>   latest(static_cast<size_t>(width), SpectrumEngine::kFloorDb);
    std::vector<float>   sorted(static_cast<size_t>(width));
    std::vector<double>  mean_power(static_cast<size_t>(width), 0.0);
    uint64_t             mean_rows = 0;
    int                  fft_size  = options.spectrum.fft_size;
    bool                 have_row  = false;

    using clock      = std::chrono::steady_clock;
    const auto start = clock::now();
    auto       next_report = start + std::chrono::seconds(1);
    uint64_t   rows_at_last_report = 0;
    auto       last_report = start;
    bool       failed = false;
    // Reference point for the average rate: taken at the first report, so that
    // start-up (and the samples buffered during it) does not skew the figure.
    bool       have_reference = false;
    auto       reference_time = start;
    uint64_t   reference_samples = 0;

    while (!quit_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        for (;;) {
            const int n = engine.pop_rows(rows.data(), kBatch, infos.data());
            if (n == 0)
                break;
            for (int i = 0; i < n; ++i) {
                if (infos[static_cast<size_t>(i)].gap)
                    continue;
                const float* row = rows.data() + static_cast<size_t>(i) * static_cast<size_t>(width);
                for (int c = 0; c < width; ++c)
                    mean_power[static_cast<size_t>(c)] += std::pow(10.0, static_cast<double>(row[c]) / 10.0);
                ++mean_rows;
                std::copy_n(row, width, latest.begin());
                fft_size = infos[static_cast<size_t>(i)].fft_size;
                have_row = true;
            }
        }

        const auto             now     = clock::now();
        const double           elapsed = std::chrono::duration<double>(now - start).count();
        const AcquisitionStats acq     = pipeline.acquisition().stats();

        if (now >= next_report) {
            const EngineStats es       = engine.stats();
            const double      interval = std::chrono::duration<double>(now - last_report).count();
            const double      row_rate = static_cast<double>(es.rows_emitted - rows_at_last_report) / interval;
            rows_at_last_report        = es.rows_emitted;
            last_report                = now;
            next_report += std::chrono::seconds(1);
            if (!have_reference) {
                have_reference    = true;
                reference_time    = now;
                reference_samples = acq.samples + acq.ring_dropped;
            }

            char peak_text[96] = "no data yet";
            if (have_row) {
                const auto   peak_it  = std::max_element(latest.begin(), latest.end());
                const int    peak_col = static_cast<int>(peak_it - latest.begin());
                const double peak_hz  = source.frequency() +
                                        SpectrumEngine::column_frequency(peak_col, width, fft_size, rate);
                sorted = latest;
                std::nth_element(sorted.begin(), sorted.begin() + width / 2, sorted.end());
                std::snprintf(peak_text, sizeof(peak_text), "peak %.4f MHz %6.1f dBFS  floor %6.1f dBFS",
                              peak_hz / 1e6, static_cast<double>(*peak_it),
                              static_cast<double>(sorted[static_cast<size_t>(width / 2)]));
            }
            std::printf("[%6.1f s] %7.3f MS/s  lost %llu  discarded %llu  ring %4.1f%%  dsp %3.0f%%  "
                        "%5.1f rows/s  %s\n",
                        elapsed, acq.measured_rate / 1e6,
                        static_cast<unsigned long long>(acq.lost_samples),
                        static_cast<unsigned long long>(acq.ring_dropped), acq.ring_fill * 100.0,
                        static_cast<double>(es.cpu_load) * 100.0, row_rate, peak_text);
            std::fflush(stdout);
        }

        if (acq.failed) {
            failed = true;
            break;
        }
        if (options.duration > 0.0 && elapsed >= options.duration)
            break;
    }

    const auto             end     = clock::now();
    const double           elapsed = std::chrono::duration<double>(end - start).count();
    const AcquisitionStats acq     = pipeline.acquisition().stats();
    const EngineStats      es      = engine.stats();
    pipeline.stop();

    const uint64_t received = acq.samples + acq.ring_dropped;
    const double   span     = std::chrono::duration<double>(end - reference_time).count();
    std::printf("\nReceived %.2f M samples in %.2f s", static_cast<double>(received) / 1e6, elapsed);
    if (have_reference && span >= 1.0)
        std::printf(" (%.3f MS/s average)", static_cast<double>(received - reference_samples) / span / 1e6);
    std::printf("\n");
    std::printf("Stream breaks: %llu   lost by the device/USB: %llu samples   discarded by the host: %llu samples\n",
                static_cast<unsigned long long>(acq.discontinuities),
                static_cast<unsigned long long>(acq.lost_samples),
                static_cast<unsigned long long>(acq.ring_dropped));
    std::printf("FFTs: %llu (%d-point, RBW %s)   rows: %llu produced, %llu dropped\n",
                static_cast<unsigned long long>(es.frames), es.fft_size, format_si(es.rbw_hz, "Hz").c_str(),
                static_cast<unsigned long long>(es.rows_emitted),
                static_cast<unsigned long long>(es.rows_dropped));

    if (!options.spectrum_csv.empty()) {
        if (mean_rows == 0) {
            LOG_ERROR("No spectrum was produced; not writing %s", options.spectrum_csv.c_str());
        } else if (std::FILE* f = std::fopen(options.spectrum_csv.c_str(), "w")) {
            std::fprintf(f, "frequency_hz,power_dbfs\n");
            for (int c = 0; c < width; ++c) {
                const double hz = source.frequency() + SpectrumEngine::column_frequency(c, width, fft_size, rate);
                const double db = 10.0 * std::log10(mean_power[static_cast<size_t>(c)] /
                                                    static_cast<double>(mean_rows) + 1e-30);
                std::fprintf(f, "%.1f,%.3f\n", hz, db);
            }
            std::fclose(f);
            std::printf("Wrote the time-averaged spectrum (%llu rows) to %s\n",
                        static_cast<unsigned long long>(mean_rows), options.spectrum_csv.c_str());
        } else {
            LOG_ERROR("Could not write %s", options.spectrum_csv.c_str());
        }
    }

    if (failed) {
        LOG_ERROR("The stream ended with an error: %s", acq.error.c_str());
        return 1;
    }
    return 0;
}

}  // namespace spectro
