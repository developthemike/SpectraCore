#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "app/pipeline.hpp"
#include "radio/acquisition.hpp"
#include "radio/file_source.hpp"
#include "radio/sim_source.hpp"
#include "test_framework.hpp"

using namespace spectro;

namespace {

// Writes raw bytes to a uniquely named temporary file, removed on destruction.
struct TempFile {
    std::filesystem::path path;

    TempFile(const std::string& extension, const void* data, size_t bytes)
    {
        static int counter = 0;
        path = std::filesystem::temp_directory_path() /
               ("spectro_test_" + std::to_string(static_cast<long long>(
                    std::chrono::steady_clock::now().time_since_epoch().count())) +
                "_" + std::to_string(counter++) + "." + extension);
        std::ofstream out(path, std::ios::binary);
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    }
    ~TempFile()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

SourceConfig file_config(const TempFile& file, const std::string& format, bool loop)
{
    SourceConfig c;
    c.kind        = "file";
    c.file_path   = file.path.string();
    c.file_format = format;
    c.file_loop   = loop;
    c.sample_rate = 500e6;  // play back as fast as possible
    return c;
}

// Produces a counting pattern, reports one stream break, then fails.
class FakeSource final : public IqSource {
public:
    explicit FakeSource(int fail_after) : fail_after_(fail_after) { info_.kind = "fake"; }

    void start() override {}
    void stop() override {}

    int read(int16_t* iq, int max_samples, ReadStatus& status) override
    {
        status = {};
        if (++reads_ > fail_after_)
            return -1;
        if (reads_ == 3) {
            status.discontinuity = true;
            status.lost_samples  = 1000;
        }
        if (reads_ == 5)
            return 0;  // a timeout must simply be retried
        const int n = std::min(max_samples, 256);
        for (int i = 0; i < n; ++i) {
            iq[2 * i]     = static_cast<int16_t>(counter_ & 0x7ff);
            iq[2 * i + 1] = static_cast<int16_t>((counter_ >> 11) & 0x7ff);
            ++counter_;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        return n;
    }

    const SourceInfo& info() const override { return info_; }
    double sample_rate() const override { return 1e6; }
    double frequency() const override { return 100e6; }
    double bandwidth() const override { return 1e6; }
    double gain() const override { return 0.0; }
    std::string last_error() const override { return "fake failure"; }
    int preferred_chunk() const override { return 256; }

private:
    SourceInfo info_;
    int        fail_after_;
    int        reads_   = 0;
    uint32_t   counter_ = 0;
};

}  // namespace

TEST(sim_source_is_paced_to_real_time)
{
    SourceConfig config;
    config.kind        = "sim";
    config.sample_rate = 2e6;
    SimSource source(config);
    source.start();

    std::vector<int16_t> buffer(2 * 16384);
    const auto           t0      = std::chrono::steady_clock::now();
    size_t               total   = 0;
    int                  largest = 0;
    while (total < 600'000) {  // 0.3 s of signal
        ReadStatus status;
        const int  n = source.read(buffer.data(), 16384, status);
        CHECK(n > 0);
        if (n <= 0)
            break;
        CHECK(!status.discontinuity);
        for (int i = 0; i < 2 * n; ++i)
            largest = std::max(largest, std::abs(static_cast<int>(buffer[static_cast<size_t>(i)])));
        total += static_cast<size_t>(n);
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    source.stop();

    CHECK(elapsed > 0.25 && elapsed < 0.6);
    CHECK(largest <= 2047);  // stays inside the Q11 range
    CHECK(largest > 60);     // and is not silent

    ReadStatus status;
    CHECK_EQ(source.read(buffer.data(), 16384, status), -1);  // stopped
}

TEST(sim_source_scene_follows_tuning)
{
    // End-to-end through the real pipeline: the documented carriers must show
    // up at the right frequency and level, and move when the "radio" retunes.
    Options options;
    options.source.kind        = "sim";
    options.source.sample_rate = 2e6;
    options.source.frequency   = 100e6;
    options.source.gain        = SimSource::kReferenceGainDb;
    options.columns            = 2048;
    options.spectrum.fft_size  = 2048;
    options.spectrum.window    = WindowType::FlatTop;  // amplitude-accurate between bins
    options.spectrum.rows_per_sec = 10.0;
    options.spectrum.dc_block  = false;

    Pipeline pipeline(options);
    pipeline.start();
    const double fs = pipeline.source().sample_rate();
    CHECK_NEAR(fs, 2e6, 1e-9);

    const auto carriers = SimSource::carriers(fs);
    CHECK(carriers.size() >= 2);

    auto next_data_row = [&](std::vector<float>& row) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            RowInfo info;
            if (pipeline.engine().pop_rows(row.data(), 1, &info) == 1) {
                if (!info.gap)
                    return true;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        return false;
    };
    // Level near a given offset from the current centre frequency.
    auto level_at = [&](const std::vector<float>& row, double offset_hz) {
        const double column = (offset_hz / fs + 0.5) * 2048.0;
        const int    c      = static_cast<int>(std::lround(column));
        float        best   = -400.0f;
        for (int k = std::max(0, c - 3); k <= std::min(2047, c + 3); ++k)
            best = std::max(best, row[static_cast<size_t>(k)]);
        return best;
    };

    std::vector<float> row(2048);
    CHECK(next_data_row(row));
    CHECK(next_data_row(row));
    for (const auto& carrier : carriers)
        CHECK_NEAR(level_at(row, carrier.offset_hz), carrier.level_dbfs, 0.7);
    // The strong carrier is the highest point of the averaged spectrum.
    const int peak = static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin());
    CHECK_NEAR(SpectrumEngine::column_frequency(peak, 2048, 2048, fs), carriers[0].offset_hz, 3.0 * fs / 2048);

    // Tune up by 0.1 fs: everything moves down by the same amount.
    const double step = 0.1 * fs;
    CHECK(pipeline.source().set_frequency(100e6 + step));
    pipeline.engine().mark_gap();
    CHECK(next_data_row(row));  // may straddle the retune
    CHECK(next_data_row(row));
    CHECK(next_data_row(row));
    CHECK_NEAR(level_at(row, carriers[0].offset_hz - step), carriers[0].level_dbfs, 0.7);
    CHECK_NEAR(level_at(row, carriers[1].offset_hz - step), carriers[1].level_dbfs, 0.7);
    CHECK(level_at(row, carriers[0].offset_hz) < carriers[0].level_dbfs - 15.0);

    // 10 dB more gain raises the carrier by 10 dB.
    CHECK(pipeline.source().set_gain(SimSource::kReferenceGainDb + 10.0));
    CHECK(next_data_row(row));
    CHECK(next_data_row(row));
    CHECK(next_data_row(row));
    CHECK_NEAR(level_at(row, carriers[0].offset_hz - step), carriers[0].level_dbfs + 10.0, 0.7);

    const AcquisitionStats stats = pipeline.acquisition().stats();
    CHECK_EQ(stats.lost_samples, uint64_t{0});
    CHECK_EQ(stats.ring_dropped, uint64_t{0});
    CHECK(!stats.failed);
    CHECK_NEAR(stats.measured_rate, fs, 0.05 * fs);
    pipeline.stop();
}

TEST(file_source_converts_every_format)
{
    ReadStatus           status;
    std::vector<int16_t> out(2 * 16);

    {   // sc16: already Q11, passed through untouched
        const int16_t data[] = {100, -200, 2047, -2048, 0, 1};
        TempFile      file("sc16", data, sizeof(data));
        FileSource    source(file_config(file, "auto", false));
        source.start();
        CHECK_EQ(source.read(out.data(), 16, status), 3);
        CHECK_EQ(out[0], int16_t{100});
        CHECK_EQ(out[3], int16_t{-2048});
        CHECK_EQ(out[5], int16_t{1});
    }
    {   // cs16: full-scale 16 bit -> Q11
        const int16_t data[] = {32767, -32768, 1600, -1600};
        TempFile      file("cs16", data, sizeof(data));
        FileSource    source(file_config(file, "auto", false));
        source.start();
        CHECK_EQ(source.read(out.data(), 16, status), 2);
        CHECK_EQ(out[0], int16_t{2047});
        CHECK_EQ(out[1], int16_t{-2048});
        CHECK_EQ(out[2], int16_t{100});
        CHECK_EQ(out[3], int16_t{-100});
    }
    {   // cf32: +/-1.0 full scale, clipped
        const float data[] = {0.5f, -0.25f, 1.5f, -1.5f};
        TempFile    file("cf32", data, sizeof(data));
        FileSource  source(file_config(file, "auto", false));
        source.start();
        CHECK_EQ(source.read(out.data(), 16, status), 2);
        CHECK_EQ(out[0], int16_t{1024});
        CHECK_EQ(out[1], int16_t{-512});
        CHECK_EQ(out[2], int16_t{2047});
        CHECK_EQ(out[3], int16_t{-2047});
    }
    {   // cs8
        const int8_t data[] = {127, -128, 10, -10};
        TempFile     file("cs8", data, sizeof(data));
        FileSource   source(file_config(file, "auto", false));
        source.start();
        CHECK_EQ(source.read(out.data(), 16, status), 2);
        CHECK_EQ(out[0], int16_t{2032});
        CHECK_EQ(out[1], int16_t{-2048});
        CHECK_EQ(out[2], int16_t{160});
    }
    {   // cu8 (rtl_sdr): 127.5 is zero
        const uint8_t data[] = {255, 0, 128, 127};
        TempFile      file("cu8", data, sizeof(data));
        FileSource    source(file_config(file, "auto", false));
        source.start();
        CHECK_EQ(source.read(out.data(), 16, status), 2);
        CHECK_EQ(out[0], int16_t{2040});
        CHECK_EQ(out[1], int16_t{-2040});
        CHECK_EQ(out[2], int16_t{8});
        CHECK_EQ(out[3], int16_t{-8});
    }
}

TEST(file_source_loops_or_ends)
{
    const int16_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};  // four samples
    TempFile      file("sc16", data, sizeof(data));
    ReadStatus    status;
    std::vector<int16_t> out(2 * 16);

    {
        FileSource source(file_config(file, "sc16", true));
        source.start();
        CHECK_EQ(source.read(out.data(), 3, status), 3);
        CHECK(!status.discontinuity);
        CHECK_EQ(source.read(out.data(), 3, status), 1);  // the rest of the file
        CHECK_EQ(out[0], int16_t{7});
        CHECK_EQ(source.read(out.data(), 3, status), 3);  // wrapped around
        CHECK(status.discontinuity);
        CHECK_EQ(out[0], int16_t{1});
    }
    {
        FileSource source(file_config(file, "sc16", false));
        source.start();
        CHECK_EQ(source.read(out.data(), 16, status), 4);
        CHECK_EQ(source.read(out.data(), 16, status), -1);
        CHECK(source.last_error() == "End of file");
    }
}

TEST(file_source_rejects_bad_input)
{
    SourceConfig missing;
    missing.kind      = "file";
    missing.file_path = "/nonexistent/definitely/not/here.sc16";
    bool threw = false;
    try {
        FileSource source(missing);
        source.start();
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    const char one_byte = 0;
    TempFile   tiny("sc16", &one_byte, 1);
    threw = false;
    try {
        FileSource source(file_config(tiny, "sc16", true));
        source.start();
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    FileSource::Format format;
    CHECK(FileSource::resolve_format("auto", "capture.cfile", format) && format == FileSource::Format::CF32);
    CHECK(FileSource::resolve_format("auto", "capture.bin", format) && format == FileSource::Format::SC16Q11);
    CHECK(FileSource::resolve_format("CU8", "whatever.bin", format) && format == FileSource::Format::CU8);
    CHECK(!FileSource::resolve_format("wav", "x", format));

    threw = false;
    try {
        SourceConfig unknown;
        unknown.kind = "hackrf";
        auto source  = make_source(unknown);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(acquisition_tracks_stream_health)
{
    FakeSource  source(200);  // fails on read 201
    IqRing      ring(size_t{1} << 16);
    Acquisition acquisition(source, ring);
    acquisition.start();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!acquisition.stats().failed && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));

    const AcquisitionStats stats = acquisition.stats();
    CHECK(stats.failed);
    CHECK(!stats.running);
    CHECK(stats.error == "fake failure");
    CHECK_EQ(stats.discontinuities, uint64_t{1});
    CHECK_EQ(stats.lost_samples, uint64_t{1000});
    CHECK_EQ(stats.samples, uint64_t{199 * 256});  // 200 reads, one of them a timeout
    CHECK_EQ(stats.ring_dropped, uint64_t{0});
    CHECK_EQ(ring.readable(), size_t{199 * 256});

    // The samples arrived in order.
    const auto spans = ring.peek(199 * 256);
    bool       ok    = true;
    for (uint32_t i = 0; i < 199 * 256; ++i) {
        const int16_t* s = i < spans.first_count ? spans.first + 2 * i
                                                 : spans.second + 2 * (i - spans.first_count);
        ok = ok && s[0] == static_cast<int16_t>(i & 0x7ff) && s[1] == static_cast<int16_t>((i >> 11) & 0x7ff);
    }
    CHECK(ok);
    acquisition.stop();
}

TEST(acquisition_never_blocks_on_a_full_ring)
{
    // Nobody consumes: once the ring is full the reader keeps reading (so the
    // device never overflows) and counts what it had to throw away.
    FakeSource  source(1 << 30);
    IqRing      ring(1024);
    Acquisition acquisition(source, ring);
    acquisition.start();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (acquisition.stats().ring_dropped < 2000 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));

    const auto t0 = std::chrono::steady_clock::now();
    acquisition.stop();
    const double stop_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const AcquisitionStats stats = acquisition.stats();
    CHECK_EQ(stats.samples, uint64_t{1024});
    CHECK(stats.ring_dropped >= 2000);
    CHECK(!stats.failed);
    CHECK(stop_time < 1.0);
}

TEST(pipeline_restarts_cleanly)
{
    Options options;
    options.source.kind        = "sim";
    options.source.sample_rate = 1e6;
    options.columns            = 1024;
    options.spectrum.fft_size  = 1024;

    for (int round = 0; round < 3; ++round) {
        Pipeline pipeline(options);
        pipeline.start();
        CHECK(pipeline.started());
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        pipeline.stop();
        CHECK(!pipeline.started());
        CHECK(!pipeline.acquisition().stats().failed);
    }
}
