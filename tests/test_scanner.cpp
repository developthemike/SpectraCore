#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

#include "app/pipeline.hpp"
#include "audio/scanner.hpp"
#include "radio/sim_source.hpp"
#include "test_framework.hpp"

using namespace spectro;

namespace {

// 2048 columns over 2.4 MHz around 100 MHz: 1.17 kHz per column.
SpectrumView test_view()
{
    return {100e6, 2.4e6, 2048, 4096};
}

float power_of(float dbfs) { return std::pow(10.0f, dbfs / 10.0f); }

// A spectrum row in dBFS: a noise floor with a little ripple, plus signals.
struct Row {
    std::vector<float> db;
    Row() : db(2048)
    {
        for (size_t c = 0; c < db.size(); ++c)
            db[c] = -90.0f + 1.5f * std::sin(0.37f * static_cast<float>(c));
    }
    // A narrow signal: a peak with steep sides.
    Row& narrow(int column, float peak)
    {
        const float shape[] = {-20.0f, -6.0f, 0.0f, -6.0f, -20.0f};
        for (int k = 0; k < 5; ++k)
            db[static_cast<size_t>(column - 2 + k)] = std::max(db[static_cast<size_t>(column - 2 + k)], peak + shape[k]);
        return *this;
    }
    // A flat-topped signal `width` columns wide.
    Row& flat(int first, int width, float level)
    {
        for (int c = first; c < first + width; ++c)
            db[static_cast<size_t>(c)] = std::max(db[static_cast<size_t>(c)], level);
        return *this;
    }
    std::vector<float> power() const
    {
        std::vector<float> p(db.size());
        for (size_t c = 0; c < db.size(); ++c)
            p[c] = power_of(db[c]);
        return p;
    }
};

// Stands in for the Listener: settles on whatever it is told at once.
class FakeReceiver final : public ScanReceiver {
public:
    struct Tune {
        double frequency, bandwidth, search;
    };
    std::vector<Tune> tunes;
    bool              muted = false;

    void   scan_tune(double frequency_hz, double bandwidth_hz, double search_hz) override
    {
        tunes.push_back({frequency_hz, bandwidth_hz, search_hz});
    }
    void   scan_mute(bool m) override { muted = m; }
    double scan_frequency() const override
    {
        return tunes.empty() ? std::numeric_limits<double>::quiet_NaN() : tunes.back().frequency;
    }
};

// Feeds `seconds` of identical rows, 50 per second.
void feed(Scanner& scanner, const Row& row, double seconds)
{
    RowInfo info;
    info.period_s = 0.02;
    info.fft_size = 4096;
    for (double t = 0.0; t < seconds - 1e-9; t += info.period_s)
        scanner.on_row(row.db.data(), info, test_view());
}

}  // namespace

TEST(scanner_finds_and_classifies_signals)
{
    const SpectrumView view = test_view();
    const double       col  = view.column_hz();
    Row row;
    row.narrow(300, -40.0f);                       // a two-way radio channel
    row.flat(1000, 150, -55.0f);                   // 176 kHz: a broadcast station
    row.db[1060] = -95.0f;                         // ... with a dip that does not split it
    row.flat(1400, 430, -62.0f);                   // 500 kHz: no FM channel is that wide
    const std::vector<float> power = row.power();
    const float              floor = spectrum_noise_floor(power.data(), view.columns);
    CHECK_NEAR(10.0 * std::log10(floor), -90.0, 1.6);

    std::vector<DetectedSignal> found = find_signals(power.data(), view, floor);
    CHECK_EQ(found.size(), size_t{3});
    if (found.size() != 3)
        return;
    std::sort(found.begin(), found.end(),
              [](const DetectedSignal& a, const DetectedSignal& b) { return a.frequency_hz < b.frequency_hz; });

    const DetectedSignal& voice = found[0];
    CHECK_NEAR(voice.frequency_hz, view.frequency(300), 0.2 * col);
    CHECK_NEAR(voice.peak_dbfs, -40.0, 0.01);
    CHECK(voice.occupied_hz <= 5.0 * col);
    CHECK(!voice.wide);
    CHECK_NEAR(listen_bandwidth_for(voice, col), 12.5e3, 1e-9);

    const DetectedSignal& station = found[1];
    CHECK_NEAR(station.frequency_hz, view.frequency(1074.5), 2.0 * col);
    CHECK_NEAR(station.occupied_hz, 150.0 * col, 2.0 * col);
    CHECK(station.wide);
    CHECK_NEAR(listen_bandwidth_for(station, col), 200e3, 1e-9);

    CHECK(found[2].occupied_hz > kFmMaxOccupied);

    // Narrowband channel widths follow the measured width.
    DetectedSignal s;
    s.occupied_hz = 15e3 + col;
    CHECK_NEAR(listen_bandwidth_for(s, col), 16e3, 1e-9);
    s.occupied_hz = 24e3;
    CHECK_NEAR(listen_bandwidth_for(s, col), 25e3, 1e-9);
}

TEST(scanner_follows_transmissions)
{
    const SpectrumView view = test_view();
    FakeReceiver       rx;
    Scanner            scanner(rx);
    scanner.set_threshold(-60.0f);
    scanner.set_hang(0.5);

    Row noise;
    Row a = Row().narrow(500, -40.0f);
    Row ab = Row().narrow(500, -40.0f).narrow(1500, -45.0f);
    Row b = Row().narrow(1500, -45.0f);
    Row weak = Row().narrow(500, -70.0f);  // below the threshold

    scanner.start();
    CHECK(rx.muted);
    feed(scanner, noise, 0.5);
    feed(scanner, weak, 0.5);
    CHECK(scanner.status().state == Scanner::State::Searching);
    CHECK(rx.tunes.empty());

    // Two transmissions: it takes the lower one, and stays with it.
    feed(scanner, ab, 1.0);
    CHECK(scanner.status().state == Scanner::State::Receiving);
    CHECK_EQ(rx.tunes.size(), size_t{1});
    CHECK_NEAR(rx.tunes[0].frequency, view.frequency(500), 0.5 * view.column_hz());
    CHECK_NEAR(rx.tunes[0].bandwidth, 12.5e3, 1e-9);
    CHECK(rx.tunes[0].search > 0.0);
    CHECK(!rx.muted);
    CHECK_EQ(scanner.candidates().size(), size_t{2});

    // A pause shorter than the hang time: muted at once, but it waits.
    feed(scanner, b, 0.12);
    CHECK(scanner.status().state == Scanner::State::Holding);
    CHECK(rx.muted);
    feed(scanner, b, 0.18);
    CHECK(scanner.status().state == Scanner::State::Holding);
    feed(scanner, ab, 0.3);
    CHECK(scanner.status().state == Scanner::State::Receiving);
    CHECK(!rx.muted);
    CHECK_EQ(rx.tunes.size(), size_t{1});

    // The end of the transmission: on to the next signal up.
    feed(scanner, b, 1.0);
    CHECK(scanner.status().state == Scanner::State::Receiving);
    CHECK_EQ(rx.tunes.size(), size_t{2});
    CHECK_NEAR(rx.tunes.back().frequency, view.frequency(1500), 0.5 * view.column_hz());

    // When that one ends too, it wraps around to the bottom of the band.
    feed(scanner, a, 1.5);
    CHECK_EQ(rx.tunes.size(), size_t{3});
    CHECK_NEAR(rx.tunes.back().frequency, view.frequency(500), 0.5 * view.column_hz());

    // Skipping leaves a signal out from then on.
    scanner.skip();
    CHECK_EQ(scanner.lockouts(), size_t{1});
    feed(scanner, a, 1.0);
    CHECK(scanner.status().state == Scanner::State::Searching);
    CHECK_EQ(rx.tunes.size(), size_t{3});
    CHECK(scanner.candidates().empty());
    scanner.clear_lockouts();
    feed(scanner, a, 0.5);
    CHECK_EQ(rx.tunes.size(), size_t{4});

    // Retuning the radio (a gap row) starts the search afresh.
    RowInfo gap;
    gap.gap = true;
    scanner.on_row(noise.db.data(), gap, view);
    CHECK(scanner.status().state == Scanner::State::Searching);
    CHECK(rx.muted);
    feed(scanner, a, 0.5);
    CHECK_EQ(rx.tunes.size(), size_t{5});

    // Wideband signals get a wideband channel.
    Scanner wide_scanner(rx);
    wide_scanner.set_threshold(-60.0f);
    wide_scanner.start();
    feed(wide_scanner, Row().flat(1000, 150, -55.0f), 0.5);
    CHECK(wide_scanner.status().state == Scanner::State::Receiving);
    CHECK(wide_scanner.status().wide);
    CHECK_NEAR(rx.tunes.back().bandwidth, 200e3, 1e-9);

    // Stopping leaves the receiver where it is, audible.
    scanner.stop();
    CHECK(!scanner.active());
    CHECK(!rx.muted || wide_scanner.active());
    wide_scanner.stop();
    CHECK(!rx.muted);
}

TEST(scanner_tours_the_simulator)
{
    // The real thing: simulator -> spectrum engine -> scanner -> listener, as
    // in the application.  At 2.4 MS/s from the bottom of the band up, the
    // simulator has a steady carrier, its narrowband voice channel and, among
    // others, its broadcast station.
    Options options;
    options.source.kind        = "sim";
    options.source.sample_rate = 2.4e6;
    options.source.frequency   = 98e6;
    options.source.gain        = SimSource::kReferenceGainDb;
    Pipeline pipeline(options);
    pipeline.start();
    Listener& listener = pipeline.listener();
    listener.set_volume(1.0f);
    listener.set_enabled(true);

    Scanner scanner(listener);
    scanner.set_threshold(-55.0f);
    scanner.start();

    const int          columns = pipeline.columns();
    std::vector<float> row(static_cast<size_t>(columns));
    // Runs the scanner until it is on a signal and the listener has found the
    // carrier, as the application would every frame.
    const auto wait_for_signal = [&]() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            RowInfo info;
            if (pipeline.engine().pop_rows(row.data(), 1, &info) == 1) {
                scanner.on_row(row.data(), info, {98e6, 2.4e6, columns, info.fft_size});
                if (scanner.status().state == Scanner::State::Receiving && !listener.status().searching)
                    return true;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        return false;
    };

    const SimSource::VoiceChannel     voice   = SimSource::voice_channel(2.4e6);
    const SimSource::BroadcastStation station = SimSource::broadcast_station(2.4e6);
    const double carrier_hz = 98e6 + SimSource::carriers(2.4e6)[0].offset_hz;
    const double voice_hz   = 98e6 + voice.offset_hz;
    const double station_hz = 98e6 + station.offset_hz;

    CHECK(wait_for_signal());
    CHECK_NEAR(listener.status().frequency, carrier_hz, 1000.0);
    CHECK(!scanner.status().wide);
    scanner.skip();

    CHECK(wait_for_signal());
    CHECK_NEAR(listener.status().frequency, voice_hz, 1000.0);
    CHECK(!scanner.status().wide);
    CHECK(listener.status().mode == FmMode::Narrow);

    // Further up, eventually the broadcast station, recognised as wideband.
    bool reached = false;
    for (int hop = 0; hop < 12 && !reached; ++hop) {
        scanner.skip();
        if (!wait_for_signal())
            break;
        const double at = listener.status().frequency;
        if (std::fabs(at - station_hz) < 50e3) {
            reached = true;
            CHECK(scanner.status().wide);
            CHECK(listener.status().mode == FmMode::Wide);
            CHECK_NEAR(at, station_hz, 3000.0);
        } else if (std::fabs(at - voice_hz) < 5e3 || std::fabs(at - carrier_hz) < 5e3) {
            CHECK(false);  // skipped signals must stay out
        }
    }
    CHECK(reached);
    if (reached) {
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        const double loudness = station.deviation_hz / FmDemodulator::kWideDeviation / std::sqrt(2.0);
        CHECK_NEAR(listener.status().audio_rms, loudness, 0.08);
    }

    scanner.stop();
    listener.set_enabled(false);
    const AcquisitionStats stats = pipeline.acquisition().stats();
    CHECK_EQ(stats.lost_samples, uint64_t{0});
    pipeline.stop();
}
