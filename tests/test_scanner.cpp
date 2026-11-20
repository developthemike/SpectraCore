#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <thread>
#include <vector>

#include "app/pipeline.hpp"
#include "audio/listener.hpp"
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
    ScanContent       content;  // what it reports the channel carries

    void   scan_tune(double frequency_hz, double bandwidth_hz, double search_hz) override
    {
        tunes.push_back({frequency_hz, bandwidth_hz, search_hz});
    }
    void   scan_mute(bool m) override { muted = m; }
    double scan_frequency() const override
    {
        return tunes.empty() ? std::numeric_limits<double>::quiet_NaN() : tunes.back().frequency;
    }
    ScanContent scan_content() const override { return content; }
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

    // Skipping moves on at once, and leaves the signal out while the rest of
    // the list is gone through: here, the steady one at the bottom.
    feed(scanner, ab, 0.3);
    CHECK(!scanner.skip());
    CHECK_EQ(scanner.lockouts(), size_t{1});
    feed(scanner, ab, 0.3);
    CHECK_EQ(rx.tunes.size(), size_t{4});
    CHECK_NEAR(rx.tunes.back().frequency, view.frequency(1500), 0.5 * view.column_hz());
    // When the other one ends, the skipped one is not taken up again ...
    feed(scanner, a, 1.0);
    CHECK(scanner.status().state == Scanner::State::Searching);
    CHECK_EQ(rx.tunes.size(), size_t{4});
    CHECK(scanner.candidates().empty());
    // ... but skipping the last of the list goes round again from the first.
    feed(scanner, ab, 0.3);
    CHECK_EQ(rx.tunes.size(), size_t{5});
    CHECK(scanner.skip());
    CHECK_EQ(scanner.lockouts(), size_t{0});
    feed(scanner, ab, 0.3);
    CHECK_EQ(rx.tunes.size(), size_t{6});
    CHECK_NEAR(rx.tunes.back().frequency, view.frequency(500), 0.5 * view.column_hz());
    // A list of one: skipping comes straight back to it.
    feed(scanner, a, 0.3);
    CHECK(scanner.skip());
    feed(scanner, a, 0.3);
    CHECK_EQ(rx.tunes.size(), size_t{7});
    CHECK_NEAR(rx.tunes.back().frequency, view.frequency(500), 0.5 * view.column_hz());
    // Clearing brings the skipped signals back at once.
    feed(scanner, ab, 0.3);
    scanner.skip();
    CHECK_EQ(scanner.lockouts(), size_t{1});
    scanner.clear_lockouts();
    CHECK_EQ(scanner.lockouts(), size_t{0});

    // Retuning the radio (a gap row) starts the search afresh.
    RowInfo gap;
    gap.gap = true;
    scanner.on_row(noise.db.data(), gap, view);
    CHECK(scanner.status().state == Scanner::State::Searching);
    CHECK(rx.muted);
    feed(scanner, a, 0.5);
    CHECK_EQ(rx.tunes.size(), size_t{8});

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

    // Skipping on past the top of the list goes round to the first again.
    bool wrapped = false;
    for (int hop = 0; hop < 12 && reached && !wrapped; ++hop) {
        const bool round = scanner.skip();
        if (!wait_for_signal())
            break;
        if (round) {
            wrapped = true;
            CHECK_NEAR(listener.status().frequency, carrier_hz, 1000.0);
            CHECK_EQ(scanner.lockouts(), size_t{0});
        }
    }
    CHECK(wrapped);

    scanner.stop();
    listener.set_enabled(false);
    const AcquisitionStats stats = pipeline.acquisition().stats();
    CHECK_EQ(stats.lost_samples, uint64_t{0});
    pipeline.stop();
}

TEST(scanner_waits_only_for_dmr_voice)
{
    const SpectrumView view = test_view();
    FakeReceiver       rx;
    Scanner            scanner(rx);
    scanner.set_threshold(-60.0f);
    scanner.set_dmr_only(true);
    scanner.set_dmr_hold(2.0);
    const Row rows = Row().narrow(500, -40.0f).narrow(1500, -45.0f).flat(1700, 150, -55.0f);

    // Not DMR: it moves on after about a second, and the broadcast-wide
    // signal is not even a candidate.
    scanner.start();
    feed(scanner, rows, 0.3);
    CHECK_EQ(rx.tunes.size(), size_t{1});
    CHECK_NEAR(rx.tunes.back().frequency, view.frequency(500), 0.5 * view.column_hz());
    CHECK_EQ(scanner.candidates().size(), size_t{2});
    CHECK(!scanner.status().dmr);
    CHECK(scanner.status().wait_left > 0.0 && scanner.status().wait_left <= Scanner::kRecogniseSeconds);
    feed(scanner, rows, 1.2);
    CHECK_EQ(rx.tunes.size(), size_t{2});
    CHECK_NEAR(rx.tunes.back().frequency, view.frequency(1500), 0.5 * view.column_hz());

    // DMR without a call: it waits for the hold time.
    rx.content.dmr = true;
    feed(scanner, rows, 1.0);
    CHECK_EQ(rx.tunes.size(), size_t{2});
    CHECK(scanner.status().dmr && !scanner.status().voice);
    CHECK(scanner.status().wait_left > 0.0);
    feed(scanner, rows, 1.5);
    CHECK_EQ(rx.tunes.size(), size_t{3});
    CHECK_NEAR(rx.tunes.back().frequency, view.frequency(500), 0.5 * view.column_hz());

    // A call: it stays as long as it lasts ...
    rx.content.dmr_voice = true;
    feed(scanner, rows, 6.0);
    CHECK_EQ(rx.tunes.size(), size_t{3});
    CHECK(scanner.status().voice);
    CHECK(!rx.muted);
    // ... and waits the hold time for the next over before moving on.
    rx.content.dmr_voice = false;
    feed(scanner, rows, 1.5);
    CHECK_EQ(rx.tunes.size(), size_t{3});
    rx.content.dmr_voice = true;  // the next over
    feed(scanner, rows, 1.0);
    rx.content.dmr_voice = false;
    feed(scanner, rows, 1.5);
    CHECK_EQ(rx.tunes.size(), size_t{3});
    feed(scanner, rows, 1.0);
    CHECK_EQ(rx.tunes.size(), size_t{4});
    CHECK_NEAR(rx.tunes.back().frequency, view.frequency(1500), 0.5 * view.column_hz());

    // Without DMR only it stays put, whatever the channel carries.
    scanner.set_dmr_only(false);
    rx.content = {};
    feed(scanner, rows, 3.0);
    CHECK_EQ(rx.tunes.size(), size_t{4});
}

TEST(scanner_stays_for_a_whole_dmr_call)
{
    // A call holds the scanner until it ends: past the hold time, and through
    // a fade below the threshold far longer than the hang time - with DMR
    // only or without.
    const SpectrumView view = test_view();
    const Row          both = Row().narrow(500, -40.0f).narrow(1500, -45.0f);
    const Row          fade = Row().narrow(500, -70.0f).narrow(1500, -45.0f);  // the call's signal below the threshold
    for (const bool dmr_only : {true, false}) {
        FakeReceiver rx;
        Scanner      scanner(rx);
        scanner.set_threshold(-60.0f);
        scanner.set_hang(0.5);
        scanner.set_dmr_only(dmr_only);
        scanner.set_dmr_hold(2.0);
        rx.content = {true, true};  // DMR, a call on

        scanner.start();
        feed(scanner, both, 0.3);
        CHECK_EQ(rx.tunes.size(), size_t{1});
        feed(scanner, both, 4.0);  // longer than the hold time
        feed(scanner, fade, 3.0);  // and a fade far longer than the hang time
        CHECK_EQ(rx.tunes.size(), size_t{1});
        CHECK(scanner.status().state == Scanner::State::Receiving);
        CHECK(scanner.status().voice);
        CHECK(!rx.muted);

        // The call ends: the usual rules again, here the hang time of a signal gone.
        rx.content.dmr_voice = false;
        feed(scanner, fade, 1.0);
        CHECK_EQ(rx.tunes.size(), size_t{2});
        CHECK_NEAR(rx.tunes.back().frequency, view.frequency(1500), 0.5 * view.column_hz());
    }
}

TEST(scanner_waits_for_the_systems_asked_for)
{
    // P25 only: DMR is passed by like anything else; P25 is waited on, and a
    // P25 call holds the scan.  With both, either.
    const Row          rows = Row().narrow(500, -40.0f).narrow(1500, -45.0f);
    FakeReceiver       rx;
    Scanner            scanner(rx);
    scanner.set_threshold(-60.0f);
    scanner.set_p25_only(true);
    scanner.set_dmr_hold(2.0);
    rx.content = {true, true, false, false};  // a DMR call: not asked for
    scanner.start();
    feed(scanner, rows, 0.3);
    CHECK_EQ(rx.tunes.size(), size_t{1});
    CHECK(!scanner.status().voice && !scanner.status().wanted);
    feed(scanner, rows, 1.2);
    CHECK_EQ(rx.tunes.size(), size_t{2});  // moved on after about a second

    rx.content = {false, false, true, false};  // P25, no call: the hold time
    feed(scanner, rows, 1.5);
    CHECK_EQ(rx.tunes.size(), size_t{2});
    CHECK(scanner.status().p25 && scanner.status().wanted);
    rx.content.p25_voice = true;  // a call
    feed(scanner, rows, 5.0);
    CHECK_EQ(rx.tunes.size(), size_t{2});
    CHECK(scanner.status().voice);
    rx.content.p25_voice = false;
    feed(scanner, rows, 2.5);
    CHECK_EQ(rx.tunes.size(), size_t{3});

    // Both asked for: the DMR call holds it too.
    scanner.set_dmr_only(true);
    rx.content = {true, true, false, false};
    feed(scanner, rows, 4.0);
    CHECK_EQ(rx.tunes.size(), size_t{3});
    CHECK(scanner.status().voice);
}

TEST(scanner_finds_dmr_calls_in_the_simulator)
{
    // Scanning the simulator with DMR only: the carriers, the analog voice
    // channel and the pager are passed by within seconds; the DMR repeater is
    // kept while its first call lasts (a second to about 4.7 s in), then left
    // once the hold time is up.
    Options options;
    options.source.kind        = "sim";
    options.source.sample_rate = 2.4e6;
    options.source.frequency   = 98e6;
    options.source.gain        = SimSource::kReferenceGainDb;
    Pipeline pipeline(options);
    pipeline.start();
    Listener& listener = pipeline.listener();
    listener.set_volume(0.0f);
    listener.set_dmr_only(true);
    listener.set_enabled(true);

    constexpr double kHold = 1.5;
    Scanner          scanner(listener);
    scanner.set_threshold(-55.0f);
    scanner.set_dmr_only(true);
    scanner.set_dmr_hold(kHold);
    scanner.start();

    struct Visit {
        double frequency, seconds;
        bool   voice;
    };
    std::vector<Visit> visits;
    const double       dmr_hz  = 98e6 + SimSource::digital_stations(2.4e6)[0].offset_hz;
    const int          columns = pipeline.columns();
    std::vector<float> row(static_cast<size_t>(columns));
    const auto         start = std::chrono::steady_clock::now();
    auto               since = start;
    double             at    = std::numeric_limits<double>::quiet_NaN();
    bool               voice = false;
    const auto         now_s = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
    while (now_s() < 8.0) {
        RowInfo info;
        if (pipeline.engine().pop_rows(row.data(), 1, &info) != 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        scanner.on_row(row.data(), info, {98e6, 2.4e6, columns, info.fft_size});
        const Scanner::Status st     = scanner.status();
        const bool            locked = st.state == Scanner::State::Receiving || st.state == Scanner::State::Holding;
        const double          here   = locked ? st.frequency : std::numeric_limits<double>::quiet_NaN();
        const bool            same   = std::isnan(here) ? std::isnan(at) : !std::isnan(at) && std::fabs(here - at) < 5e3;
        if (!same) {
            const auto t = std::chrono::steady_clock::now();
            if (!std::isnan(at))
                visits.push_back({at, std::chrono::duration<double>(t - since).count(), voice});
            at    = here;
            since = t;
            voice = false;
        }
        voice |= locked && st.voice;
    }
    if (!std::isnan(at))
        visits.push_back({at, std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count(), voice});

    bool kept = false, left_after = false;
    for (size_t i = 0; i < visits.size(); ++i) {
        const Visit& v      = visits[i];
        const bool   is_dmr = std::fabs(v.frequency - dmr_hz) < 5e3;
        std::printf("    %.4f MHz for %.1f s%s%s\n", v.frequency / 1e6, v.seconds, is_dmr ? " (DMR)" : "",
                    v.voice ? ", a call" : "");
        if (is_dmr && v.voice && v.seconds > 2.0) {
            kept       = true;
            left_after = i + 1 < visits.size();
        }
        if (!is_dmr && i + 1 < visits.size())
            CHECK(v.seconds < 2.0);  // not DMR: passed by
    }
    CHECK(kept);
    CHECK(left_after);

    scanner.stop();
    listener.set_enabled(false);
    pipeline.stop();
}
