#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "app/pipeline.hpp"
#include "dsp/fm_demod.hpp"
#include "dsp/signal_classifier.hpp"
#include "radio/sim_source.hpp"
#include "test_framework.hpp"

using namespace spectro;

namespace {

constexpr double kPi   = std::numbers::pi;
constexpr double kRate = 240e3;  // IQ sample rate of the test signals
constexpr double kOffset = 30e3; // channel offset from the centre

// The frequency (Hz) of a test transmitter at time t.
using Modulation = std::function<double(double t)>;

// Transmits `modulation` for `seconds` with the given SNR in a channel `bandwidth`
// wide, receives it with FmDemodulator and returns what the classifier makes of it.
SignalClass classify(const Modulation& modulation, double snr_db, double bandwidth = 12.5e3, double seconds = 3.0,
                     unsigned seed = 1)
{
    const size_t                     n = static_cast<size_t>(seconds * kRate);
    std::mt19937                     rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    // Noise in the channel at the requested SNR: its power spreads over the whole stream.
    const double amplitude = 2048.0 * 0.1;
    const double sigma =
        amplitude * std::sqrt(std::pow(10.0, -snr_db / 10.0) * (kRate / std::min(bandwidth, kRate)) / 2.0);

    std::vector<int16_t> iq(2 * n);
    double               phase = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / kRate;
        phase          = std::fmod(phase + 2.0 * kPi * (kOffset + modulation(t)) / kRate, 2.0 * kPi);
        iq[2 * i]      = static_cast<int16_t>(std::lround(amplitude * std::cos(phase) + sigma * gauss(rng)));
        iq[2 * i + 1]  = static_cast<int16_t>(std::lround(amplitude * std::sin(phase) + sigma * gauss(rng)));
    }

    FmDemodulator demod(kRate, 48000.0);
    demod.set_channel(kOffset, bandwidth);
    SignalClassifier classifier(48000.0);
    classifier.set_channel(demod.mode(), demod.bandwidth());
    FmDemodulator::Analysis analysis;
    std::vector<float>      audio;
    for (size_t done = 0; done < n; done += 8192) {
        demod.process(iq.data() + 2 * done, std::min<size_t>(8192, n - done), audio, &analysis);
        classifier.process(analysis.frequency.data(), analysis.power.data(), analysis.frequency.size());
        analysis.frequency.clear();
        analysis.power.clear();
        audio.clear();
    }
    return classifier.result();
}

// A raised-cosine pulse: 1 at its own symbol, 0 at all others.
double raised_cosine(double t, double alpha)
{
    if (std::fabs(t) < 1e-9)
        return 1.0;
    const double edge = 1.0 - 4.0 * alpha * alpha * t * t;
    const double s    = std::sin(kPi * t) / (kPi * t);
    if (std::fabs(edge) < 1e-9)
        return 0.25 * kPi * s;  // the limit there
    return s * std::cos(kPi * alpha * t) / edge;
}

// Pulse-shaped multi-level FSK: `symbols` (values such as -3..+3) times
// `step` Hz, each shaped by `pulse` (t in symbols), at `rate` symbols per second.
Modulation shaped_fsk(std::vector<int> symbols, double rate, double step, std::function<double(double)> pulse,
                      int reach = 8)
{
    return [=](double t) {
        const double position = t * rate;
        const long   k        = static_cast<long>(std::floor(position));
        double       f        = 0.0;
        for (long j = k - reach; j <= k + reach; ++j) {
            if (j >= 0 && j < static_cast<long>(symbols.size()))
                f += symbols[static_cast<size_t>(j)] * step * pulse(position - static_cast<double>(j));
        }
        return f;
    };
}

// Random 4FSK symbols (-3, -1, +1, +3) with a sync pattern ('1' = +3, '3' = -3)
// every `period` symbols.
std::vector<int> fsk4_symbols(size_t count, const std::string& sync, size_t period, unsigned seed = 7)
{
    std::mt19937     rng(seed);
    std::vector<int> s(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t at = i % period;
        if (!sync.empty() && at >= 60 && at < 60 + sync.size())
            s[i] = sync[at - 60] == '1' ? 3 : -3;
        else
            s[i] = 2 * static_cast<int>(rng() % 4) - 3;
    }
    return s;
}

const std::string kDmrDataSync = "313333111331131131331131";
const std::string kP25Sync     = "111113113311333313133333";

// Something like speech: a voice whose pitch wanders between 100 and 190 Hz,
// with two formants and syllables about four times a second.
double speech(double t)
{
    const double pitch = 145.0 + 45.0 * std::sin(2.0 * kPi * 0.7 * t) + 10.0 * std::sin(2.0 * kPi * 3.1 * t);
    // Phase of the pitch: its integral.
    const double cycles = 145.0 * t - 45.0 / (2.0 * kPi * 0.7) * std::cos(2.0 * kPi * 0.7 * t) -
                          10.0 / (2.0 * kPi * 3.1) * std::cos(2.0 * kPi * 3.1 * t);
    (void)pitch;
    double v = 0.0;
    for (int h = 1; h <= 25; ++h) {
        const double f    = h * pitch;
        if (f > 3500.0)
            break;
        const double gain = std::exp(-std::pow((f - 500.0) / 250.0, 2.0)) + 0.6 * std::exp(-std::pow((f - 1500.0) / 400.0, 2.0)) + 0.05;
        v += gain * std::sin(2.0 * kPi * h * cycles + 0.3 * h);
    }
    const double syllable = std::max(0.0, std::sin(2.0 * kPi * 2.0 * t + 0.4 * std::sin(2.0 * kPi * 0.3 * t)));
    return v * syllable;
}

}  // namespace

TEST(classifier_names_dmr_and_p25)
{
    const auto rc = [](double t) { return raised_cosine(t, 0.2); };
    // DMR: 4800 Bd, levels +/-648 and +/-1944 Hz, a sync pattern in every
    // 144-symbol burst.
    SignalClass dmr = classify(shaped_fsk(fsk4_symbols(20000, kDmrDataSync, 144), 4800.0, 648.0, rc), 25.0);
    CHECK(dmr.kind == SignalKind::Fsk4);
    CHECK_NEAR(dmr.symbol_rate, 4800.0, 1e-9);
    CHECK_NEAR(dmr.deviation_hz, 1944.0, 150.0);
    CHECK(dmr.protocol == "DMR");
    CHECK(dmr.digital());
    CHECK(describe(dmr).find("4FSK, 4800 Bd") != std::string::npos);
    CHECK(short_name(dmr) == "DMR");

    // Inverted (a receiver that swaps the sidebands): still DMR.
    std::vector<int> inverted = fsk4_symbols(20000, kDmrDataSync, 144);
    for (int& s : inverted)
        s = -s;
    CHECK(classify(shaped_fsk(inverted, 4800.0, 648.0, rc), 25.0).protocol == "DMR");

    // P25 Phase 1: 4800 Bd, +/-600 and +/-1800 Hz, its frame sync.
    SignalClass p25 = classify(shaped_fsk(fsk4_symbols(20000, kP25Sync, 864), 4800.0, 600.0, rc), 25.0);
    CHECK(p25.kind == SignalKind::Fsk4);
    CHECK(p25.protocol == "P25 Phase 1");

    // The same modulation without a known sync pattern: only the family.
    SignalClass plain = classify(shaped_fsk(fsk4_symbols(20000, "", 144), 4800.0, 648.0, rc), 25.0);
    CHECK(plain.kind == SignalKind::Fsk4);
    CHECK(plain.protocol.empty());
    CHECK(typical_systems(plain).find("DMR") != std::string::npos);

    // 2400 Bd 4FSK, as NXDN in 6.25 kHz channels and dPMR use.
    SignalClass nxdn = classify(shaped_fsk(fsk4_symbols(10000, "", 144), 2400.0, 350.0, rc), 25.0);
    CHECK(nxdn.kind == SignalKind::Fsk4);
    CHECK_NEAR(nxdn.symbol_rate, 2400.0, 1e-9);
    CHECK(typical_systems(nxdn).find("NXDN") != std::string::npos);
}

TEST(classifier_tells_two_level_modulations)
{
    // POCSAG: 1200 bit/s 2FSK at +/-4.5 kHz, batches starting with the frame sync.
    std::vector<int> bits;
    std::mt19937     rng(3);
    for (int batch = 0; batch < 8; ++batch) {
        for (int b = 31; b >= 0; --b)
            bits.push_back((0x7CD215D8u >> b) & 1u ? -1 : 1);
        for (int b = 0; b < 16 * 32; ++b)
            bits.push_back(rng() & 1u ? -1 : 1);
    }
    SignalClass pocsag = classify(shaped_fsk(bits, 1200.0, 4500.0, [](double t) { return raised_cosine(t, 1.0); }), 25.0,
                                  25e3);
    CHECK(pocsag.kind == SignalKind::Fsk2);
    CHECK_NEAR(pocsag.symbol_rate, 1200.0, 1e-9);
    CHECK_NEAR(pocsag.deviation_hz, 4500.0, 500.0);
    CHECK(pocsag.protocol == "POCSAG");

    // GMSK, BT 0.3, at 9600 and 4800 Bd: a modulation index of 0.5.
    const auto gaussian = [](double bt) {
        return [bt](double t) {
            const double k = kPi * bt * std::sqrt(2.0 / std::log(2.0));
            return 0.5 * (std::erf(k * (t + 0.5)) - std::erf(k * (t - 0.5)));
        };
    };
    std::vector<int> random(30000);
    for (int& b : random)
        b = rng() & 1u ? -1 : 1;
    SignalClass gmsk = classify(shaped_fsk(random, 9600.0, 2400.0, gaussian(0.3)), 25.0, 25e3);
    CHECK(gmsk.kind == SignalKind::Gmsk);
    CHECK_NEAR(gmsk.symbol_rate, 9600.0, 1e-9);
    SignalClass dstar = classify(shaped_fsk(random, 4800.0, 1200.0, gaussian(0.5)), 25.0);
    CHECK(dstar.kind == SignalKind::Gmsk);
    CHECK_NEAR(dstar.symbol_rate, 4800.0, 1e-9);
    CHECK(typical_systems(dstar) == "D-STAR");

    // AFSK: Bell 202 tones (1200 / 2200 Hz) keyed at 1200 Bd, on FM.
    std::vector<int> keyed(4000);
    for (int& b : keyed)
        b = static_cast<int>(rng() & 1u);
    auto tone_phase = std::make_shared<std::vector<double>>();
    {
        // Continuous-phase tones, precomputed per bit.
        double phase = 0.0;
        for (const int b : keyed) {
            tone_phase->push_back(phase);
            phase += 2.0 * kPi * (b ? 1200.0 : 2200.0) / 1200.0;
        }
    }
    const Modulation afsk = [keyed, tone_phase](double t) {
        const size_t k = std::min(static_cast<size_t>(t * 1200.0), keyed.size() - 1);
        const double u = t * 1200.0 - static_cast<double>(k);
        return 3000.0 * std::sin((*tone_phase)[k] + 2.0 * kPi * (keyed[k] ? 1200.0 : 2200.0) * u / 1200.0);
    };
    SignalClass packet = classify(afsk, 25.0);
    CHECK(packet.kind == SignalKind::Afsk);
    CHECK(typical_systems(packet).find("APRS") != std::string::npos);
}

TEST(classifier_tells_analog_and_carriers)
{
    // Speech-like audio at up to about 2.5 kHz deviation: analog.
    SignalClass voice = classify([](double t) { return 900.0 * speech(t); }, 25.0);
    CHECK(voice.kind == SignalKind::Analog);
    CHECK(!voice.digital());
    CHECK_NEAR(voice.ctcss_hz, 0.0, 1e-9);

    // ... with a sub-audible tone, which is named.
    SignalClass toned = classify([](double t) { return 900.0 * speech(t) + 500.0 * std::sin(2.0 * kPi * 100.0 * t); }, 25.0);
    CHECK(toned.kind == SignalKind::Analog);
    CHECK_NEAR(toned.ctcss_hz, 100.0, 1e-9);
    CHECK(describe(toned).find("CTCSS 100.0 Hz") != std::string::npos);

    // A carrier without audio, with and without a tone.
    CHECK(classify([](double) { return 0.0; }, 25.0).kind == SignalKind::Carrier);
    SignalClass keyed = classify([](double t) { return 600.0 * std::sin(2.0 * kPi * 131.8 * t); }, 25.0);
    CHECK(keyed.kind == SignalKind::Carrier);
    CHECK_NEAR(keyed.ctcss_hz, 131.8, 1e-9);

    // A plain tone makes a line at twice its frequency among the
    // "transitions" - 2400 Hz looks like 4800 Bd - but it is no data.
    for (const double tone : {1000.0, 2400.0}) {
        SignalClass t = classify([tone](double s) { return 2500.0 * std::sin(2.0 * kPi * tone * s); }, 25.0);
        CHECK(t.kind == SignalKind::Analog);
    }

    // Noise, and a digital signal too weak to tell.
    CHECK(classify([](double) { return 0.0; }, -10.0).kind == SignalKind::Noise);
    const auto rc = [](double t) { return raised_cosine(t, 0.2); };
    SignalClass weak = classify(shaped_fsk(fsk4_symbols(20000, kDmrDataSync, 144), 4800.0, 648.0, rc), 2.0);
    CHECK(!weak.digital());
}

TEST(classifier_tells_stereo_broadcasts)
{
    // Wideband: programme audio and, for stereo, the 19 kHz pilot.
    const Modulation mono   = [](double t) { return 20000.0 * speech(t); };
    const Modulation stereo = [](double t) { return 20000.0 * speech(t) + 6750.0 * std::sin(2.0 * kPi * 19000.0 * t); };
    SignalClass a = classify(mono, 25.0, 200e3);
    SignalClass b = classify(stereo, 25.0, 200e3);
    CHECK(a.kind == SignalKind::Broadcast);
    CHECK(!a.stereo);
    CHECK(b.kind == SignalKind::Broadcast);
    CHECK(b.stereo);
    CHECK(describe(b) == "Broadcast FM, stereo");
}

TEST(classifier_waits_for_agreement)
{
    // One window is not enough for a verdict, and it starts afresh on a new channel.
    SignalClassifier   classifier(48000.0);
    std::vector<float> f(12000, 0.0f), p(12000, 1.0f);
    classifier.set_channel(FmMode::Narrow, 12.5e3);
    classifier.process(f.data(), p.data(), f.size());
    classifier.process(f.data(), p.data(), f.size());
    CHECK_EQ(classifier.windows(), uint64_t{1});
    CHECK(classifier.result().kind == SignalKind::Analysing);
    classifier.process(f.data(), p.data(), f.size());
    classifier.process(f.data(), p.data(), f.size());
    CHECK_EQ(classifier.windows(), uint64_t{2});
    CHECK(classifier.result().kind == SignalKind::Carrier);
    classifier.set_channel(FmMode::Narrow, 25e3);
    CHECK(classifier.result().kind == SignalKind::Analysing);
}

TEST(listener_tells_what_the_simulator_sends)
{
    // Through the whole receive path: simulator -> receive thread -> listener.
    Options options;
    options.source.kind        = "sim";
    options.source.sample_rate = 2.4e6;
    options.source.frequency   = 150e6;
    options.source.gain        = SimSource::kReferenceGainDb;
    Pipeline pipeline(options);
    pipeline.start();
    Listener& listener = pipeline.listener();
    listener.set_volume(0.0f);
    listener.set_enabled(true);

    const auto verdict = [&](double offset_hz, double bandwidth) {
        listener.tune(150e6 + offset_hz, 0.0, bandwidth);
        std::this_thread::sleep_for(std::chrono::milliseconds(1800));
        return listener.status().signal;
    };
    const std::vector<SimSource::DigitalStation> digital = SimSource::digital_stations(2.4e6);

    const SignalClass dmr = verdict(digital[0].offset_hz, 12.5e3);
    CHECK(dmr.kind == SignalKind::Fsk4);
    CHECK_NEAR(dmr.symbol_rate, 4800.0, 1e-9);
    CHECK(dmr.protocol == "DMR");

    const SignalClass pager = verdict(digital[1].offset_hz, 16e3);
    CHECK(pager.kind == SignalKind::Fsk2);
    CHECK(pager.protocol == "POCSAG");

    const SignalClass tune = verdict(SimSource::voice_channel(2.4e6).offset_hz, 12.5e3);
    CHECK(tune.kind == SignalKind::Analog);
    CHECK(verdict(SimSource::carriers(2.4e6)[0].offset_hz, 12.5e3).kind == SignalKind::Carrier);
    const SignalClass station = verdict(SimSource::broadcast_station(2.4e6).offset_hz, 200e3);
    CHECK(station.kind == SignalKind::Broadcast);
    CHECK(station.stereo);

    // Nothing there: noise.
    CHECK(verdict(0.47 * 2.4e6 - 300e3, 12.5e3).kind == SignalKind::Noise);

    listener.set_enabled(false);
    pipeline.stop();
}

TEST(classifier_holds_up_at_low_snr)
{
    // A digital signal is still named down to 8 dB in its channel, and below
    // that it is never taken for analog audio.
    const auto rc = [](double t) { return raised_cosine(t, 0.2); };
    for (unsigned seed = 1; seed <= 3; ++seed) {
        const Modulation dmr = shaped_fsk(fsk4_symbols(20000, kDmrDataSync, 144, seed), 4800.0, 648.0, rc);
        CHECK(classify(dmr, 9.0, 12.5e3, 3.0, seed).protocol == "DMR");
        const SignalClass weak = classify(dmr, 6.5, 12.5e3, 3.0, seed);
        CHECK(weak.kind == SignalKind::Fsk4 || weak.kind == SignalKind::Weak);
    }
}
