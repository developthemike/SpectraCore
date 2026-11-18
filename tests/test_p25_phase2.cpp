#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

#include "dsp/ambe_vocoder.hpp"
#include "dsp/fm_demod.hpp"
#include "dsp/p25_coding.hpp"
#include "dsp/p25_phase2_coding.hpp"
#include "dsp/p25_phase2_decoder.hpp"
#include "dsp/p25_phase2_transmitter.hpp"
#include "test_framework.hpp"

using namespace spectro;

namespace {

constexpr double kPi = std::numbers::pi;
constexpr p25p2::System kSystem{0xBEE00, 0x669, 0x663};

double rrc_pulse(double t)
{
    constexpr double a = 0.2;
    if (std::fabs(t) < 1e-9)
        return 1.0 - a + 4.0 * a / kPi;
    if (std::fabs(std::fabs(t) - 1.0 / (4.0 * a)) < 1e-9)
        return a / std::sqrt(2.0) * ((1.0 + 2.0 / kPi) * std::sin(kPi / (4.0 * a)) + (1.0 - 2.0 / kPi) * std::cos(kPi / (4.0 * a)));
    return (std::sin(kPi * t * (1.0 - a)) + 4.0 * a * t * std::cos(kPi * t * (1.0 + a))) /
           (kPi * t * (1.0 - 16.0 * a * a * t * t));
}

// A Phase 2 base station's complex baseband, `rate` samples per second: real
// differential QPSK (phase steps of 45 and 135 degrees) through a root raised
// cosine, after `quiet` seconds off the air; `offset_hz` off tune.
// `at_symbol` is called before each symbol is taken (to start calls on time).
std::vector<std::complex<float>> phase2_baseband(P25Phase2Transmitter& tx, double seconds, double rate,
                                                 double quiet = 0.1, double offset_hz = 0.0,
                                                 const std::function<void(size_t)>& at_symbol = nullptr)
{
    const auto                        total = static_cast<size_t>(seconds * p25p2::kSymbolRate) + 32;
    const auto                        lead  = static_cast<size_t>(quiet * p25p2::kSymbolRate);
    std::vector<std::complex<double>> a(total, 0.0);
    double                            phase = 0.0;
    for (size_t k = lead; k < total; ++k) {
        if (at_symbol)
            at_symbol(k);
        int symbol = 0;
        if (!tx.next_symbol(symbol))
            continue;
        phase += symbol * kPi / 4.0;
        a[k] = std::polar(1.0, phase);
    }
    constexpr int       kReach = 8, kSteps = 1000;
    std::vector<double> pulse(2 * kReach * kSteps + 1);
    for (size_t i = 0; i < pulse.size(); ++i)
        pulse[i] = rrc_pulse(static_cast<double>(i) / kSteps - kReach);
    const auto                       n = static_cast<size_t>(seconds * rate);
    std::vector<std::complex<float>> x(n);
    for (size_t i = 0; i < n; ++i) {
        const double         position = static_cast<double>(i) / rate * p25p2::kSymbolRate - 0.27;
        const long           k0       = static_cast<long>(std::floor(position));
        std::complex<double> v        = 0.0;
        for (long k = k0 - kReach + 1; k <= k0 + kReach; ++k) {
            if (k < 0 || k >= static_cast<long>(total))
                continue;
            v += a[static_cast<size_t>(k)] * pulse[static_cast<size_t>(std::lround((position - static_cast<double>(k) + kReach) * kSteps))];
        }
        v *= std::polar(1.0, 2.0 * kPi * offset_hz * static_cast<double>(i) / rate);
        x[i] = std::complex<float>(v);
    }
    return x;
}

// What an ideal FM demodulator makes of it: the phase step per sample, in Hz,
// with noise where there is no signal.
std::vector<float> discriminate(const std::vector<std::complex<float>>& x, double rate, double noise_hz, unsigned seed = 1)
{
    std::mt19937                     rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<float>               f(x.size(), 0.0f);
    for (size_t i = 1; i < x.size(); ++i) {
        const auto d = x[i] * std::conj(x[i - 1]);
        f[i] = std::abs(d) < 1e-6f ? static_cast<float>(4000.0 * gauss(rng))
                                   : static_cast<float>(std::arg(d) * rate / (2.0 * kPi) + noise_hz * gauss(rng));
    }
    return f;
}

std::vector<P25Phase2Voice> decode(P25Phase2Decoder& decoder, const std::vector<float>& f, double rate,
                                   const std::function<void(const P25Phase2Status&)>& watch = nullptr)
{
    std::vector<P25Phase2Voice> voice;
    for (size_t done = 0; done < f.size(); done += 1000) {
        decoder.process(f.data() + done, std::min<size_t>(1000, f.size() - done), rate, voice);
        if (watch)
            watch(decoder.status());
    }
    return voice;
}

P25Phase2Transmitter::VoiceSource counting_voice(std::vector<dmr::AmbeFrame>& sent, unsigned seed)
{
    auto rng = std::make_shared<std::mt19937>(seed);
    return [&sent, rng] {
        dmr::AmbeFrame f;
        static constexpr size_t kBits[4] = {24, 23, 11, 14};
        for (size_t c = 0; c < 4; ++c)
            for (size_t b = 0; b < kBits[c]; ++b)
                f.c[c][b] = static_cast<char>((*rng)() & 1u);
        sent.push_back(f);
        return f;
    };
}

// How many frames heard were sent as they are.
size_t matching(const std::vector<dmr::AmbeFrame>& sent, const std::vector<P25Phase2Voice>& got)
{
    size_t n = 0;
    for (const P25Phase2Voice& v : got)
        n += std::find(sent.begin(), sent.end(), v.frame) != sent.end() ? 1 : 0;
    return n;
}

}  // namespace

TEST(p25_phase2_codes)
{
    std::mt19937 rng(9);
    // DUID: one error mended.
    bool duid_ok = true;
    for (unsigned v = 0; v < 16; ++v)
        duid_ok &= p25p2::duid_decode(static_cast<uint8_t>(p25p2::duid_codeword(static_cast<uint8_t>(v)) ^ (1u << (rng() % 8)))) ==
                   static_cast<int>(v);
    CHECK(duid_ok);
    // ISCH: up to 7 errors; the sync word too; and the timeslot in it.
    bool isch_ok = true;
    for (unsigned v = 0; v < 128; ++v) {
        uint64_t w = p25p2::isch_encode(v);
        for (int e = 0; e < 7; ++e)
            w ^= uint64_t{1} << (rng() % 40);
        isch_ok &= p25p2::isch_decode(w) == static_cast<int>(v) || std::popcount(w ^ p25p2::isch_encode(v)) < 7;
    }
    CHECK(isch_ok);
    CHECK_EQ(p25p2::isch_decode(p25p2::kSync ^ 0x1001), -2);
    for (int t : {0, 1, 4, 5, 8, 9})
        CHECK_EQ(p25p2::isch_slot(p25p2::isch_value(t)), t);

    // Scrambling: a sequence per system, 2160 dibits, not all alike.
    const std::vector<uint8_t> a = p25p2::scrambling(kSystem);
    const std::vector<uint8_t> b = p25p2::scrambling({0xBEE00, 0x669, 0x664});
    CHECK_EQ(a.size(), size_t{2160});
    CHECK(a != b);
    CHECK(std::count(a.begin(), a.end(), 0) > 400 && std::count(a.begin(), a.end(), 3) > 400);

    // Reed-Solomon (63,35) with its punctured parity as erasures, and errors.
    for (int trial = 0; trial < 30; ++trial) {
        uint8_t h[63] = {};
        for (size_t i = 9; i < 35; ++i)
            h[i] = static_cast<uint8_t>(rng() & 63u);
        p25::rs_encode(h, 63, 28);
        uint8_t sent[63];
        std::copy(h, h + 63, sent);
        std::vector<size_t> erasures;
        for (size_t i = 0; i < 63; ++i)
            if (i < 9 || i >= 54) {
                erasures.push_back(i);
                h[i] = 0;
            }
        for (int e = 0; e < 5; ++e) {
            const size_t at = 9 + rng() % 45;
            h[at]           = static_cast<uint8_t>(h[at] ^ (1 + rng() % 63));
        }
        CHECK(p25::rs63_decode(h, 28, erasures) >= 0);
        CHECK(std::equal(h, h + 63, sent));
    }

    // MAC PDUs survive a few errors in a FACCH and a SACCH, and say who calls.
    for (const bool facch : {true, false}) {
        std::array<uint8_t, p25p2::kSlotDibits> slot{};
        p25p2::encode_mac(p25p2::mac_ptt(facch, 200, 3001, 0x80), facch, slot.data());
        for (int e = 0; e < 3; ++e)
            slot[21 + rng() % 150] ^= 1u;
        std::vector<uint8_t> pdu;
        CHECK(p25p2::decode_mac(slot.data(), facch, pdu));
        CHECK_EQ(pdu.size(), facch ? p25p2::kFacchBytes : p25p2::kSacchBytes);
        CHECK_EQ(p25p2::mac_opcode(pdu), uint8_t{p25p2::kMacPtt});
        const auto call = p25p2::mac_call(pdu);
        CHECK(call && call->destination == 200u && call->source == 3001u && call->encryption_known && !call->encrypted);
    }
    const auto active = p25p2::mac_call(p25p2::mac_active(false, 7, 8, 0x40));
    CHECK(active && active->destination == 7u && active->source == 8u && active->encrypted);
}

TEST(p25_phase2_system_from_a_burst)
{
    // A clean signalling burst gives its system away; one with an error does not.
    std::mt19937 rng(12);
    int          right = 0, trials = 0, errored = 0, wrong = 0;
    for (int trial = 0; trial < 12; ++trial) {
        const p25p2::System system{static_cast<uint32_t>(rng() & 0xFFFFFu), static_cast<uint16_t>(rng() & 0xFFFu),
                                   static_cast<uint16_t>(rng() & 0xFFFu)};
        const std::vector<uint8_t> mask = p25p2::scrambling(system);
        for (const bool facch : {true, false}) {
            const int t = trial;  // each timeslot of the superframe in turn
            std::vector<uint8_t> pdu(facch ? p25p2::kFacchBytes : p25p2::kSacchBytes);
            for (uint8_t& b : pdu)
                b = static_cast<uint8_t>(rng());
            std::array<uint8_t, p25p2::kSlotDibits> slot{};
            p25p2::encode_mac(pdu, facch, slot.data());
            for (size_t i = 20; i < p25p2::kSlotDibits; ++i)
                slot[i] = static_cast<uint8_t>(slot[i] ^ mask[static_cast<size_t>(t) * p25p2::kSlotDibits + i - 10]);
            const auto found = p25p2::recover_system(slot.data(), t, facch);
            ++trials;
            right += found && *found == system ? 1 : 0;
            slot[21 + rng() % 36] ^= static_cast<uint8_t>(1u + rng() % 3);  // one or two bits of a dibit both kinds carry
            const auto damaged = p25p2::recover_system(slot.data(), t, facch);
            errored += damaged ? 0 : 1;
            wrong += damaged && !(*damaged == system) ? 1 : 0;
        }
    }
    std::printf("    %d of %d systems worked out; with an error, %d of %d gave none\n", right, trials, errored, trials);
    CHECK_EQ(right, trials);
    CHECK_EQ(wrong, 0);
    CHECK_EQ(errored, trials);
}

TEST(p25_phase2_decoder_follows_calls)
{
    // A call on slot 0, then while it goes on one on slot 1; the decoder
    // knows the system, so it reads both and plays the first.
    P25Phase2Transmitter        tx(kSystem);
    std::vector<dmr::AmbeFrame> sent0, sent1;
    tx.start_call(0, 200, 3001, 6, counting_voice(sent0, 1));
    P25Phase2Decoder decoder(48000.0);
    decoder.set_systems({{1, 2, 3}, kSystem});  // tries both: the second checks
    bool       saw0 = false, saw1 = false, both = false;
    // The first call alone for 0.8 s, then the second joins.
    const auto x = phase2_baseband(tx, 4.0, 48000.0, 0.1, 300.0, [&](size_t k) {
        if (k == static_cast<size_t>(0.8 * p25p2::kSymbolRate))
            tx.start_call(1, 201, 3002, 3, counting_voice(sent1, 2));
    });
    const auto voice = decode(decoder, discriminate(x, 48000.0, 100.0), 48000.0, [&](const P25Phase2Status& s) {
        saw0 |= s.slot[0].active && s.slot[0].ids_known && s.slot[0].destination == 200 && s.slot[0].source == 3001;
        saw1 |= s.slot[1].active && s.slot[1].ids_known && s.slot[1].destination == 201 && s.slot[1].source == 3002;
        both |= s.slot[0].active && s.slot[1].active;
    });
    const P25Phase2Status& s = decoder.status();
    CHECK(s.keyed);
    CHECK(s.system == kSystem);
    CHECK(saw0);
    CHECK(saw1);
    CHECK(both);
    CHECK(!s.slot[0].active && !s.slot[1].active);  // both ended
    // Slot 0's voice, all of it (6 superframes of 18 frames, less the 4 its PTT takes).
    CHECK(!voice.empty() && voice.front().slot == 0 && voice.front().first);
    CHECK(std::all_of(voice.begin(), voice.end(), [](const P25Phase2Voice& v) { return v.slot == 0; }));
    std::printf("    %zu of %zu frames sent on slot 0 heard intact; %llu signalling PDUs\n", matching(sent0, voice),
                sent0.size(), static_cast<unsigned long long>(s.macs));
    CHECK_EQ(voice.size(), sent0.size());
    CHECK_EQ(matching(sent0, voice), sent0.size());
}

TEST(p25_phase2_works_out_the_system)
{
    // Not told the system - or told only others: it is worked out from the
    // signalling, and the call heard all the same.  With set_slot(1) only
    // slot 1 is heard; an encrypted call not at all.
    for (const bool others : {false, true}) {
        P25Phase2Transmitter        tx(kSystem);
        std::vector<dmr::AmbeFrame> sent;
        tx.start_call(0, 200, 3001, 4, counting_voice(sent, 3));
        P25Phase2Decoder decoder(48000.0);
        if (others)
            decoder.set_systems({{0xBEE00, 0x669, 0x664}, {1, 2, 3}});
        bool       named = false;
        const auto voice = decode(decoder, discriminate(phase2_baseband(tx, 2.4, 48000.0), 48000.0, 100.0), 48000.0,
                                  [&](const P25Phase2Status& s) {
                                      named |= s.slot[0].active && s.slot[0].ids_known && s.slot[0].destination == 200 &&
                                               s.slot[0].source == 3001;
                                  });
        const P25Phase2Status& s = decoder.status();
        CHECK(s.keyed);
        CHECK(s.recovered);
        CHECK(s.system == kSystem);
        CHECK(named);
        std::printf("    %s: %zu of %zu frames heard intact\n", others ? "other systems given" : "no system given",
                    matching(sent, voice), sent.size());
        CHECK(matching(sent, voice) + 4 >= sent.size());  // perhaps less the first burst, before a second checked
    }
    {
        P25Phase2Transmitter        tx(kSystem);
        std::vector<dmr::AmbeFrame> sent0, sent1;
        tx.start_call(0, 200, 3001, 4, counting_voice(sent0, 4));
        tx.start_call(1, 300, 4001, 4, counting_voice(sent1, 5), true);  // encrypted
        P25Phase2Decoder decoder(48000.0);
        decoder.set_systems({kSystem});
        decoder.set_slot(1);
        bool       encrypted = false;
        const auto voice = decode(decoder, discriminate(phase2_baseband(tx, 1.8, 48000.0), 48000.0, 100.0), 48000.0,
                                  [&](const P25Phase2Status& s) { encrypted |= s.slot[1].active && s.slot[1].encrypted; });
        CHECK(encrypted);
        CHECK(voice.empty());  // slot 0 not asked for, slot 1 encrypted
    }
}

TEST(p25_phase2_works_through_the_radio)
{
    // Received by FmDemodulator at 16 dB SNR, 400 Hz off tune.
    constexpr double     kIqRate = 240e3, kOffset = 50e3;
    P25Phase2Transmitter tx(kSystem);
    std::vector<dmr::AmbeFrame> sent;
    tx.start_call(1, 2403, 77, 5, counting_voice(sent, 6));
    const std::vector<std::complex<float>> x = phase2_baseband(tx, 2.2, kIqRate, 0.2, 400.0);

    std::mt19937                     rng(19);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const double                     amplitude = 2048.0 * 0.1, bandwidth = 12.5e3;
    const double sigma = amplitude * std::sqrt(std::pow(10.0, -16.0 / 10.0) * (kIqRate / bandwidth) / 2.0);
    std::vector<int16_t> iq(2 * x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const auto shifted = std::complex<double>(x[i]) * std::polar(1.0, 2.0 * kPi * kOffset * static_cast<double>(i) / kIqRate);
        iq[2 * i]     = static_cast<int16_t>(std::lround(amplitude * shifted.real() + sigma * gauss(rng)));
        iq[2 * i + 1] = static_cast<int16_t>(std::lround(amplitude * shifted.imag() + sigma * gauss(rng)));
    }
    FmDemodulator demod(kIqRate, 48000.0);
    demod.set_channel(kOffset, bandwidth);
    P25Phase2Decoder decoder(48000.0);
    decoder.set_systems({kSystem});
    FmDemodulator::Analysis     analysis;
    std::vector<float>          audio;
    std::vector<P25Phase2Voice> voice;
    bool                        named = false;
    for (size_t done = 0; done < x.size(); done += 8192) {
        demod.process(iq.data() + 2 * done, std::min<size_t>(8192, x.size() - done), audio, &analysis);
        decoder.process(analysis.frequency.data(), analysis.frequency.size(), 48000.0, voice);
        named |= decoder.status().slot[1].ids_known && decoder.status().slot[1].destination == 2403;
        analysis.frequency.clear();
        analysis.power.clear();
        audio.clear();
    }
    CHECK(decoder.status().keyed);
    CHECK(named);
    const size_t good = matching(sent, voice);
    std::printf("    %zu of %zu frames sent heard intact\n", good, sent.size());
    CHECK(good >= sent.size() * 90 / 100);
}

TEST(p25_phase2_voice_is_heard)
{
    if (!AmbeDecoder::available()) {
        std::printf("    (built without a vocoder: skipped)\n");
        return;
    }
    AmbeEncoder encoder;
    size_t      t     = 0;
    const auto  voice = [&] {
        float frame[AmbeEncoder::kFrameSamples];
        for (float& v : frame) {
            const double s = static_cast<double>(t++) / 8000.0;
            double       x = 0.0;
            for (int h = 1; h <= 20; ++h)
                x += std::exp(-std::pow((h * 120.0 - 600.0) / 400.0, 2.0)) * std::sin(2.0 * kPi * h * 120.0 * s) / h;
            v = static_cast<float>(0.3 * x);
        }
        return encoder.encode(frame);
    };
    P25Phase2Transmitter tx(kSystem);
    tx.start_call(0, 1, 1, 6, voice);
    P25Phase2Decoder decoder(48000.0);
    decoder.set_systems({kSystem});
    const auto  frames = decode(decoder, discriminate(phase2_baseband(tx, 2.4, 48000.0), 48000.0, 100.0), 48000.0);
    AmbeDecoder vocoder;
    std::vector<float> heard;
    for (const P25Phase2Voice& v : frames) {
        if (v.first)
            vocoder.reset();
        float pcm[AmbeDecoder::kFrameSamples];
        vocoder.decode(v.frame, pcm);
        heard.insert(heard.end(), pcm, pcm + AmbeDecoder::kFrameSamples);
    }
    CHECK(frames.size() >= 80u);
    const size_t from = heard.size() / 2, n = 1600;
    double       best = 0.0, rms = 0.0;
    int          lag  = 0;
    for (int l = 30; l < 120; ++l) {
        double c = 0.0;
        for (size_t i = 0; i < n; ++i)
            c += heard[from + i] * heard[from + i + static_cast<size_t>(l)];
        if (c > best) {
            best = c;
            lag  = l;
        }
    }
    for (size_t i = 0; i < n; ++i)
        rms += heard[from + i] * heard[from + i];
    rms = std::sqrt(rms / n);
    std::printf("    pitch %.1f Hz, level %.3f\n", 8000.0 / lag, rms);
    CHECK(std::fabs(8000.0 / lag - 120.0) < 6.0);
    CHECK(rms > 0.02);
}
