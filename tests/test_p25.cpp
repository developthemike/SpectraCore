#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <random>
#include <thread>
#include <vector>

#include "app/pipeline.hpp"
#include "audio/heard_channels.hpp"
#include "audio/listener.hpp"
#include "dsp/ambe_vocoder.hpp"
#include "dsp/fm_demod.hpp"
#include "dsp/p25_coding.hpp"
#include "dsp/p25_decoder.hpp"
#include "dsp/p25_transmitter.hpp"
#include "dsp/p25_trunking.hpp"
#include "radio/sim_source.hpp"
#include "test_framework.hpp"

using namespace spectro;

namespace {

constexpr double kPi = std::numbers::pi;

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

// The frequency a P25 radio sends, `rate` samples per second: `quiet`
// seconds of nothing (noise, as an FM receiver hears no carrier), then the
// call `tx` has been given, 600 Hz per unit symbol.  `on_air` reports which.
std::vector<float> p25_frequency(P25Transmitter& tx, double seconds, double rate, double quiet = 0.2,
                                 double offset_hz = 0.0, double noise_hz = 0.0, unsigned seed = 1,
                                 std::vector<bool>* on_air = nullptr)
{
    const auto       total = static_cast<size_t>(seconds * P25Decoder::kSymbolRate) + 32;
    const auto       lead  = static_cast<size_t>(quiet * P25Decoder::kSymbolRate);
    std::vector<int> s(total, 0);
    std::vector<bool> air(total, false);
    for (size_t k = lead; k < total; ++k) {
        int symbol = 0;
        if (tx.next_symbol(symbol)) {
            s[k]   = symbol;
            air[k] = true;
        }
    }
    constexpr int       kReach = 8, kSteps = 1000;
    std::vector<double> pulse(2 * kReach * kSteps + 1);
    for (size_t i = 0; i < pulse.size(); ++i)
        pulse[i] = rrc_pulse(static_cast<double>(i) / kSteps - kReach);

    std::mt19937                     rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const auto                       n = static_cast<size_t>(seconds * rate);
    std::vector<float>               f(n);
    if (on_air)
        on_air->assign(n, false);
    for (size_t i = 0; i < n; ++i) {
        const double position = static_cast<double>(i) / rate * P25Decoder::kSymbolRate - 0.31;
        const long   k0       = static_cast<long>(std::floor(position));
        double       v        = 0.0;
        for (long k = k0 - kReach + 1; k <= k0 + kReach; ++k) {
            if (k < 0 || k >= static_cast<long>(total))
                continue;
            v += s[static_cast<size_t>(k)] * pulse[static_cast<size_t>(std::lround((position - static_cast<double>(k) + kReach) * kSteps))];
        }
        const bool here = k0 >= 0 && k0 < static_cast<long>(total) && air[static_cast<size_t>(k0)];
        if (on_air)
            (*on_air)[i] = here;
        f[i] = static_cast<float>(here ? 600.0 * v + offset_hz + noise_hz * gauss(rng) : 4000.0 * gauss(rng));
    }
    return f;
}

std::vector<P25Voice> decode(P25Decoder& decoder, const std::vector<float>& frequency, double rate,
                             const std::function<void(const P25Status&)>& watch = nullptr)
{
    std::vector<P25Voice> voice;
    for (size_t done = 0; done < frequency.size(); done += 1000) {
        decoder.process(frequency.data() + done, std::min<size_t>(1000, frequency.size() - done), rate, voice);
        if (watch)
            watch(decoder.status());
    }
    return voice;
}

P25Transmitter::VoiceSource counting_voice(std::vector<p25::ImbeFrame>& sent, unsigned seed)
{
    auto rng = std::make_shared<std::mt19937>(seed);
    return [&sent, rng] {
        static constexpr size_t kBits[8] = {23, 23, 23, 23, 15, 15, 15, 7};
        p25::ImbeFrame          f;
        for (size_t c = 0; c < 8; ++c)
            for (size_t b = 0; b < kBits[c]; ++b)
                f.c[c][b] = static_cast<char>((*rng)() & 1u);
        sent.push_back(f);
        return f;
    };
}

size_t matching(const std::vector<p25::ImbeFrame>& sent, const std::vector<P25Voice>& got)
{
    if (got.empty())
        return 0;
    const auto first = std::find(sent.begin(), sent.end(), got.front().frame);
    if (first == sent.end())
        return 0;
    size_t n = 0;
    for (size_t i = 0; i < got.size() && first + static_cast<long>(i) < sent.end(); ++i)
        n += got[i].frame == *(first + static_cast<long>(i)) ? 1 : 0;
    return n;
}

p25::LinkControl group_call(uint32_t talkgroup, uint32_t source)
{
    p25::LinkControl lc;
    lc.destination = talkgroup;
    lc.source      = source;
    return lc;
}

}  // namespace

TEST(p25_codes_correct_what_they_can)
{
    std::mt19937 rng(5);
    // NID: BCH (63,16), distance 23 - eight errors are corrected here.
    for (int trial = 0; trial < 20; ++trial) {
        const auto     nac  = static_cast<uint16_t>(rng() & 0xFFFu);
        const uint8_t  duid = trial % 2 ? p25::kLdu1 : p25::kTdu;
        uint64_t       nid  = p25::nid_encode(nac, duid);
        CHECK_EQ(nid & 1u, duid == p25::kLdu1 ? 1u : 0u);  // the parity bit
        for (int e = 0; e < 8; ++e)
            nid ^= uint64_t{1} << (1 + rng() % 63);
        uint16_t got_nac  = 0;
        uint8_t  got_duid = 0;
        CHECK(p25::nid_decode(nid, got_nac, got_duid) >= 0 && got_nac == nac && got_duid == duid);
    }
    int min_weight = 64;
    for (uint32_t v = 1; v < 65536; v += 97)
        min_weight = std::min(min_weight, std::popcount(p25::nid_encode(static_cast<uint16_t>(v >> 4), static_cast<uint8_t>(v & 15)) >> 1));
    CHECK(min_weight >= 23);

    // Hamming (10,6,3): one error.
    bool hamming_ok = true;
    for (unsigned v = 0; v < 64; ++v) {
        uint8_t h = 0;
        hamming_ok &= p25::hamming1063_decode(static_cast<uint16_t>(p25::hamming1063_encode(static_cast<uint8_t>(v)) ^ (1u << (rng() % 10))), h) == 1 &&
                      h == v;
    }
    CHECK(hamming_ok);

    // Reed-Solomon over GF(64): up to half the parity in errors.
    for (const size_t parity : {size_t{12}, size_t{8}, size_t{16}}) {
        const size_t n = parity == 16 ? 36 : 24;
        for (int trial = 0; trial < 30; ++trial) {
            std::vector<uint8_t> h(n);
            for (size_t i = 0; i < n - parity; ++i)
                h[i] = static_cast<uint8_t>(rng() & 63u);
            p25::rs_encode(h.data(), n, parity);
            const std::vector<uint8_t> sent = h;
            CHECK_EQ(p25::rs_decode(h.data(), n, parity), 0);
            std::vector<size_t> where(n);
            for (size_t i = 0; i < n; ++i)
                where[i] = i;
            std::shuffle(where.begin(), where.end(), rng);
            for (size_t e = 0; e < parity / 2; ++e)
                h[where[e]] = static_cast<uint8_t>(h[where[e]] ^ (1 + rng() % 63));
            CHECK_EQ(p25::rs_decode(h.data(), n, parity), static_cast<int>(parity / 2));
            CHECK(h == sent);
        }
    }
}

TEST(p25_link_control_round_trips)
{
    p25::LinkControl lc = group_call(100, 2001);
    lc.options          = p25::LinkControl::kEncrypted;
    std::array<uint8_t, 240> bits = p25::encode_link_control(lc);
    // A few bit errors: the Hamming and Reed-Solomon codes mend them.
    for (size_t i = 3; i < 240; i += 47)
        bits[i] ^= 1u;
    p25::LinkControl got;
    CHECK(p25::decode_link_control(bits, got));
    CHECK(got.group() && got.encrypted() && got.destination == 100u && got.source == 2001u);

    p25::LinkControl to_radio;
    to_radio.lcf         = p25::LinkControl::kPrivateVoice;
    to_radio.destination = 1234567;
    to_radio.source      = 7654321;
    CHECK(p25::decode_link_control(p25::encode_link_control(to_radio), got));
    CHECK(!got.group() && got.destination == 1234567u && got.source == 7654321u);

    p25::EncryptionSync es;
    es.algorithm = 0x84;
    es.key       = 0x1234;
    p25::EncryptionSync es_got;
    CHECK(p25::decode_encryption_sync(p25::encode_encryption_sync(es), es_got));
    CHECK(es_got.encrypted() && es_got.algorithm == 0x84 && es_got.key == 0x1234);
    CHECK(p25::decode_encryption_sync(p25::encode_encryption_sync({}), es_got) && !es_got.encrypted());
}

TEST(p25_voice_frames_interleave_as_sent)
{
    // As DSD's interleave tables have it: dibit i's high bit is bit 2i.
    const auto where = [](size_t k) {
        uint8_t bits[144] = {};
        bits[k]           = 1;
        p25::ImbeFrame f;
        p25::imbe_deinterleave(bits, f);
        for (size_t c = 0; c < 8; ++c)
            for (size_t b = 0; b < 23; ++b)
                if (f.c[c][b])
                    return static_cast<int>(100 * c + b);
        return -1;
    };
    CHECK_EQ(where(0), 22);     // C0 bit 22
    CHECK_EQ(where(1), 121);    // C1 bit 21
    CHECK_EQ(where(5), 501);    // C5 bit 1
    CHECK_EQ(where(17), 614);   // C6 bit 14
    CHECK_EQ(where(143), 502);  // C5 bit 2
    p25::ImbeFrame f;
    f.c[7][6] = 1;
    f.c[3][0] = 1;
    uint8_t bits[144];
    p25::imbe_interleave(f, bits);
    p25::ImbeFrame back;
    p25::imbe_deinterleave(bits, back);
    CHECK(back == f);
}

TEST(p25_decoder_follows_a_call)
{
    P25Transmitter              tx(0x293);
    std::vector<p25::ImbeFrame> sent;
    tx.start_call(group_call(100, 2001), 10, counting_voice(sent, 1));
    P25Decoder decoder(48000.0);
    bool       saw_ids = false, saw_call = false;
    const auto voice = decode(decoder, p25_frequency(tx, 2.6, 48000.0, 0.2, 250.0, 120.0), 48000.0, [&](const P25Status& s) {
        saw_call |= s.call.active;
        saw_ids |= s.call.active && s.call.ids_known && s.call.destination == 100 && s.call.source == 2001;
    });
    const P25Status& s = decoder.status();
    CHECK(saw_call);
    CHECK(saw_ids);
    CHECK_EQ(s.nac, 0x293);
    CHECK(!s.call.active);  // ended by its terminator
    CHECK(s.call.group);
    CHECK_EQ(sent.size(), 90u);
    CHECK_EQ(voice.size(), 90u);
    CHECK_EQ(matching(sent, voice), 90u);
    CHECK(!voice.empty() && voice.front().first);
    std::printf("    level %.0f Hz, %llu frames; %s\n", s.level_hz, static_cast<unsigned long long>(s.frames),
                describe_p25(s).c_str());
}

TEST(p25_decoder_joins_late_and_mutes_encryption)
{
    // Tuning in after the call's first LDU: the link control of the next
    // LDU1 names it.
    {
        P25Transmitter              tx;
        std::vector<p25::ImbeFrame> sent;
        tx.start_call(group_call(7, 31), 12, counting_voice(sent, 2));
        int skipped = 0, symbol = 0;
        while (skipped < 864 + 400 && tx.next_symbol(symbol))
            ++skipped;
        P25Decoder decoder(48000.0);
        bool       saw_ids = false;
        const auto voice = decode(decoder, p25_frequency(tx, 2.4, 48000.0, 0.1), 48000.0, [&](const P25Status& s) {
            saw_ids |= s.call.ids_known && s.call.destination == 7 && s.call.source == 31;
        });
        CHECK(saw_ids);
        CHECK(voice.size() >= 80u);
        CHECK(matching(sent, voice) + 2 >= voice.size());
    }
    // An encrypted call: flagged, and nothing passed on.
    {
        P25Transmitter              tx;
        std::vector<p25::ImbeFrame> sent;
        p25::LinkControl            lc = group_call(5, 6);
        lc.options                     = p25::LinkControl::kEncrypted;
        tx.start_call(lc, 6, counting_voice(sent, 3));
        P25Decoder decoder(48000.0);
        bool       saw = false;
        const auto voice = decode(decoder, p25_frequency(tx, 1.6, 48000.0), 48000.0, [&](const P25Status& s) {
            saw |= s.call.active && s.call.encrypted;
        });
        CHECK(saw);
        CHECK(voice.empty());
    }
}

TEST(p25_decoder_keeps_time_and_ignores_what_is_not_p25)
{
    for (const double trim : {0.004, -0.004}) {
        P25Transmitter              tx;
        std::vector<p25::ImbeFrame> sent;
        tx.start_call(group_call(1, 1), 10, counting_voice(sent, 4));
        const double rate = 48000.0 * (1.0 + trim);
        P25Decoder   decoder(48000.0);
        const auto   voice = decode(decoder, p25_frequency(tx, 2.6, rate * (1.0 + 100e-6), 0.2, -300.0, 120.0), rate);
        CHECK_EQ(matching(sent, voice), 90u);
    }
    std::mt19937                     rng(3);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<float>               noise(48000 * 2);
    for (float& v : noise)
        v = static_cast<float>(3000.0 * gauss(rng));
    P25Decoder decoder(48000.0);
    bool       present = false;
    CHECK(decode(decoder, noise, 48000.0, [&](const P25Status& s) { present |= s.present; }).empty());
    CHECK(!present);
}

TEST(p25_decoder_works_through_the_radio)
{
    // At 14 dB SNR and 500 Hz off tune, received by FmDemodulator.
    constexpr double            kIqRate = 240e3, kOffset = -40e3;
    P25Transmitter              tx(0x5A1);
    std::vector<p25::ImbeFrame> sent;
    tx.start_call(group_call(4242, 99), 8, counting_voice(sent, 5));
    std::vector<bool>        air;
    const std::vector<float> freq = p25_frequency(tx, 2.2, kIqRate, 0.2, 500.0, 0.0, 1, &air);

    std::mt19937                     rng(17);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const double                     amplitude = 2048.0 * 0.1, bandwidth = 12.5e3;
    const double sigma = amplitude * std::sqrt(std::pow(10.0, -14.0 / 10.0) * (kIqRate / bandwidth) / 2.0);
    std::vector<int16_t> iq(2 * freq.size());
    double               phase = 0.0;
    for (size_t i = 0; i < freq.size(); ++i) {
        const double a = air[i] ? amplitude : 0.0;  // no carrier off the air
        phase          = std::fmod(phase + 2.0 * kPi * (kOffset + (air[i] ? freq[i] : 0.0)) / kIqRate, 2.0 * kPi);
        iq[2 * i]      = static_cast<int16_t>(std::lround(a * std::cos(phase) + sigma * gauss(rng)));
        iq[2 * i + 1]  = static_cast<int16_t>(std::lround(a * std::sin(phase) + sigma * gauss(rng)));
    }
    FmDemodulator demod(kIqRate, 48000.0);
    demod.set_channel(kOffset, bandwidth);
    P25Decoder              decoder(48000.0);
    FmDemodulator::Analysis analysis;
    std::vector<float>      audio;
    std::vector<P25Voice>   voice;
    bool                    saw_ids = false;
    for (size_t done = 0; done < freq.size(); done += 8192) {
        demod.process(iq.data() + 2 * done, std::min<size_t>(8192, freq.size() - done), audio, &analysis);
        decoder.process(analysis.frequency.data(), analysis.frequency.size(), 48000.0, voice);
        saw_ids |= decoder.status().call.ids_known && decoder.status().call.destination == 4242;
        analysis.frequency.clear();
        analysis.power.clear();
        audio.clear();
    }
    CHECK_EQ(decoder.status().nac, 0x5A1);
    CHECK(saw_ids);
    const size_t good = matching(sent, voice);
    std::printf("    %zu of %zu frames sent heard intact\n", good, sent.size());
    CHECK(good >= sent.size() * 95 / 100);
}

TEST(p25_voice_is_heard)
{
    if (!ImbeDecoder::available()) {
        std::printf("    (built without a vocoder: skipped)\n");
        return;
    }
    // A voice at 160 Hz through the IMBE vocoder, P25 and back.
    ImbeEncoder encoder;
    size_t      t = 0;
    const auto  voice = [&] {
        float frame[ImbeEncoder::kFrameSamples];
        for (float& v : frame) {
            const double s = static_cast<double>(t++) / 8000.0;
            double       x = 0.0;
            for (int h = 1; h <= 20; ++h)
                x += std::exp(-std::pow((h * 160.0 - 700.0) / 400.0, 2.0)) * std::sin(2.0 * kPi * h * 160.0 * s) / h;
            v = static_cast<float>(0.3 * x);
        }
        return encoder.encode(frame);
    };
    P25Transmitter tx;
    tx.start_call(group_call(1, 1), 10, voice);
    P25Decoder decoder(48000.0);
    const auto frames = decode(decoder, p25_frequency(tx, 2.6, 48000.0, 0.2, 0.0, 120.0), 48000.0);
    ImbeDecoder        vocoder;
    std::vector<float> heard;
    for (const P25Voice& v : frames) {
        if (v.first)
            vocoder.reset();
        float pcm[ImbeDecoder::kFrameSamples];
        vocoder.decode(v.frame, pcm);
        heard.insert(heard.end(), pcm, pcm + ImbeDecoder::kFrameSamples);
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
    CHECK(std::fabs(8000.0 / lag - 160.0) < 8.0);
    CHECK(rms > 0.02);
}

TEST(listener_decodes_the_simulators_p25)
{
    // The simulator's P25 radio makes a call four seconds in.  With P25 only
    // it is heard (if there is a vocoder); with DMR only, not.
    Options options;
    options.source.kind        = "sim";
    options.source.sample_rate = 2.4e6;
    options.source.frequency   = 150e6;
    options.source.gain        = SimSource::kReferenceGainDb;
    Pipeline pipeline(options);
    pipeline.start();
    Listener& listener = pipeline.listener();
    listener.set_volume(1.0f);    // the plot shows the audio as played (no sound card here: silent all the same)
    listener.set_dmr_only(true);  // to start with: P25 is not to be heard
    listener.set_enabled(true);
    listener.tune(150e6 + SimSource::digital_stations(2.4e6)[2].offset_hz, 0.0, 12.5e3);

    const auto loudest_played = [&] {
        std::vector<float> played(9600);
        played.resize(listener.scope().latest(played.data(), played.size()));
        float peak = 0.0f;
        for (float v : played)
            peak = std::max(peak, std::fabs(v));
        return peak;
    };
    bool         called = false, silent_for_dmr_only = true;
    float        loudest = 0.0f;
    ListenStatus s;
    bool         switched = false;
    const auto   start    = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(6600)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        s = listener.status();
        if (!s.p25.present || !s.p25.call.active || !s.p25.call.ids_known)
            continue;
        called = s.p25.call.group && s.p25.call.destination == SimSource::kP25Call.talkgroup &&
                 s.p25.call.source == SimSource::kP25Call.source;
        if (!switched) {
            // Half a second into the call with DMR only: nothing played.
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            silent_for_dmr_only = loudest_played() == 0.0f;
            listener.set_dmr_only(false);
            listener.set_p25_only(true);
            switched = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(700));
            continue;
        }
        loudest = std::max(loudest, loudest_played());
    }
    CHECK(called);
    CHECK_EQ(s.p25.nac, static_cast<int>(SimSource::kP25Nac));
    CHECK(silent_for_dmr_only);
    CHECK(switched);
    if (ImbeDecoder::available())
        CHECK(loudest > 0.02f);
    std::printf("    %s; heard at %.3f\n", describe_p25(s.p25).c_str(), static_cast<double>(loudest));

    listener.set_enabled(false);
    pipeline.stop();
}

TEST(heard_channels_remember_p25_voice)
{
    HeardChannels heard;
    P25Status     p25;
    p25.present          = true;
    p25.nac              = 0x293;
    p25.call.active      = true;
    p25.call.voice       = true;
    p25.call.ids_known   = true;
    p25.call.destination = 100;
    p25.call.source      = 2001;
    heard.update(155.1e6, DmrStatus{}, p25, 1.0);
    heard.update(155.1e6, DmrStatus{}, p25, 1.5);
    p25.call.active = false;
    heard.update(155.1e6, DmrStatus{}, p25, 2.0);
    p25.call.active = true;
    heard.update(155.1e6, DmrStatus{}, p25, 3.0);
    CHECK_EQ(heard.channels().size(), size_t{1});
    const HeardChannel& c = heard.channels()[0];
    CHECK(c.system == "P25");
    CHECK_EQ(c.nac, 0x293);
    CHECK_EQ(c.calls, 2);
    CHECK(c.talkgroup == "TG 100");
    CHECK(c.last_call == "Group call to TG 100 from 2001");
}

TEST(p25_trunking_blocks_survive_errors)
{
    std::mt19937 rng(21);
    for (int trial = 0; trial < 40; ++trial) {
        p25::Tsbk block{};
        for (size_t i = 0; i < 10; ++i)
            block[i] = static_cast<uint8_t>(rng());
        std::array<uint8_t, 196> bits = p25::encode_tsbk(block);
        p25::Tsbk                got{};
        CHECK_EQ(p25::decode_tsbk(bits.data(), got), 0);
        // A few scattered errors: the trellis code mends them.
        for (int e = 0; e < 4; ++e)
            bits[(rng() % 49) * 4 + static_cast<size_t>(e % 4)] ^= 1u;
        CHECK(p25::decode_tsbk(bits.data(), got) >= 0);
        CHECK(std::equal(block.begin(), block.begin() + 10, got.begin()));
    }
    // A wrong block is caught by its CRC.
    std::array<uint8_t, 196> bits = p25::encode_tsbk(tsbk::network_status(0x12345, 0x1F3, 0x100A));
    p25::Tsbk                got{};
    for (size_t i = 0; i < 196; i += 9)
        bits[i] ^= 1u;
    CHECK_EQ(p25::decode_tsbk(bits.data(), got), -1);
}

TEST(p25_control_messages_are_understood)
{
    P25Site               site;
    std::vector<P25Grant> grants;
    CHECK(apply_tsbk(tsbk::network_status(0xBEE01, 0x2A3, 0), site, grants, 0.0));
    CHECK(apply_tsbk(tsbk::channel_plan(1, 155.1375e6, 12.5e3, -4e6), site, grants, 0.0));
    CHECK(apply_tsbk(tsbk::channel_plan_vhf_uhf(2, 450.0e6, 6.25e3, 5e6), site, grants, 0.0));
    CHECK(apply_tsbk(tsbk::rfss_status(0x2A3, 3, 7, 0x1028), site, grants, 0.0));
    CHECK_EQ(site.wacn, 0xBEE01);
    CHECK_EQ(site.system, 0x2A3);
    CHECK_EQ(site.rfss, 3);
    CHECK_EQ(site.site, 7);
    CHECK_NEAR(site.control_hz, 155.1375e6 + 40 * 12.5e3, 1e-3);
    CHECK_NEAR(site.plans[1].tx_offset_hz, -4e6, 1e-3);
    CHECK_NEAR(site.plans[2].tx_offset_hz, 5e6, 1e-3);
    CHECK(!site.phase2());

    // Grants: the frequency from the plan; updates keep the source.
    CHECK(apply_tsbk(tsbk::group_grant(0x1005, 100, 2001, 0x40), site, grants, 1.0));
    CHECK(apply_tsbk(tsbk::group_grant_update(0x1005, 100, 0x2010, 200), site, grants, 2.0));
    CHECK(apply_tsbk(tsbk::unit_grant(0x1007, 1234567, 7654321), site, grants, 3.0));
    CHECK_EQ(grants.size(), size_t{3});
    CHECK(!grants[0].group && grants[0].destination == 1234567u && grants[0].source == 7654321u);
    const auto tg100 = std::find_if(grants.begin(), grants.end(), [](const P25Grant& g) { return g.destination == 100; });
    CHECK(tg100 != grants.end() && tg100->source == 2001u && tg100->encrypted && tg100->slot == -1);
    CHECK(tg100 != grants.end() && std::fabs(tg100->frequency - (155.1375e6 + 5 * 12.5e3)) < 1e-3);
    const auto tg200 = std::find_if(grants.begin(), grants.end(), [](const P25Grant& g) { return g.destination == 200; });
    CHECK(tg200 != grants.end() && std::fabs(tg200->frequency - (450.0e6 + 16 * 6.25e3)) < 1e-3);

    // A TDMA plan: two slots per carrier, so voice in Phase 2.
    CHECK(apply_tsbk(tsbk::channel_plan_tdma(3, 3, 151.8375e6, 12.5e3, 7e6), site, grants, 4.0));
    CHECK(site.phase2());
    CHECK(apply_tsbk(tsbk::group_grant(0x3009, 300, 5), site, grants, 5.0));
    CHECK_EQ(grants[0].slot, 1);
    CHECK_NEAR(grants[0].frequency, 151.8375e6 + 4 * 12.5e3, 1e-3);
    CHECK(p25_grant_destination(grants[0]) == "TG 300");
}

TEST(p25_decoder_reads_a_control_channel)
{
    // Control channels with one, two and three blocks to a TSDU, the second
    // through the radio at 14 dB SNR.
    for (const int per_tsdu : {3, 1, 2}) {
        P25Transmitter tx(0x663);
        int            n = 0;
        tx.control([&n] {
            switch (n++ % 4) {
            case 0: return tsbk::network_status(0xBEE01, 0x2A3, 0x1028);
            case 1: return tsbk::channel_plan(1, 155.1375e6, 12.5e3, -4e6);
            case 2: return tsbk::rfss_status(0x2A3, 1, 4, 0x1028);
            default: return tsbk::group_grant(0x1005, 100, 2001);
            }
        }, per_tsdu);
        P25Decoder decoder(48000.0);
        bool       control = false;
        decode(decoder, p25_frequency(tx, 2.0, 48000.0, 0.1, 200.0, 120.0), 48000.0,
               [&](const P25Status& s) { control |= s.control; });
        const P25Status& s = decoder.status();
        CHECK(control);
        CHECK_EQ(s.nac, 0x663);
        CHECK_EQ(s.site.wacn, 0xBEE01);
        CHECK_EQ(s.site.system, 0x2A3);
        CHECK_EQ(s.site.site, 4);
        CHECK(s.site.plans[1].known);
        CHECK(!s.grants.empty() && s.grants[0].destination == 100u && s.grants[0].source == 2001u &&
              std::fabs(s.grants[0].frequency - 155.2e6) < 1.0);
        // Most blocks get through: about 2 s worth.
        const uint64_t expected = static_cast<uint64_t>(1.8 * 4800.0 / (56.0 + 98.0 * per_tsdu + 4.0) * per_tsdu);
        std::printf("    %d to a TSDU: %llu blocks (about %llu sent), %s\n", per_tsdu,
                    static_cast<unsigned long long>(s.tsbks), static_cast<unsigned long long>(expected),
                    describe_p25(s).c_str());
        CHECK(s.tsbks * 10 >= expected * 8);
    }
}

TEST(listener_reads_the_simulators_p25_control_channel)
{
    // The control channel names its system, and while the P25 radio talks
    // (four seconds in) grants its call on the radio's frequency.
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
    const std::vector<SimSource::DigitalStation> stations = SimSource::digital_stations(2.4e6);
    listener.tune(150e6 + stations[3].offset_hz, 0.0, 12.5e3);

    bool         granted = false;
    ListenStatus s;
    const auto   start = std::chrono::steady_clock::now();
    while (!granted && std::chrono::steady_clock::now() - start < std::chrono::milliseconds(6000)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        s = listener.status();
        for (const P25Grant& g : s.p25.grants)
            granted |= g.group && g.destination == SimSource::kP25Call.talkgroup &&
                       g.source == SimSource::kP25Call.source && std::fabs(g.frequency - (150e6 + stations[2].offset_hz)) < 1.0 &&
                       g.slot < 0;
    }
    CHECK(s.p25.present);
    CHECK(s.p25.control);
    CHECK_EQ(s.p25.site.system, static_cast<int>(SimSource::kP25System));
    CHECK_EQ(s.p25.site.wacn, static_cast<int>(SimSource::kP25Wacn));
    CHECK(s.p25.site.phase1() && s.p25.site.phase2());  // the P25 radio's channel, and a Phase 2 one
    CHECK(std::fabs(s.p25.site.control_hz - (150e6 + stations[3].offset_hz)) < 1.0);
    CHECK(granted);
    std::printf("    %llu trunking blocks; %zu grant(s)\n", static_cast<unsigned long long>(s.p25.tsbks), s.p25.grants.size());

    listener.set_enabled(false);
    pipeline.stop();
}
