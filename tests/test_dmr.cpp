#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <thread>
#include <vector>

#include "app/pipeline.hpp"
#include "audio/heard_channels.hpp"
#include "audio/listener.hpp"
#include "audio/voice_playout.hpp"
#include "dsp/ambe_vocoder.hpp"
#include "dsp/dmr_coding.hpp"
#include "dsp/dmr_decoder.hpp"
#include "dsp/dmr_transmitter.hpp"
#include "dsp/fm_demod.hpp"
#include "radio/sim_source.hpp"
#include "test_framework.hpp"

using namespace spectro;

namespace {

constexpr double kPi = std::numbers::pi;

// Flips `count` distinct random bits among the lowest `width`.
uint32_t flip_bits(uint32_t word, int width, int count, std::mt19937& rng)
{
    std::vector<int> positions(static_cast<size_t>(width));
    for (int i = 0; i < width; ++i)
        positions[static_cast<size_t>(i)] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    for (int i = 0; i < count; ++i)
        word ^= 1u << positions[static_cast<size_t>(i)];
    return word;
}

// Root raised cosine, alpha 0.2, t in symbols: what DMR transmitters send.
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

// The frequency a DMR transmitter sends, `rate` samples per second, its
// symbols drawn from `tx` (each 648 Hz per unit), starting `delay` symbols in.
std::vector<float> dmr_frequency(DmrTransmitter& tx, double seconds, double rate, double offset_hz = 0.0,
                                 double noise_hz = 0.0, unsigned seed = 1, double delay = 0.37)
{
    const auto       symbols = static_cast<size_t>(seconds * DmrDecoder::kSymbolRate) + 32;
    std::vector<int> s(symbols);
    for (int& v : s)
        v = tx.next_symbol();
    // The pulse, tabulated over +-8 symbols.
    constexpr int       kReach = 8, kSteps = 1000;
    std::vector<double> pulse(2 * kReach * kSteps + 1);
    for (size_t i = 0; i < pulse.size(); ++i)
        pulse[i] = rrc_pulse(static_cast<double>(i) / kSteps - kReach);

    std::mt19937                     rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const auto                       n = static_cast<size_t>(seconds * rate);
    std::vector<float>               f(n);
    for (size_t i = 0; i < n; ++i) {
        const double position = static_cast<double>(i) / rate * DmrDecoder::kSymbolRate - delay;
        const long   k0       = static_cast<long>(std::floor(position));
        double       v        = 0.0;
        for (long k = k0 - kReach + 1; k <= k0 + kReach; ++k) {
            if (k < 0 || k >= static_cast<long>(symbols))
                continue;
            const double t = position - static_cast<double>(k);
            v += s[static_cast<size_t>(k)] * pulse[static_cast<size_t>(std::lround((t + kReach) * kSteps))];
        }
        f[i] = static_cast<float>(648.0 * v + offset_hz + noise_hz * gauss(rng));
    }
    return f;
}

// Feeds `frequency` to a decoder in blocks, as the listener would.
std::vector<DmrVoice> decode(DmrDecoder& decoder, const std::vector<float>& frequency, double rate,
                             const std::function<void(const DmrStatus&)>& watch = nullptr)
{
    std::vector<DmrVoice> voice;
    for (size_t done = 0; done < frequency.size(); done += 1000) {
        decoder.process(frequency.data() + done, std::min<size_t>(1000, frequency.size() - done), rate, voice);
        if (watch)
            watch(decoder.status());
    }
    return voice;
}

// A voice source that numbers its frames, and remembers them.
DmrTransmitter::VoiceSource counting_voice(std::vector<dmr::AmbeFrame>& sent, unsigned seed)
{
    auto rng = std::make_shared<std::mt19937>(seed);
    return [&sent, rng] {
        dmr::AmbeFrame f;
        for (auto& row : f.c)
            for (char& b : row)
                b = static_cast<char>((*rng)() & 1u);
        // Unused bits stay zero, as a real frame has them.
        for (size_t i = 23; i < 24; ++i)
            f.c[1][i] = 0;
        for (size_t i = 11; i < 24; ++i)
            f.c[2][i] = 0;
        for (size_t i = 14; i < 24; ++i)
            f.c[3][i] = 0;
        sent.push_back(f);
        return f;
    };
}

// How many of `got` match `sent` in order, from wherever the first one is found.
size_t matching_frames(const std::vector<dmr::AmbeFrame>& sent, const std::vector<DmrVoice>& got)
{
    if (got.empty())
        return 0;
    const auto first = std::find(sent.begin(), sent.end(), got.front().frame);
    if (first == sent.end())
        return 0;
    size_t matches = 0;
    for (size_t i = 0; i < got.size() && first + static_cast<long>(i) < sent.end(); ++i)
        matches += got[i].frame == *(first + static_cast<long>(i)) ? 1 : 0;
    return matches;
}

dmr::LinkControl group_call(uint32_t talkgroup, uint32_t source)
{
    dmr::LinkControl lc;
    lc.flco        = dmr::LinkControl::kGroupVoice;
    lc.destination = talkgroup;
    lc.source      = source;
    return lc;
}

}  // namespace

TEST(dmr_codes_correct_what_they_can)
{
    std::mt19937 rng(3);
    // Golay (20,8): distance 8, three errors corrected, four detected.
    int min_weight = 99;
    for (unsigned v = 1; v < 256; ++v)
        min_weight = std::min(min_weight, std::popcount(dmr::golay2087_encode(static_cast<uint8_t>(v))));
    CHECK_EQ(min_weight, 8);
    bool golay_ok = true;
    for (unsigned v = 0; v < 256; ++v) {
        const uint32_t cw   = dmr::golay2087_encode(static_cast<uint8_t>(v));
        uint8_t        info = 0;
        golay_ok &= dmr::golay2087_decode(flip_bits(cw, 20, 3, rng), info) == 3 && info == v;
        golay_ok &= dmr::golay2087_decode(flip_bits(cw, 20, 4, rng), info) == -1;
    }
    CHECK(golay_ok);

    // QR (16,7): distance 6, two errors corrected.
    bool qr_ok = true;
    for (unsigned v = 0; v < 128; ++v) {
        uint8_t info = 0;
        qr_ok &= dmr::qr1676_decode(static_cast<uint16_t>(flip_bits(dmr::qr1676_encode(static_cast<uint8_t>(v)), 16, 2, rng)),
                                    info) == 2 &&
                 info == v;
    }
    CHECK(qr_ok);

    // Hamming (7,4): one error.
    bool tact_ok = true;
    for (unsigned v = 0; v < 16; ++v) {
        uint8_t info = 0;
        tact_ok &= dmr::hamming743_decode(static_cast<uint8_t>(flip_bits(dmr::hamming743_encode(static_cast<uint8_t>(v)), 7, 1, rng)),
                                          info) == 1 &&
                   info == v;
    }
    CHECK(tact_ok);

    // BPTC (196,96): a few errors anywhere.
    for (int trial = 0; trial < 50; ++trial) {
        std::array<uint8_t, 96> data{};
        for (uint8_t& b : data)
            b = static_cast<uint8_t>(rng() & 1u);
        std::array<uint8_t, 196> sent = dmr::bptc19696_encode(data);
        std::array<uint8_t, 96>  got{};
        CHECK_EQ(dmr::bptc19696_decode(sent, got), 0);
        for (int e = 0; e < 3; ++e)
            sent[rng() % 196] ^= 1u;
        CHECK(dmr::bptc19696_decode(sent, got) >= 0 && got == data);
    }

    // Reed-Solomon (12,9) catches a changed byte.
    uint8_t cw[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    dmr::rs129_parity(cw, cw + 9);
    CHECK(dmr::rs129_check(cw));
    cw[4] ^= 0x10;
    CHECK(!dmr::rs129_check(cw));
}

TEST(dmr_link_control_round_trips)
{
    dmr::LinkControl lc = group_call(91, 3100123);
    lc.options          = dmr::LinkControl::kPrivacy;
    dmr::LinkControl got;

    const auto header = dmr::encode_full_lc(lc, dmr::kVoiceLcHeaderMask);
    CHECK(dmr::decode_full_lc(header, dmr::kVoiceLcHeaderMask, got) && got == lc && got.encrypted());
    CHECK(!dmr::decode_full_lc(header, dmr::kTerminatorMask, got));  // the masks keep them apart

    std::array<uint8_t, 128> embedded = dmr::encode_embedded_lc(lc);
    got                               = {};
    CHECK(dmr::decode_embedded_lc(embedded, got) && got == lc);
    embedded[5] ^= 1u;  // one error: the row's Hamming code mends it
    got = {};
    CHECK(dmr::decode_embedded_lc(embedded, got) && got == lc);
    embedded[6] ^= 1u;
    embedded[7] ^= 1u;
    CHECK(!dmr::decode_embedded_lc(embedded, got) || got == lc);
}

TEST(dmr_voice_frames_interleave_as_sent)
{
    // The interleave puts bit k at C(row) bit (column) as DSD's tables have it.
    uint8_t bits[72] = {};
    const auto where = [&](size_t k) {
        std::fill(bits, bits + 72, uint8_t{0});
        bits[k] = 1;
        dmr::AmbeFrame f;
        dmr::ambe_deinterleave(bits, f);
        for (size_t r = 0; r < 4; ++r)
            for (size_t c = 0; c < 24; ++c)
                if (f.c[r][c])
                    return static_cast<int>(100 * r + c);
        return -1;
    };
    CHECK_EQ(where(0), 23);    // dibit 0, high bit: C0 bit 23
    CHECK_EQ(where(1), 5);     // dibit 0, low bit: C0 bit 5
    CHECK_EQ(where(2), 110);   // dibit 1: C1 bit 10
    CHECK_EQ(where(19), 313);  // dibit 9, low bit: C3 bit 13
    CHECK_EQ(where(46), 210);  // dibit 23, high bit: C2 bit 10
    CHECK_EQ(where(71), 300);

    dmr::AmbeFrame frame = dmr::ambe_silence();
    uint8_t        out[72];
    dmr::ambe_interleave(frame, out);
    dmr::AmbeFrame back;
    dmr::ambe_deinterleave(out, back);
    CHECK(back == frame);
}

TEST(dmr_decoder_follows_a_call)
{
    DmrTransmitter              tx(5);
    std::vector<dmr::AmbeFrame> sent;
    tx.start_call(0, group_call(91, 3100123), 8, counting_voice(sent, 1));

    DmrDecoder decoder(48000.0);
    bool       saw_call = false, saw_ids = false;
    DmrStatus  during;
    const auto voice = decode(decoder, dmr_frequency(tx, 4.0, 48000.0, 350.0, 150.0), 48000.0, [&](const DmrStatus& s) {
        if (s.slot[0].active && !saw_call) {
            saw_call = true;
            during   = s;
        }
        saw_ids |= s.slot[0].active && s.slot[0].ids_known && s.slot[0].destination == 91 && s.slot[0].source == 3100123;
    });
    const DmrStatus& s = decoder.status();
    CHECK(s.present);
    CHECK(s.repeater);
    CHECK_EQ(s.color_code, 5);
    CHECK(saw_call);
    CHECK(saw_ids);
    CHECK(during.slot[0].group);
    CHECK(!during.slot[1].active);
    CHECK(!s.slot[0].active);  // it has ended
    CHECK_EQ(s.slot[0].destination, 91u);
    // 8 superframes of 18 frames, all of them heard, bit for bit.
    CHECK_EQ(sent.size(), 144u);
    CHECK_EQ(voice.size(), 144u);
    CHECK_EQ(matching_frames(sent, voice), 144u);
    CHECK(!voice.empty() && voice.front().first && voice.front().slot == 0);
    std::printf("    level %.0f Hz, %llu frames\n", s.level_hz, static_cast<unsigned long long>(s.frames));
}

TEST(dmr_decoder_joins_a_call_late)
{
    // Tuning in during a call: no header, the IDs come from the embedded link control.
    DmrTransmitter              tx(1);
    std::vector<dmr::AmbeFrame> sent;
    tx.start_call(1, group_call(2, 1001), 10, counting_voice(sent, 2));
    for (int i = 0; i < 40; ++i) {  // 1.2 s in: headers and the first superframes have gone by
        std::array<int8_t, dmr::kFrameSymbols> skip{};
        tx.next_frame(skip);
    }
    DmrDecoder decoder(48000.0);
    bool       saw_ids = false;
    const auto voice   = decode(decoder, dmr_frequency(tx, 2.0, 48000.0), 48000.0, [&](const DmrStatus& s) {
        saw_ids |= s.slot[1].ids_known && s.slot[1].destination == 2 && s.slot[1].source == 1001;
    });
    CHECK(saw_ids);
    CHECK(voice.size() >= 60u);
    CHECK(matching_frames(sent, voice) + 3 >= voice.size());
    CHECK(!voice.empty() && voice.front().slot == 1);
}

TEST(dmr_decoder_mutes_encrypted_calls)
{
    DmrTransmitter              tx(1);
    std::vector<dmr::AmbeFrame> sent;
    dmr::LinkControl            lc = group_call(5, 77);
    lc.options                     = dmr::LinkControl::kPrivacy;
    tx.start_call(0, lc, 5, counting_voice(sent, 3));
    DmrDecoder decoder(48000.0);
    bool       saw_encrypted = false;
    const auto voice = decode(decoder, dmr_frequency(tx, 3.0, 48000.0), 48000.0, [&](const DmrStatus& s) {
        saw_encrypted |= s.slot[0].active && s.slot[0].encrypted;
    });
    CHECK(saw_encrypted);
    CHECK(voice.empty());
}

TEST(dmr_decoder_keeps_time_with_the_transmitter)
{
    // The audio rate is trimmed by up to half a percent to suit the sound
    // card; the decoder is told.  And the transmitter's clock may be off by
    // 100 ppm, which it is not told.
    for (const double trim : {0.004, -0.004}) {
        DmrTransmitter              tx(1);
        std::vector<dmr::AmbeFrame> sent;
        tx.start_call(0, group_call(1, 1), 8, counting_voice(sent, 4));
        const double rate = 48000.0 * (1.0 + trim);
        DmrDecoder   decoder(48000.0);
        const auto   voice = decode(decoder, dmr_frequency(tx, 4.0, rate * (1.0 + 100e-6), -200.0, 150.0), rate);
        CHECK_EQ(matching_frames(sent, voice), 144u);
    }
}

TEST(dmr_decoder_ignores_what_is_not_dmr)
{
    std::mt19937                     rng(9);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<float>               f(48000 * 3);
    for (size_t i = 0; i < f.size(); ++i) {
        const double t = static_cast<double>(i) / 48000.0;
        // Speech-like audio, then noise.
        f[i] = static_cast<float>(i < f.size() / 2 ? 1500.0 * std::sin(2.0 * kPi * 400.0 * t) * std::sin(2.0 * kPi * 3.0 * t)
                                                   : 3000.0 * gauss(rng));
    }
    DmrDecoder decoder(48000.0);
    bool       present = false;
    const auto voice   = decode(decoder, f, 48000.0, [&](const DmrStatus& s) { present |= s.present; });
    CHECK(!present);
    CHECK(voice.empty());
}

TEST(dmr_decoder_works_through_the_radio)
{
    // DMR on air at 15 dB SNR and 600 Hz off tune, received by FmDemodulator.
    constexpr double kIqRate = 240e3, kOffset = 30e3;
    DmrTransmitter   tx(3);
    std::vector<dmr::AmbeFrame> sent;
    tx.start_call(0, group_call(9, 2345), 6, counting_voice(sent, 5));
    const std::vector<float> freq = dmr_frequency(tx, 3.0, kIqRate, 600.0);

    std::mt19937                     rng(11);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const double                     amplitude = 2048.0 * 0.1, bandwidth = 12.5e3;
    const double sigma = amplitude * std::sqrt(std::pow(10.0, -15.0 / 10.0) * (kIqRate / bandwidth) / 2.0);
    std::vector<int16_t> iq(2 * freq.size());
    double               phase = 0.0;
    for (size_t i = 0; i < freq.size(); ++i) {
        phase         = std::fmod(phase + 2.0 * kPi * (kOffset + freq[i]) / kIqRate, 2.0 * kPi);
        iq[2 * i]     = static_cast<int16_t>(std::lround(amplitude * std::cos(phase) + sigma * gauss(rng)));
        iq[2 * i + 1] = static_cast<int16_t>(std::lround(amplitude * std::sin(phase) + sigma * gauss(rng)));
    }

    FmDemodulator demod(kIqRate, 48000.0);
    demod.set_channel(kOffset, bandwidth);
    DmrDecoder              decoder(48000.0);
    FmDemodulator::Analysis analysis;
    std::vector<float>      audio;
    std::vector<DmrVoice>   voice;
    for (size_t done = 0; done < freq.size(); done += 8192) {
        demod.process(iq.data() + 2 * done, std::min<size_t>(8192, freq.size() - done), audio, &analysis);
        decoder.process(analysis.frequency.data(), analysis.frequency.size(), 48000.0, voice);
        analysis.frequency.clear();
        analysis.power.clear();
        audio.clear();
    }
    CHECK_EQ(decoder.status().color_code, 3);
    CHECK_EQ(decoder.status().slot[0].destination, 9u);
    const size_t good = matching_frames(sent, voice);
    std::printf("    %zu of %zu frames sent heard intact\n", good, sent.size());
    CHECK(good >= sent.size() * 95 / 100);
}

TEST(voice_playout_streams_steadily)
{
    // Speech arrives three frames at a time every 60 ms; it comes out at
    // 48 kHz (trimmed) in blocks of an awkward size, without a gap.
    VoicePlayout       playout;
    std::vector<float> out;
    std::vector<float> chunk(480);
    double             t_in = 0.0;
    const double       rate = 48000.0 * 1.003;
    size_t             produced_in = 0, gaps = 0;
    bool               started = false;
    std::vector<float> block(731);
    for (int step = 0; step < 400; ++step) {
        // 731 output samples cover 731 / rate seconds of input time.
        const double until = (static_cast<double>(step) + 1.0) * 731.0 / rate;
        while (t_in < until) {
            for (size_t i = 0; i < chunk.size(); ++i)
                chunk[i] = static_cast<float>(0.5 * std::sin(2.0 * kPi * 1000.0 * static_cast<double>(produced_in + i) / 8000.0));
            produced_in += chunk.size();
            playout.push(chunk.data(), chunk.size());
            t_in += 0.06;
        }
        playout.pull(block.data(), block.size(), rate);
        if (playout.playing())
            started = true;
        else if (started)
            ++gaps;
        out.insert(out.end(), block.begin(), block.end());
    }
    CHECK(started);
    CHECK_EQ(gaps, 0u);
    // A 1 kHz tone, at the output rate.
    const size_t from = out.size() / 2, n = 4800;
    double       re = 0.0, im = 0.0, power = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double w = 2.0 * kPi * 1000.0 * static_cast<double>(i) / rate;
        re += out[from + i] * std::cos(w);
        im += out[from + i] * std::sin(w);
        power += out[from + i] * out[from + i];
    }
    const double tone = 2.0 * (re * re + im * im) / static_cast<double>(n) / power;
    CHECK(tone > 0.95);
    CHECK(std::fabs(std::sqrt(power / n) - 0.5 / std::sqrt(2.0)) < 0.02);
}

TEST(dmr_voice_is_heard)
{
    if (!AmbeDecoder::available()) {
        std::printf("    (built without a vocoder: skipped)\n");
        return;
    }
    // A voice at 140 Hz through the vocoder, DMR, the air and back.
    AmbeEncoder        encoder;
    std::vector<float> spoken;
    size_t             t = 0;
    const auto         voice = [&] {
        float frame[AmbeEncoder::kFrameSamples];
        for (float& v : frame) {
            const double s = static_cast<double>(t++) / 8000.0;
            double       x = 0.0;
            for (int h = 1; h <= 20; ++h)
                x += std::exp(-std::pow((h * 140.0 - 600.0) / 400.0, 2.0)) * std::sin(2.0 * kPi * h * 140.0 * s) / h;
            v = static_cast<float>(0.3 * x);
            spoken.push_back(v);
        }
        return encoder.encode(frame);
    };
    DmrTransmitter tx(1);
    tx.start_call(0, group_call(1, 1), 6, voice);
    DmrDecoder  decoder(48000.0);
    const auto  frames = decode(decoder, dmr_frequency(tx, 3.0, 48000.0, 0.0, 150.0), 48000.0);
    AmbeDecoder vocoder;
    std::vector<float> heard;
    for (const DmrVoice& v : frames) {
        if (v.first)
            vocoder.reset();
        float pcm[AmbeDecoder::kFrameSamples];
        vocoder.decode(v.frame, pcm);
        heard.insert(heard.end(), pcm, pcm + AmbeDecoder::kFrameSamples);
    }
    CHECK(frames.size() >= 100u);
    // Its pitch: the autocorrelation peaks at 1/140 s.
    const size_t from = heard.size() / 2, n = 1600;
    double       best = 0.0;
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
    double rms = 0.0;
    for (size_t i = 0; i < n; ++i)
        rms += heard[from + i] * heard[from + i];
    rms = std::sqrt(rms / n);
    std::printf("    pitch %.1f Hz, level %.3f\n", 8000.0 / lag, rms);
    CHECK(std::fabs(8000.0 / lag - 140.0) < 6.0);
    CHECK(rms > 0.02);

    // And the silence repeaters send - as real equipment sends it, so this
    // also checks the interleaving: its codewords arrive without an error.
    vocoder.reset();
    float  pcm[AmbeDecoder::kFrameSamples];
    double quiet = 0.0;
    for (int i = 0; i < 10; ++i) {
        CHECK_EQ(vocoder.decode(dmr::ambe_silence(), pcm), 0);
        for (float v : pcm)
            quiet = std::max(quiet, static_cast<double>(std::fabs(v)));
    }
    CHECK(quiet < 0.01);
}

TEST(listener_decodes_the_simulators_dmr)
{
    // The simulator's repeater makes a call on time slot 1 a second in; the
    // listener follows it, and while DMR is on the channel the FM buzz is
    // replaced: by the call's voice, or with no vocoder by silence.
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
    listener.tune(150e6 + SimSource::digital_stations(2.4e6)[0].offset_hz, 0.0, 12.5e3);

    const SimSource::DmrCall& expected = SimSource::kDmrCalls[0];
    bool                      called = false, quiet = true;
    float                     loudest = 0.0f;
    ListenStatus              s;
    const auto                until = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    while (std::chrono::steady_clock::now() < until) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        s = listener.status();
        const DmrCall& call = s.dmr.slot[expected.slot];
        if (s.dmr.present && call.active && call.ids_known) {
            called = call.group && call.destination == expected.talkgroup && call.source == expected.source;
            loudest = std::max(loudest, s.audio_rms);
        }
        if (s.dmr.present && !call.active)
            quiet &= s.audio_rms < 1e-3f;
        if (called && loudest > 0.01f)
            break;
    }
    CHECK(s.dmr.present);
    CHECK(s.dmr.repeater);
    CHECK_EQ(s.dmr.color_code, static_cast<int>(SimSource::kDmrColorCode));
    CHECK(called);
    CHECK(quiet);
    if (AmbeDecoder::available())
        CHECK(loudest > 0.01f);
    else
        CHECK(loudest < 1e-3f);
    std::printf("    %s; loudest %.3f\n", describe_dmr_slot(s.dmr, expected.slot).c_str(), static_cast<double>(loudest));

    listener.set_enabled(false);
    pipeline.stop();
}

TEST(listener_plays_only_dmr_when_asked)
{
    // With DMR only, the simulator's analog voice channel is silent; without,
    // it plays.  The DMR repeater is still decoded, and its call heard if
    // there is a vocoder.
    Options options;
    options.source.kind        = "sim";
    options.source.sample_rate = 2.4e6;
    options.source.frequency   = 150e6;
    options.source.gain        = SimSource::kReferenceGainDb;
    Pipeline pipeline(options);
    pipeline.start();
    Listener& listener = pipeline.listener();
    listener.set_volume(1.0f);  // the plot shows the audio as played (no sound card here: silent all the same)
    listener.set_dmr_only(true);
    listener.set_enabled(true);

    // The loudest of the last half second played.
    const auto loudest_played = [&] {
        std::vector<float> played(24000);
        played.resize(listener.scope().latest(played.data(), played.size()));
        float peak = 0.0f;
        for (float v : played)
            peak = std::max(peak, std::fabs(v));
        return peak;
    };

    listener.tune(150e6 + SimSource::voice_channel(2.4e6).offset_hz, 0.0, 12.5e3);
    std::this_thread::sleep_for(std::chrono::milliseconds(900));
    CHECK(listener.status().dmr_only);
    CHECK(!listener.status().dmr.present);
    CHECK(loudest_played() == 0.0f);
    CHECK(listener.status().audio_rms == 0.0f);

    listener.set_dmr_only(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    CHECK(loudest_played() > 0.1f);
    CHECK(listener.status().audio_rms > 0.05f);

    // The repeater's call on time slot 1 runs from 1 s to about 4.7 s.
    listener.set_dmr_only(true);
    listener.tune(150e6 + SimSource::digital_stations(2.4e6)[0].offset_hz, 0.0, 12.5e3);
    bool       called = false;
    const auto until  = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
    while (!called && std::chrono::steady_clock::now() < until) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const ListenStatus s = listener.status();
        called = s.dmr.present && s.dmr.slot[0].active;
    }
    CHECK(called);
    std::this_thread::sleep_for(std::chrono::milliseconds(600));  // half a second of nothing but the call
    CHECK(listener.status().dmr.slot[0].active);
    const float loudest = loudest_played();
    if (AmbeDecoder::available())
        CHECK(loudest > 0.02f);  // its voice
    else
        CHECK(loudest == 0.0f);  // silence, not the FM buzz
    std::printf("    DMR call heard at %.3f\n", static_cast<double>(loudest));

    listener.set_enabled(false);
    pipeline.stop();
}

TEST(heard_channels_remember_dmr_voice)
{
    HeardChannels heard;
    const auto    status = [](bool present, bool active, bool voice, bool ids, uint32_t source, bool encrypted = false) {
        DmrStatus s;
        s.present    = present;
        s.color_code = 5;
        DmrCall& c   = s.slot[0];
        c.active     = active;
        c.voice      = voice;
        c.ids_known  = ids;
        c.group      = true;
        c.destination = 1705;
        c.source     = source;
        c.encrypted  = encrypted;
        return s;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // Nothing but an idle repeater, or a call header without voice yet: nothing heard.
    heard.update(462.1745e6, status(true, false, false, false, 0), P25Status{}, 0.0);
    heard.update(462.1745e6, status(true, true, false, true, 11), P25Status{}, 0.1);
    heard.update(nan, status(true, true, true, true, 11), P25Status{}, 0.2);  // not settled on a channel
    CHECK(heard.channels().empty());

    // A call, joined late: its caller known only after a while - still one call.
    heard.update(462.1745e6, status(true, true, true, false, 0), P25Status{}, 1.0);
    heard.update(462.1745e6, status(true, true, true, true, 11), P25Status{}, 1.3);
    CHECK_EQ(heard.channels().size(), size_t{1});
    CHECK_EQ(heard.channels()[0].calls, 1);
    CHECK(heard.channels()[0].talkgroup == "TG 1705");
    CHECK_EQ(heard.channels()[0].color_code, 5);
    // Another radio takes over without a gap, then a gap, then a third call.
    heard.update(462.1745e6, status(true, true, true, true, 22), P25Status{}, 2.0);
    heard.update(462.1748e6, status(true, false, false, true, 22), P25Status{}, 3.0);  // the carrier a little off: the same channel
    heard.update(462.1745e6, status(true, true, true, true, 11), P25Status{}, 4.0);
    CHECK_EQ(heard.channels().size(), size_t{1});
    CHECK_EQ(heard.channels()[0].calls, 3);
    CHECK_NEAR(heard.channels()[0].last_heard, 4.0, 1e-12);
    CHECK(heard.channels()[0].last_call == "Group call to TG 1705 from 11");

    // Another channel, below: listed first; an encrypted call is flagged.
    heard.update(461.5e6, status(true, true, true, true, 33, true), P25Status{}, 5.0);
    CHECK_EQ(heard.channels().size(), size_t{2});
    CHECK_NEAR(heard.channels()[0].frequency, 461.5e6, 1e-6);
    CHECK(heard.channels()[0].encrypted);
    CHECK_EQ(heard.channels()[1].calls, 3);
    // Back on the first channel: the call there counts once (it was not
    // followed meanwhile), and only once.
    heard.update(462.1745e6, status(true, true, true, true, 11), P25Status{}, 6.0);
    CHECK_EQ(heard.channels()[1].calls, 4);
    heard.update(462.1745e6, status(true, true, true, true, 11), P25Status{}, 6.5);
    CHECK_EQ(heard.channels()[1].calls, 4);

    heard.clear();
    CHECK(heard.channels().empty());
}
