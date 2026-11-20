#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

#include "app/pipeline.hpp"
#include "audio/listener.hpp"
#include "audio/transcriber.hpp"
#include "audio/voice_overs.hpp"
#include "dsp/ambe_vocoder.hpp"
#include "radio/sim_source.hpp"
#include "test_framework.hpp"

using namespace spectro;

// CHECK_EQ for text, showing both.
#define CHECK_TEXT(a, b)                                                                                     \
    do {                                                                                                    \
        const std::string ta_ = (a);                                                                        \
        const std::string tb_ = (b);                                                                        \
        if (ta_ != tb_)                                                                                     \
            ::testing::report_failure(__FILE__, __LINE__, "CHECK_TEXT(" #a ") -> \"" + ta_ + "\" vs \"" + tb_ + "\""); \
    } while (0)

namespace {

struct Collected final : VoiceOverSink {
    std::mutex             mutex;
    std::vector<VoiceOver> overs;
    void voice_over(VoiceOver&& over) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        overs.push_back(std::move(over));
    }
    size_t count()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return overs.size();
    }
};

constexpr double kFrame = 0.02;  // the vocoders' frames

// `seconds` of speech on a stream, a frame at a time, the clock going on with it.
void speak(VoiceOvers& overs, VoiceOvers::Stream stream, double& now, double seconds, bool first)
{
    std::vector<float> frame(160, 0.1f);
    const long         frames = std::lround(seconds / kFrame);
    for (long i = 0; i < frames; ++i) {
        overs.push(stream, frame.data(), frame.size(), first && i == 0);
        now += kFrame;
        overs.advance(now);
    }
}

void pause(VoiceOvers& overs, double& now, double seconds)
{
    const long frames = std::lround(seconds / kFrame);
    for (long i = 0; i < frames; ++i) {
        now += kFrame;
        overs.advance(now);
    }
}

double wall_clock_now()
{
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// Letters and single spaces, lower case: to compare what was said.
[[maybe_unused]] std::string plain(const std::string& text)
{
    std::string out;
    for (const char c : text) {
        if (std::isalpha(static_cast<unsigned char>(c)))
            out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        else if (!out.empty() && out.back() != ' ')
            out += ' ';
    }
    while (!out.empty() && out.back() == ' ')
        out.pop_back();
    return out;
}

// A WAV file of 16-bit mono PCM; empty if it is not one.
[[maybe_unused]] std::vector<float> read_wav(const char* path, int& rate)
{
    std::ifstream in(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto u16 = [&](size_t at) { return static_cast<uint32_t>(static_cast<uint8_t>(bytes[at]) | static_cast<uint8_t>(bytes[at + 1]) << 8); };
    const auto u32 = [&](size_t at) { return u16(at) | u16(at + 2) << 16; };
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
        return {};
    std::vector<float> samples;
    bool               pcm16_mono = false;
    for (size_t at = 12; at + 8 <= bytes.size();) {
        const uint32_t size = u32(at + 4);
        if (std::memcmp(bytes.data() + at, "fmt ", 4) == 0 && at + 24 <= bytes.size()) {
            pcm16_mono = u16(at + 8) == 1 && u16(at + 10) == 1 && u16(at + 22) == 16;
            rate       = static_cast<int>(u32(at + 12));
        } else if (std::memcmp(bytes.data() + at, "data", 4) == 0 && pcm16_mono) {
            for (size_t k = at + 8; k + 1 < std::min<size_t>(bytes.size(), at + 8 + size); k += 2)
                samples.push_back(static_cast<float>(static_cast<int16_t>(u16(k))) / 32768.0f);
        }
        at += 8 + size + (size & 1);
    }
    return samples;
}

// From 16 kHz to the vocoders' 8 kHz: a windowed-sinc low-pass, every other sample.
[[maybe_unused]] std::vector<float> to_8k(const std::vector<float>& in)
{
    constexpr int      kHalf = 32;
    std::vector<float> taps(2 * kHalf + 1);
    double             sum = 0.0;
    for (int j = -kHalf; j <= kHalf; ++j) {
        const double t      = j * 3600.0 / 16000.0 * 2.0;  // cut-off 3.6 kHz
        const double sinc   = j == 0 ? 1.0 : std::sin(std::numbers::pi * t) / (std::numbers::pi * t);
        const double window = 0.42 + 0.5 * std::cos(std::numbers::pi * j / (kHalf + 1)) +
                              0.08 * std::cos(2.0 * std::numbers::pi * j / (kHalf + 1));
        taps[j + kHalf] = static_cast<float>(sinc * window);
        sum += taps[j + kHalf];
    }
    std::vector<float> out;
    for (size_t k = 0; k < in.size(); k += 2) {
        double acc = 0.0;
        for (int j = -kHalf; j <= kHalf; ++j) {
            const ptrdiff_t at = static_cast<ptrdiff_t>(k) + j;
            if (at >= 0 && static_cast<size_t>(at) < in.size())
                acc += taps[j + kHalf] * in[at];
        }
        out.push_back(static_cast<float>(acc / sum));
    }
    return out;
}

}  // namespace

TEST(voice_overs_end_with_a_pause)
{
    Collected  sink;
    VoiceOvers overs;
    overs.set_sink(&sink);
    overs.set_channel(155.9625e6);
    double now = 0.0;
    const double began = wall_clock_now();
    overs.identify(VoiceOvers::P25Phase1, "TG 100", 2001);
    speak(overs, VoiceOvers::P25Phase1, now, 1.2, true);
    CHECK(overs.open(VoiceOvers::P25Phase1));
    pause(overs, now, 0.5);  // a breath, within the over
    CHECK_EQ(sink.count(), size_t{0});
    speak(overs, VoiceOvers::P25Phase1, now, 0.6, false);
    pause(overs, now, 0.7);  // the over is done
    CHECK(!overs.open(VoiceOvers::P25Phase1));
    CHECK_EQ(sink.count(), size_t{1});
    if (sink.count() == 1) {
        const VoiceOver& o = sink.overs[0];
        CHECK_NEAR(o.seconds(), 1.8, 1e-9);
        CHECK(o.system == "P25");
        CHECK_EQ(o.slot, -1);
        CHECK(o.destination == "TG 100");
        CHECK_EQ(o.source, 2001u);
        CHECK_EQ(o.part, 0);
        CHECK_NEAR(o.frequency, 155.9625e6, 1e-3);
        CHECK(o.start >= began - 1.0 && o.start <= wall_clock_now() + 1.0);
    }
}

TEST(voice_overs_end_with_the_call_or_the_caller)
{
    Collected  sink;
    VoiceOvers overs;
    overs.set_sink(&sink);
    overs.set_channel(450e6);
    double now = 0.0;
    // The caller becomes known during the over: still the same one.
    overs.identify(VoiceOvers::DmrSlot2, "TG 2", 0);
    speak(overs, VoiceOvers::DmrSlot2, now, 1.0, true);
    overs.identify(VoiceOvers::DmrSlot2, "TG 2", 1002);
    speak(overs, VoiceOvers::DmrSlot2, now, 0.5, false);
    // Another radio takes over, without a gap.
    overs.identify(VoiceOvers::DmrSlot2, "TG 2", 1003);
    speak(overs, VoiceOvers::DmrSlot2, now, 1.0, false);
    // A new call: who it is comes later.
    speak(overs, VoiceOvers::DmrSlot2, now, 0.8, true);
    // A press of the button: too short to keep.
    overs.flush();
    speak(overs, VoiceOvers::DmrSlot1, now, 0.3, true);
    overs.flush();

    CHECK_EQ(sink.count(), size_t{3});
    if (sink.count() == 3) {
        CHECK_NEAR(sink.overs[0].seconds(), 1.5, 1e-9);
        CHECK_EQ(sink.overs[0].source, 1002u);
        CHECK(sink.overs[0].destination == "TG 2");
        CHECK_NEAR(sink.overs[1].seconds(), 1.0, 1e-9);
        CHECK_EQ(sink.overs[1].source, 1003u);
        CHECK_NEAR(sink.overs[2].seconds(), 0.8, 1e-9);
        CHECK_EQ(sink.overs[2].source, 0u);  // not the last caller's
        CHECK(sink.overs[2].destination.empty());
        for (const VoiceOver& o : sink.overs) {
            CHECK(o.system == "DMR");
            CHECK_EQ(o.slot, 1);
        }
    }
}

TEST(voice_overs_come_in_parts_when_long)
{
    Collected  sink;
    VoiceOvers overs;
    overs.set_sink(&sink);
    overs.set_channel(154.0875e6);
    double now = 0.0;
    overs.identify(VoiceOvers::P25Phase2Slot2, "TG 7", 77);
    speak(overs, VoiceOvers::P25Phase2Slot2, now, 30.0, true);
    CHECK_EQ(sink.count(), size_t{1});  // the first 28 s
    overs.flush();
    CHECK_EQ(sink.count(), size_t{2});
    if (sink.count() == 2) {
        CHECK_NEAR(sink.overs[0].seconds(), VoiceOvers::kMaxSeconds, 1e-9);
        CHECK_EQ(sink.overs[0].part, 0);
        CHECK_NEAR(sink.overs[1].seconds(), 30.0 - VoiceOvers::kMaxSeconds, 1e-9);
        CHECK_EQ(sink.overs[1].part, 1);
        CHECK_NEAR(sink.overs[1].start, sink.overs[0].start + VoiceOvers::kMaxSeconds, 1e-6);
        CHECK(sink.overs[1].destination == "TG 7");
        CHECK_EQ(sink.overs[1].source, 77u);
        CHECK(sink.overs[1].system == "P25 Phase 2");
        CHECK_EQ(sink.overs[1].slot, 1);
    }
}

TEST(voice_overs_end_on_another_channel)
{
    Collected  sink;
    VoiceOvers overs;
    overs.set_sink(&sink);
    overs.set_channel(460e6);
    double now = 0.0;
    speak(overs, VoiceOvers::DmrSlot1, now, 1.0, true);
    overs.set_channel(460e6 + 300.0);  // the same channel, found a little off
    CHECK(overs.open(VoiceOvers::DmrSlot1));
    overs.set_channel(460.0125e6);
    CHECK(!overs.open(VoiceOvers::DmrSlot1));
    CHECK_EQ(sink.count(), size_t{1});
    if (sink.count() == 1)
        CHECK_NEAR(sink.overs[0].frequency, 460e6, 1e-3);

    // Without a sink nothing is kept.
    VoiceOvers idle;
    speak(idle, VoiceOvers::DmrSlot1, now, 1.0, true);
    CHECK(!idle.open(VoiceOvers::DmrSlot1));
}

TEST(transcripts_are_cleaned_of_whispers_notes)
{
    CHECK(clean_transcript(" [BLANK_AUDIO]").empty());
    CHECK(clean_transcript("*beep*").empty());
    CHECK(clean_transcript(" \xE2\x99\xAA \xE2\x99\xAA").empty());
    CHECK(clean_transcript("   ").empty());
    CHECK(clean_transcript(" ...").empty());
    CHECK_TEXT(clean_transcript(" (static) Engine 5 responding."), "Engine 5 responding.");
    CHECK_TEXT(clean_transcript(" Copy that , unit 12 ."), "Copy that, unit 12.");
    CHECK_TEXT(clean_transcript(" Ten-four. [inaudible] Clear."), "Ten-four. Clear.");
    CHECK_TEXT(clean_transcript(" Unit (unclear"), "Unit (unclear");  // nothing closes it
    // Whisper stuck in a loop - but not someone saying a thing twice, or three times.
    CHECK_TEXT(clean_transcript(" Blue and white, white, white, white, white, white street."), "Blue and white street.");
    CHECK_TEXT(clean_transcript(" No, no. Go go go!"), "No, no. Go go go!");
}

TEST(speech_is_given_to_whisper_at_twice_the_rate)
{
    // A 1 kHz tone, quiet: the samples in between follow it, and its level is brought up.
    constexpr double kAmplitude = 0.02;
    std::vector<float> tone(4000);  // 0.5 s
    for (size_t k = 0; k < tone.size(); ++k)
        tone[k] = static_cast<float>(kAmplitude * std::sin(2.0 * std::numbers::pi * 1000.0 * static_cast<double>(k) / 8000.0));
    const std::vector<float> out = speech_for_whisper(tone.data(), tone.size());
    CHECK_EQ(out.size(), size_t{16000});  // made up to a second
    double power = 0.0, error = 0.0;
    const double gain = 0.1 / (kAmplitude / std::sqrt(2.0));
    for (size_t i = 400; i < 7600; ++i) {  // away from the ends
        const double expected = gain * kAmplitude * std::sin(2.0 * std::numbers::pi * 1000.0 * static_cast<double>(i) / 16000.0);
        power += out[i] * out[i];
        error += (out[i] - expected) * (out[i] - expected);
    }
    CHECK_NEAR(std::sqrt(power / 7200.0), 0.1, 0.002);
    CHECK(std::sqrt(error / power) < 0.01);  // within 1 %
    for (size_t i = 8000; i < out.size(); ++i)
        CHECK(out[i] == 0.0f);
}

TEST(transcript_lines)
{
    Transcript t;
    t.frequency   = 154.0875e6;
    t.system      = "P25 Phase 2";
    t.slot        = 0;
    t.destination = "TG 2601";
    t.source      = 21;
    t.seconds     = 3.24;
    t.text        = "Engine 5, on scene.";
    t.start       = wall_clock_now();
    const std::string line = format_transcript(t);
    CHECK(line.find("154.08750 MHz  P25 Phase 2 slot 1  TG 2601 from 21  (3.2 s): Engine 5, on scene.") !=
          std::string::npos);
    CHECK_TEXT(transcript_speaker(t), "TG 2601 from 21");
    t.destination.clear();
    CHECK_TEXT(transcript_speaker(t), "from 21");
    t.source = 0;
    CHECK(transcript_speaker(t).empty());
    t.system = "P25";
    t.slot   = -1;
    CHECK_TEXT(transcript_channel(t), "P25");
}

TEST(transcriber_without_a_model)
{
    Transcriber::Settings settings;
    settings.model = "/nonexistent/ggml-nothing.bin";
    Transcriber transcriber(settings);
    CHECK(transcriber.wait_idle(10.0));
    const TranscriberStatus s = transcriber.status();
    CHECK(s.state == TranscriberStatus::State::Failed);
    CHECK(s.model == "ggml-nothing.bin");
    if (Transcriber::available())
        CHECK(s.error.find("could not load") != std::string::npos);
    else
        CHECK(s.error.find("SPECTRO_WITH_WHISPER") != std::string::npos);
    VoiceOver over;
    over.speech.assign(8000, 0.1f);
    transcriber.voice_over(std::move(over));
    CHECK_EQ(transcriber.status().waiting, size_t{0});
    CHECK(transcriber.since(0).empty());
}

TEST(listener_hands_the_voice_over_an_over_at_a_time)
{
    // The simulator's DMR repeater: a call on time slot 1, a second into each
    // 12 s cycle.  (Without a vocoder its speech is silence, which is still an over.)
    Options options;
    options.source.kind        = "sim";
    options.source.sample_rate = 2.4e6;
    options.source.frequency   = 150e6;
    options.source.gain        = SimSource::kReferenceGainDb;
    Pipeline pipeline(options);
    pipeline.start();
    Collected sink;
    Listener& listener = pipeline.listener();
    listener.set_voice_sink(&sink);
    listener.set_volume(0.0f);
    listener.set_enabled(true);
    listener.tune(150e6 + SimSource::digital_stations(2.4e6)[0].offset_hz, 0.0, 12.5e3);

    const SimSource::DmrCall& call  = SimSource::kDmrCalls[0];
    const auto                until = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (sink.count() == 0 && std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipeline.stop();
    listener.set_voice_sink(nullptr);

    CHECK(sink.count() >= 1);
    if (sink.count() >= 1) {
        const VoiceOver& o = sink.overs[0];
        CHECK(o.system == "DMR");
        CHECK_EQ(o.slot, call.slot);
        CHECK(o.destination == "TG " + std::to_string(call.talkgroup));
        CHECK_EQ(o.source, call.source);
        CHECK(o.seconds() > call.seconds - 0.6 && o.seconds() <= call.seconds + 0.1);
        CHECK_NEAR(o.frequency, 150e6 + SimSource::digital_stations(2.4e6)[0].offset_hz, 1.0);
    }
}

// With a model (SPECTRO_WHISPER_MODEL=/path/to/ggml-base.en.bin): a recording of
// speech, as it is and through the AMBE+2 vocoder, comes out as what was said.
TEST(transcribing_speech)
{
#if SPECTRO_HAVE_WHISPER
    const char* model = std::getenv("SPECTRO_WHISPER_MODEL");
    if (!model || !*model) {
        std::printf("    (skipped: set SPECTRO_WHISPER_MODEL to a Whisper model to run it)\n");
        return;
    }
    int                rate   = 0;
    const std::vector<float> wav = read_wav(SPECTRO_WHISPER_SAMPLE, rate);
    CHECK(!wav.empty() && rate == 16000);
    if (wav.empty() || rate != 16000)
        return;
    const std::vector<float> speech = to_8k(wav);

    Transcriber::Settings settings;
    settings.model = model;
    Transcriber transcriber(settings);
    const auto over_of = [](std::vector<float> samples) {
        VoiceOver o;
        o.start       = wall_clock_now();
        o.frequency   = 155.9625e6;
        o.system      = "DMR";
        o.slot        = 0;
        o.destination = "TG 1";
        o.source      = 1001;
        o.speech      = std::move(samples);
        return o;
    };
    transcriber.voice_over(over_of(speech));
    transcriber.voice_over(over_of(std::vector<float>(16000, 0.0f)));  // two seconds of silence
    if (AmbeEncoder::available()) {
        AmbeEncoder        encoder;
        AmbeDecoder        decoder;
        std::vector<float> coded(speech.size() / 160 * 160);
        for (size_t k = 0; k + 160 <= speech.size(); k += 160)
            decoder.decode(encoder.encode(speech.data() + k), coded.data() + k);
        transcriber.voice_over(over_of(std::move(coded)));
    }
    CHECK(transcriber.wait_idle(120.0));
    const TranscriberStatus       s     = transcriber.status();
    const std::vector<Transcript> texts = transcriber.since(0);
    CHECK(s.state == TranscriberStatus::State::Ready);
    CHECK_EQ(s.silent, uint64_t{1});
    CHECK_EQ(texts.size(), size_t{AmbeEncoder::available() ? 2u : 1u});
    for (const Transcript& t : texts) {
        std::printf("    %s  [confidence %.2f, %.1fx real time]\n", format_transcript(t).c_str(),
                    static_cast<double>(t.confidence), s.speed);
        // ("Ask not" can come out as "asked not": the rest is clear.)
        CHECK(plain(t.text).find("what your country can do for you") != std::string::npos);
        CHECK(plain(t.text).find("ask what you can do for your country") != std::string::npos);
        CHECK(t.destination == "TG 1");
        CHECK_EQ(t.source, 1001u);
    }
#endif
}
