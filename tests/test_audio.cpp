#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <numbers>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "app/pipeline.hpp"
#include "audio/audio_fifo.hpp"
#include "audio/listener.hpp"
#include "audio/rate_control.hpp"
#include "dsp/fm_demod.hpp"
#include "radio/sim_source.hpp"
#include "test_framework.hpp"

using namespace spectro;

namespace {

constexpr double kTwoPi     = 2.0 * std::numbers::pi;
constexpr double kAudioRate = 48000.0;

double amplitude_of(double dbfs) { return 2048.0 * std::pow(10.0, dbfs / 20.0); }

// Adds an FM signal to `iq`: a carrier `offset_hz` from the centre of the
// stream, frequency-modulated by a tone (deviation 0: a plain carrier).
void add_fm(std::vector<double>& iq, double fs, double offset_hz, double level_dbfs, double tone_hz,
            double deviation_hz)
{
    const double amp   = amplitude_of(level_dbfs);
    const double index = tone_hz > 0.0 ? deviation_hz / tone_hz : 0.0;
    const size_t n     = iq.size() / 2;
    for (size_t i = 0; i < n; ++i) {
        const double t     = static_cast<double>(i) / fs;
        const double phase = kTwoPi * std::fmod(offset_hz * t, 1.0) + index * std::sin(kTwoPi * tone_hz * t);
        iq[2 * i]     += amp * std::cos(phase);
        iq[2 * i + 1] += amp * std::sin(phase);
    }
}

// Adds an FM signal modulated by several tones, each with its own deviation.
struct ToneMod {
    double hz;
    double deviation_hz;
};
void add_fm_tones(std::vector<double>& iq, double fs, double offset_hz, double level_dbfs,
                  const std::vector<ToneMod>& tones)
{
    const double amp = amplitude_of(level_dbfs);
    const size_t n   = iq.size() / 2;
    for (size_t i = 0; i < n; ++i) {
        const double t     = static_cast<double>(i) / fs;
        double       phase = kTwoPi * std::fmod(offset_hz * t, 1.0);
        for (const ToneMod& m : tones)
            phase += m.deviation_hz / m.hz * std::sin(kTwoPi * m.hz * t);
        iq[2 * i]     += amp * std::cos(phase);
        iq[2 * i + 1] += amp * std::sin(phase);
    }
}

// The gain of a broadcast transmitter's pre-emphasis (1 + j w tau).
double pre_emphasis(double hz, double tau = 75e-6) { return std::hypot(1.0, kTwoPi * hz * tau); }

// Receiver noise of `sigma` counts per component, then the converter.
std::vector<int16_t> digitise(const std::vector<double>& iq, double sigma, unsigned seed = 1)
{
    std::mt19937                     rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<int16_t>             out(iq.size());
    for (size_t j = 0; j < iq.size(); ++j) {
        const double v = iq[j] + (sigma > 0.0 ? sigma * gauss(rng) : 0.0);
        out[j]         = static_cast<int16_t>(std::lround(std::clamp(v, -2047.0, 2047.0)));
    }
    return out;
}

// Feeds a recording through the demodulator in radio-sized pieces.
std::vector<float> demodulate(FmDemodulator& demod, const std::vector<int16_t>& iq, size_t piece = 3001)
{
    std::vector<float> audio;
    const size_t       n = iq.size() / 2;
    for (size_t done = 0; done < n; done += piece)
        demod.process(iq.data() + 2 * done, std::min(piece, n - done), audio);
    return audio;
}

struct Tone {
    double amplitude = 0.0;  // of the component at the probed frequency
    double residual  = 0.0;  // RMS of everything else
};

// Measures one frequency component of audio[first, first + count).  Choose
// `count` to span a whole number of its cycles.
Tone measure_tone(const std::vector<float>& audio, size_t first, size_t count, double hz, double rate = kAudioRate)
{
    CHECK(first + count <= audio.size());
    if (first + count > audio.size())
        return {};
    const double w  = kTwoPi * hz / rate;
    double       re = 0.0, im = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double v = audio[first + i];
        re += v * std::cos(w * static_cast<double>(i));
        im += v * std::sin(w * static_cast<double>(i));
    }
    re *= 2.0 / static_cast<double>(count);
    im *= 2.0 / static_cast<double>(count);

    double rest = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double fit = re * std::cos(w * static_cast<double>(i)) + im * std::sin(w * static_cast<double>(i));
        const double e   = audio[first + i] - fit;
        rest += e * e;
    }
    Tone t;
    t.amplitude = std::hypot(re, im);
    t.residual  = std::sqrt(rest / static_cast<double>(count));
    return t;
}

// Stands in for a sound card: fetches a period of audio at a time, at its own
// pace, and keeps what it "played".
class FakeSoundCard {
public:
    FakeSoundCard(AudioFifo& fifo, double rate, size_t period) : fifo_(fifo)
    {
        thread_ = std::thread([this, rate, period] {
            std::vector<float> buffer(period);
            const auto         start = std::chrono::steady_clock::now();
            for (uint64_t periods = 1; !stop_.load(); ++periods) {
                std::this_thread::sleep_until(
                    start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                std::chrono::duration<double>(static_cast<double>(periods * period) / rate)));
                fifo_.read(buffer.data(), period);
                std::lock_guard<std::mutex> lock(mutex_);
                played_.insert(played_.end(), buffer.begin(), buffer.end());
            }
        });
    }
    ~FakeSoundCard()
    {
        stop_.store(true);
        thread_.join();
    }

    // Everything played since the previous call.
    std::vector<float> take()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<float>          out;
        out.swap(played_);
        return out;
    }

private:
    AudioFifo&         fifo_;
    std::thread        thread_;
    std::atomic<bool>  stop_{false};
    std::mutex         mutex_;
    std::vector<float> played_;
};

void wait_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// The share of 40 ms pieces of `audio` that hold one of `notes` at the
// expected loudness.
const std::vector<double> kVoiceNotes     = {523.25, 659.25, 783.99, 1046.50};
const std::vector<double> kBroadcastNotes = {220.0, 261.63, 329.63, 440.0};

double share_of_tune(const std::vector<float>& audio, double amplitude, const std::vector<double>& notes = kVoiceNotes)
{
    constexpr size_t kPiece = 1920;
    int              pieces = 0, good = 0;
    for (size_t first = 0; first + kPiece <= audio.size(); first += kPiece) {
        double best = 0.0;
        for (const double hz : notes)
            best = std::max(best, measure_tone(audio, first, kPiece, hz).amplitude);
        ++pieces;
        if (std::fabs(best - amplitude) < 0.15 * amplitude)
            ++good;
    }
    return pieces > 0 ? static_cast<double>(good) / pieces : 0.0;
}

}  // namespace

TEST(nbfm_recovers_a_tone)
{
    // A -30 dBFS signal 237.3 kHz above the centre of a 1 MS/s stream, carrying
    // a 1 kHz tone with 2 kHz deviation.
    const double        fs = 1e6;
    std::vector<double> scene(2 * 1'000'000, 0.0);
    add_fm(scene, fs, 237.3e3, -30.0, 1000.0, 2000.0);
    const std::vector<int16_t> iq = digitise(scene, 1.0);

    FmDemodulator demod(fs, kAudioRate);
    demod.set_channel(237.3e3, 12.5e3);
    CHECK_EQ(demod.decimation(), 10);
    const std::vector<float> audio = demodulate(demod, iq);

    // One second of signal gives one second of audio, less the filters' fill time.
    CHECK(audio.size() > 47'800 && audio.size() <= 48'000);

    // Full scale is a deviation of 0.4 x 12.5 kHz = 5 kHz, so 2 kHz reads 0.4.
    const Tone tone = measure_tone(audio, 9'600, 36'000, 1000.0);
    CHECK_NEAR(tone.amplitude, 0.4, 0.008);
    CHECK(tone.residual < 0.004);  // noise and distortion more than 37 dB down

    const FmDemodulator::Stats stats = demod.stats();
    CHECK_NEAR(stats.level_dbfs, -30.0, 0.3);  // the same scale as the spectrum display
    CHECK(stats.squelch_open);
    CHECK_NEAR(stats.audio_rms, 0.4 / std::sqrt(2.0), 0.02);
}

TEST(nbfm_works_at_any_sample_rate)
{
    // From an IQ recording barely wider than the channel up to a full-rate
    // bladeRF stream; the last two exercise the decimating first stage.
    for (const double fs : {24e3, 48e3, 250e3, 2.4e6, 20e6}) {
        const double        offset = 0.1 * fs;
        std::vector<double> scene(2 * static_cast<size_t>(0.4 * fs), 0.0);
        add_fm(scene, fs, offset, -20.0, 1000.0, 2000.0);
        const std::vector<int16_t> iq = digitise(scene, 0.0);

        FmDemodulator demod(fs, kAudioRate);
        demod.set_channel(offset, 12.5e3);
        const std::vector<float> audio = demodulate(demod, iq, 16384);
        CHECK(audio.size() > 18'900 && audio.size() <= 19'200);
        if (audio.size() < 18'900)
            continue;
        const Tone tone = measure_tone(audio, 7'200, 11'040, 1000.0);
        CHECK_NEAR(tone.amplitude, 0.4, 0.012);
        CHECK(tone.residual < 0.006);
        CHECK_NEAR(demod.stats().level_dbfs, -20.0, 0.4);
    }

    bool threw = false;
    try {
        FmDemodulator too_slow(8000.0, kAudioRate);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(nbfm_rejects_other_channels)
{
    const double fs = 2.4e6;

    // A signal two channels away and 30 dB stronger must not be heard.
    {
        std::vector<double> scene(2 * static_cast<size_t>(0.5 * fs), 0.0);
        add_fm(scene, fs, -400e3, -40.0, 1000.0, 2000.0);
        add_fm(scene, fs, -400e3 + 25e3, -10.0, 1700.0, 2000.0);
        const std::vector<int16_t> iq = digitise(scene, 0.5);

        FmDemodulator demod(fs, kAudioRate);
        demod.set_channel(-400e3, 12.5e3);
        const std::vector<float> audio = demodulate(demod, iq);
        CHECK(audio.size() > 23'000);
        if (audio.size() > 23'000) {
            CHECK_NEAR(measure_tone(audio, 9'600, 12'960, 1000.0).amplitude, 0.4, 0.02);
            CHECK(measure_tone(audio, 9'600, 12'960, 1700.0).amplitude < 0.004);
        }
        CHECK_NEAR(demod.stats().level_dbfs, -40.0, 0.5);
    }

    // The first stage decimates 2.4 MS/s to 100 kS/s.  A strong carrier 100 kHz
    // away would fold right into the channel if that stage let it through.
    {
        std::vector<double> scene(2 * static_cast<size_t>(0.3 * fs), 0.0);
        add_fm(scene, fs, 300e3 + 100e3 + 3e3, -6.0, 0.0, 0.0);
        const std::vector<int16_t> iq = digitise(scene, 0.5);

        FmDemodulator demod(fs, kAudioRate);
        demod.set_channel(300e3, 12.5e3);
        CHECK_EQ(demod.decimation(), 24);
        demodulate(demod, iq);
        CHECK(demod.stats().level_dbfs < -80.0f);  // only the noise floor is left
    }
}

TEST(nbfm_squelch_follows_the_signal)
{
    const double fs      = 250e3;
    const size_t half_s  = 125'000;
    const double offset  = 40e3;

    // Noise of 4 counts per component is -51.2 dBFS over the whole stream and
    // 13 dB less in a 12.5 kHz channel.
    std::vector<double> silence(2 * half_s, 0.0);
    std::vector<double> signal(2 * half_s, 0.0);
    add_fm(signal, fs, offset, -40.0, 1000.0, 2000.0);

    FmDemodulator demod(fs, kAudioRate);
    demod.set_channel(offset, 12.5e3);
    demod.set_squelch(-50.0f);

    std::vector<float> quiet = demodulate(demod, digitise(silence, 4.0, 1));
    CHECK(!demod.stats().squelch_open);
    CHECK_NEAR(demod.stats().level_dbfs, -64.2, 1.5);
    CHECK(!quiet.empty());
    CHECK(std::all_of(quiet.begin(), quiet.end(), [](float v) { return v == 0.0f; }));

    std::vector<float> speech = demodulate(demod, digitise(signal, 4.0, 2));
    CHECK(demod.stats().squelch_open);
    CHECK_NEAR(demod.stats().level_dbfs, -40.0, 0.5);
    CHECK_NEAR(measure_tone(speech, 12'000, 11'040, 1000.0).amplitude, 0.4, 0.02);

    std::vector<float> tail = demodulate(demod, digitise(silence, 4.0, 3));
    CHECK(!demod.stats().squelch_open);
    // It closes after a hold time of 150 ms and a 30 ms ramp.
    CHECK(tail.size() > 23'000);
    if (tail.size() > 23'000) {
        CHECK(std::any_of(tail.begin(), tail.begin() + 4'800, [](float v) { return v != 0.0f; }));
        CHECK(std::all_of(tail.begin() + 12'000, tail.end(), [](float v) { return v == 0.0f; }));
    }

    // Without a threshold the noise comes through.
    demod.set_squelch(FmDemodulator::kSquelchOff);
    std::vector<float> hiss = demodulate(demod, digitise(silence, 4.0, 4));
    CHECK(demod.stats().squelch_open);
    CHECK(demod.stats().audio_rms > 0.1f);
}

TEST(nbfm_output_rate_can_be_trimmed)
{
    const double        fs = 250e3;
    std::vector<double> scene(2 * 250'000, 0.0);
    add_fm(scene, fs, 10e3, -20.0, 1000.0, 2000.0);
    const std::vector<int16_t> iq = digitise(scene, 0.0);

    size_t counts[3] = {};
    int    k         = 0;
    for (const double trim : {-0.004, 0.0, 0.004}) {
        FmDemodulator demod(fs, kAudioRate);
        demod.set_channel(10e3, 12.5e3);
        demod.set_rate_trim(trim);
        const std::vector<float> audio = demodulate(demod, iq);
        counts[k++] = audio.size();
        // The tone is still 1 kHz of signal time, now at a slightly different audio rate.
        CHECK_NEAR(measure_tone(audio, 9'600, 36'000, 1000.0, kAudioRate * (1.0 + trim)).amplitude, 0.4, 0.012);
    }
    // 0.4 % of 48 000 samples per second is 192 samples.
    CHECK_NEAR(static_cast<double>(counts[2]) - static_cast<double>(counts[1]), 192.0, 3.0);
    CHECK_NEAR(static_cast<double>(counts[1]) - static_cast<double>(counts[0]), 192.0, 3.0);
}

TEST(nbfm_reports_where_the_carrier_is)
{
    const double fs = 2.4e6;
    for (const double error : {1500.0, -2300.0}) {
        std::vector<double> scene(2 * static_cast<size_t>(0.3 * fs), 0.0);
        add_fm(scene, fs, 500e3 + error, -35.0, 600.0, 1500.0);
        const std::vector<int16_t> iq = digitise(scene, 1.0);

        FmDemodulator demod(fs, kAudioRate);
        demod.set_channel(500e3, 12.5e3);
        std::vector<float> audio;
        const size_t       settle = static_cast<size_t>(0.1 * fs);
        demod.process(iq.data(), settle, audio);
        demod.take_frequency_estimate();  // discard the start-up
        demod.process(iq.data() + 2 * settle, iq.size() / 2 - settle, audio);
        const FmDemodulator::FrequencyEstimate estimate = demod.take_frequency_estimate();
        CHECK(estimate.samples > 9'000);
        CHECK_NEAR(estimate.mean_hz, error, 30.0);
        CHECK_EQ(demod.take_frequency_estimate().samples, size_t{0});
    }
}

TEST(audio_fifo_is_a_jitter_buffer)
{
    AudioFifo fifo(1000);
    CHECK_EQ(fifo.capacity(), size_t{1024});
    fifo.set_start_level(300);
    fifo.set_flowing(true);

    std::vector<float> ramp(2000);
    for (size_t i = 0; i < ramp.size(); ++i)
        ramp[i] = static_cast<float>(i + 1);
    float out[400];

    // Nothing is played until the start level is reached.
    CHECK_EQ(fifo.write(ramp.data(), 200), size_t{200});
    fifo.read(out, 100);
    CHECK(!fifo.playing());
    CHECK(std::all_of(out, out + 100, [](float v) { return v == 0.0f; }));
    CHECK_EQ(fifo.fill(), size_t{200});

    CHECK_EQ(fifo.write(ramp.data() + 200, 150), size_t{150});
    fifo.read(out, 100);
    CHECK(fifo.playing());
    CHECK_EQ(out[0], 1.0f);
    CHECK_EQ(out[99], 100.0f);
    CHECK_EQ(fifo.underruns(), uint64_t{0});

    // Running dry: the rest is played, silence follows, and playback waits
    // for the start level again.
    fifo.read(out, 400);
    CHECK_EQ(out[0], 101.0f);
    CHECK_EQ(out[249], 350.0f);
    CHECK_EQ(out[250], 0.0f);
    CHECK(!fifo.playing());
    CHECK_EQ(fifo.underruns(), uint64_t{1});
    CHECK_EQ(fifo.largest_read(), size_t{400});

    // It never holds more than its capacity, and wraps around correctly.
    CHECK_EQ(fifo.write(ramp.data(), 2000), size_t{1024});
    fifo.read(out, 400);
    CHECK_EQ(out[0], 1.0f);
    CHECK_EQ(fifo.write(ramp.data(), 2000), size_t{400});
    fifo.read(out, 400);
    CHECK_EQ(out[0], 401.0f);
    fifo.read(out, 400);
    CHECK_EQ(out[223], 1024.0f);
    CHECK_EQ(out[224], 1.0f);

    // discard() empties it from the consumer's side; an idle producer running
    // dry is not an underrun.
    fifo.discard();
    fifo.set_flowing(false);
    fifo.read(out, 100);
    CHECK_EQ(fifo.fill(), size_t{0});
    CHECK(!fifo.playing());
    CHECK(std::all_of(out, out + 100, [](float v) { return v == 0.0f; }));
    fifo.write(ramp.data(), 350);
    fifo.read(out, 400);  // more than is there
    CHECK_EQ(fifo.underruns(), uint64_t{1});
}

TEST(audio_rate_follows_the_sound_card)
{
    // Simulated time: the radio delivers blocks of samples, a sound card whose
    // clock differs fetches periods of audio.  The control loop must settle on
    // the rate that matches, without the FIFO ever running dry or piling up.
    struct Case {
        double clock_error;    // radio clock relative to the sound card's
        double block_ms;       // radio block size
        size_t period;         // sound card period, samples
    };
    const Case cases[] = {
        {+300e-6, 2.0, 480},     // bladeRF blocks, 10 ms periods
        {-2000e-6, 10.0, 480},   // a poor crystal
        {+1000e-6, 65.0, 1024},  // coarse blocks, as from a slow recording
        {0.0, 10.0, 4096},       // a sound server with long periods
    };
    const std::vector<float> silence(8192, 0.0f);
    std::vector<float>       out(8192);

    for (const Case& c : cases) {
        AudioFifo        fifo(48000);
        AudioRateControl control(48000.0);
        fifo.set_flowing(true);

        const double block_s  = c.block_ms / 1000.0;
        const double period_s = static_cast<double>(c.period) / 48000.0;
        double       next_block = 0.0, next_period = 0.0, owed = 0.0;
        double       trim = 0.0;
        size_t       peak_fill = 0;
        uint64_t     discards = 0;
        bool         started = false;
        uint64_t     underruns_after_start = 0;

        for (double t = 0.0; t < 120.0;) {
            if (next_block <= next_period) {
                t = next_block;
                next_block += block_s;
                const AudioRateControl::Decision d =
                    control.update(block_s, fifo.fill(), fifo.largest_read(), fifo.playing());
                fifo.set_start_level(d.start_level);
                trim = d.trim;
                // What the demodulator would produce from this block.
                owed += block_s * 48000.0 * (1.0 + c.clock_error) * (1.0 + trim);
                const size_t n = static_cast<size_t>(owed);
                owed -= static_cast<double>(n);
                if (d.discard)
                    ++discards;
                else
                    fifo.write(silence.data(), n);
                peak_fill = std::max(peak_fill, fifo.fill());
            } else {
                t = next_period;
                next_period += period_s;
                const uint64_t before = fifo.underruns();
                fifo.read(out.data(), c.period);
                if (started)
                    underruns_after_start += fifo.underruns() - before;
                started = started || fifo.playing();
            }
        }

        CHECK(started);
        CHECK_EQ(underruns_after_start, uint64_t{0});
        CHECK_EQ(discards, uint64_t{0});
        // The rate correction cancels the clock error ...
        CHECK_NEAR(trim, -c.clock_error, 1e-4);
        // ... with no more audio waiting than the blocks involved require.
        CHECK(static_cast<double>(peak_fill) / 48000.0 < 0.1 + 1.5 * block_s + 3.0 * period_s);
    }

    // A sound card that stops fetching audio: the surplus is thrown away
    // rather than played late.
    AudioRateControl control(48000.0);
    CHECK(!control.update(0.01, 480, 480, true).discard);
    CHECK(control.update(0.01, 48000 / 2, 480, true).discard);
}

TEST(listener_plays_the_simulators_voice_channel)
{
    // End to end, except for the sound card: simulator -> receive thread ->
    // tap -> audio thread -> FIFO.
    Options options;
    options.source.kind        = "sim";
    options.source.sample_rate = 1e6;
    options.source.frequency   = 150e6;
    options.source.gain        = SimSource::kReferenceGainDb;

    Pipeline pipeline(options);
    pipeline.start();
    Listener& listener = pipeline.listener();
    CHECK(listener.available());

    const SimSource::VoiceChannel voice     = SimSource::voice_channel(1e6);
    const double                  voice_hz  = 150e6 + voice.offset_hz;
    // Full scale is a deviation of 0.4 x 12.5 kHz; every note is played with
    // the deviation a 1 kHz tone would have.
    const double                  amplitude = voice.deviation_hz / (0.4 * 12.5e3);

    listener.set_volume(1.0f);
    listener.tune(voice_hz);
    CHECK(!listener.status().enabled);

    {
        FakeSoundCard card(listener.audio(), 48000.0 * 1.0003, 480);
        listener.set_enabled(true);
        wait_ms(1500);

        ListenStatus status = listener.status();
        CHECK(status.enabled);
        CHECK(status.in_band);
        CHECK(status.squelch_open);
        CHECK(!status.searching);
        CHECK_NEAR(status.frequency, voice_hz, 1e-6);
        CHECK_NEAR(status.level_dbfs, voice.level_dbfs, 1.0);
        CHECK_NEAR(status.audio_rms, amplitude / std::sqrt(2.0), 0.03);
        CHECK(status.audio_samples > 60'000 && status.audio_samples < 80'000);
        CHECK_EQ(status.underruns, uint64_t{0});
        CHECK_EQ(status.overruns, uint64_t{0});
        CHECK_EQ(status.dropped_iq, uint64_t{0});
        CHECK(status.load > 0.0f && status.load < 0.5f);

        // What reached the sound card: about 1.4 s of the tune, without holes.
        std::vector<float> played = card.take();
        CHECK(played.size() > 60'000);
        if (played.size() > 60'000) {
            const std::vector<float> recent(played.end() - 48'000, played.end());
            CHECK(share_of_tune(recent, amplitude) > 0.7);  // the rest are the pieces where the note changes
            CHECK(std::none_of(recent.begin(), recent.end(), [](float v) { return std::fabs(v) > 0.5f; }));
        }

        // Half the volume setting is a quarter of the amplitude.
        listener.set_volume(0.5f);
        wait_ms(400);
        card.take();
        wait_ms(500);
        played = card.take();
        CHECK(played.size() > 19'200);
        if (played.size() > 19'200)
            CHECK(share_of_tune(std::vector<float>(played.end() - 19'200, played.end()), amplitude / 4.0) > 0.7);
        listener.set_volume(1.0f);

        // Tuned 4 kHz off, as after a click on a coarse display, with a search
        // of 5 kHz: it finds the carrier and centres the channel on it.
        listener.tune(voice_hz + 4000.0, 5000.0);
        wait_ms(500);
        status = listener.status();
        CHECK(!status.searching);
        CHECK_NEAR(status.frequency, voice_hz, 300.0);
        CHECK(status.squelch_open);
        CHECK_NEAR(status.audio_rms, amplitude / std::sqrt(2.0), 0.04);

        // Retuning the radio by 60 % of the span takes the channel out of the
        // received band: silence, but not an error.
        const uint64_t underruns = listener.status().underruns;
        pipeline.source().set_frequency(150e6 + 0.6e6);
        wait_ms(400);
        status = listener.status();
        CHECK(!status.in_band);
        CHECK(!status.squelch_open);
        CHECK_EQ(status.underruns, underruns);
        card.take();
        wait_ms(200);
        played = card.take();
        CHECK(!played.empty());
        CHECK(std::all_of(played.begin(), played.end(), [](float v) { return v == 0.0f; }));

        // Back again, and the tune is back too: the channel is an absolute frequency.
        pipeline.source().set_frequency(150e6 + 0.1e6);
        wait_ms(700);
        status = listener.status();
        CHECK(status.in_band);
        CHECK_NEAR(status.level_dbfs, voice.level_dbfs, 1.0);
        played = card.take();
        CHECK(played.size() > 19'200);
        if (played.size() > 19'200)
            CHECK(share_of_tune(std::vector<float>(played.end() - 19'200, played.end()), amplitude) > 0.7);

        // A squelch threshold above the signal mutes it.
        listener.set_squelch(-20.0f);
        wait_ms(1000);  // it closes after 150 ms; the loudness reading then decays
        CHECK(!listener.status().squelch_open);
        CHECK(listener.status().audio_rms < 0.01f);
        listener.set_squelch(FmDemodulator::kSquelchOff);

        // Switched off, the audio thread goes to sleep.
        listener.set_enabled(false);
        wait_ms(200);
        const uint64_t produced = listener.status().audio_samples;
        wait_ms(200);
        CHECK_EQ(listener.status().audio_samples, produced);
        CHECK(!listener.status().enabled);
    }

    // The spectrum path never noticed.
    const AcquisitionStats stats = pipeline.acquisition().stats();
    CHECK_EQ(stats.lost_samples, uint64_t{0});
    CHECK_EQ(stats.ring_dropped, uint64_t{0});
    CHECK(!stats.failed);
    pipeline.stop();

    // After the pipeline has stopped, the controls are harmless.
    listener.set_enabled(true);
    CHECK(!listener.status().enabled);
}

TEST(wfm_recovers_a_tone_and_removes_the_pilot)
{
    // A broadcast station 500 kHz above the centre of a 2.4 MS/s stream: a
    // 1 kHz tone at 50 kHz deviation, pre-emphasised with 75 us, and the
    // 19 kHz stereo pilot.
    const double        fs = 2.4e6;
    std::vector<double> scene(2 * static_cast<size_t>(fs), 0.0);
    add_fm_tones(scene, fs, 500e3, -20.0, {{1000.0, 50e3 * pre_emphasis(1000.0)}, {19e3, 6.75e3}});
    const std::vector<int16_t> iq = digitise(scene, 1.0);

    FmDemodulator demod(fs, kAudioRate);
    demod.set_channel(500e3, 200e3);
    CHECK(demod.mode() == FmMode::Wide);
    CHECK_EQ(demod.decimation(), 2);
    CHECK_NEAR(demod.discriminator_rate(), 8.0 * kAudioRate, 1e-9);
    const std::vector<float> audio = demodulate(demod, iq, 16384);
    CHECK(audio.size() > 47'500 && audio.size() <= 48'000);

    // Full scale is 75 kHz: 50 kHz reads 0.667 once the de-emphasis has undone
    // the pre-emphasis.  The pilot (0.09 of full scale) must be gone.
    const Tone tone = measure_tone(audio, 9'600, 36'000, 1000.0);
    CHECK_NEAR(tone.amplitude, 50.0 / 75.0, 0.007);
    CHECK(tone.residual < 0.003);
    CHECK(measure_tone(audio, 9'600, 36'000, 19000.0).amplitude < 0.0005);
    CHECK_NEAR(demod.stats().level_dbfs, -20.0, 0.3);

    // With 50 us de-emphasis the same signal comes out brighter: 75 us of
    // pre-emphasis, only 50 us taken out again.
    FmDemodulator europe(fs, kAudioRate);
    europe.set_channel(500e3, 200e3);
    europe.set_deemphasis(50e-6);
    const std::vector<float> bright = demodulate(europe, iq, 16384);
    CHECK_NEAR(measure_tone(bright, 9'600, 36'000, 1000.0).amplitude,
               50.0 / 75.0 * pre_emphasis(1000.0) / pre_emphasis(1000.0, 50e-6), 0.007);
}

TEST(wfm_works_at_any_rate_wide_enough)
{
    CHECK(!FmDemodulator::supports(48e3, kAudioRate, FmMode::Wide));
    CHECK(FmDemodulator::supports(48e3, kAudioRate, FmMode::Narrow));
    CHECK(FmDemodulator::supports(250e3, kAudioRate, FmMode::Wide));
    CHECK_NEAR(FmDemodulator::widest(250e3, kAudioRate, FmMode::Wide), 195e3, 1.0);
    CHECK_NEAR(FmDemodulator::usable_bandwidth(250e3, kAudioRate, 250e3), 195e3, 1.0);
    CHECK_NEAR(FmDemodulator::usable_bandwidth(2.4e6, kAudioRate, 250e3), 250e3, 1e-9);
    CHECK_NEAR(FmDemodulator::usable_bandwidth(2.4e6, kAudioRate, 12.5e3), 12.5e3, 1e-9);
    // Wideband FM does not fit: the widest narrowband channel instead.
    CHECK_NEAR(FmDemodulator::usable_bandwidth(48e3, kAudioRate, 200e3), FmDemodulator::kNarrowMaxBandwidth, 1.0);

    struct Case {
        double fs;
        int    decimation;
    };
    for (const Case c : {Case{250e3, 1}, Case{1e6, 1}, Case{2.4e6, 2}, Case{20e6, 16}}) {
        const double        offset = c.fs >= 1e6 ? 0.3 * c.fs : 0.0;
        std::vector<double> scene(2 * static_cast<size_t>(0.5 * c.fs), 0.0);
        add_fm_tones(scene, c.fs, offset, -20.0, {{1000.0, 50e3 * pre_emphasis(1000.0)}});
        const std::vector<int16_t> iq = digitise(scene, 0.0);

        FmDemodulator demod(c.fs, kAudioRate);
        demod.set_channel(offset, 200e3);
        CHECK_EQ(demod.decimation(), c.decimation);
        const std::vector<float> audio = demodulate(demod, iq, 16384);
        CHECK(audio.size() > 23'000);
        if (audio.size() < 23'000)
            continue;
        const Tone tone = measure_tone(audio, 9'600, 12'000, 1000.0);
        CHECK_NEAR(tone.amplitude, 50.0 / 75.0, 0.012);
        CHECK(tone.residual < 0.006);
    }
}

TEST(wfm_rejects_the_station_two_channels_away)
{
    // Stations 400 kHz apart, the unwanted one 30 dB stronger.
    const double        fs = 2.4e6;
    std::vector<double> scene(2 * static_cast<size_t>(0.5 * fs), 0.0);
    add_fm_tones(scene, fs, -300e3, -40.0, {{1000.0, 50e3 * pre_emphasis(1000.0)}});
    add_fm_tones(scene, fs, -300e3 + 400e3, -10.0, {{1700.0, 50e3 * pre_emphasis(1700.0)}});
    const std::vector<int16_t> iq = digitise(scene, 0.5);

    FmDemodulator demod(fs, kAudioRate);
    demod.set_channel(-300e3, 200e3);
    const std::vector<float> audio = demodulate(demod, iq, 16384);
    CHECK(audio.size() > 23'000);
    if (audio.size() > 23'000) {
        CHECK_NEAR(measure_tone(audio, 9'600, 12'000, 1000.0).amplitude, 50.0 / 75.0, 0.02);
        CHECK(measure_tone(audio, 9'600, 12'000, 1700.0).amplitude < 0.005);
    }
    CHECK_NEAR(demod.stats().level_dbfs, -40.0, 0.5);
}

TEST(fm_switches_between_narrow_and_wide)
{
    // One demodulator, one stream with a voice channel and a broadcast station.
    const double        fs = 2.4e6;
    std::vector<double> scene(2 * static_cast<size_t>(0.4 * fs), 0.0);
    add_fm(scene, fs, -600e3, -30.0, 1000.0, 2000.0);
    add_fm_tones(scene, fs, 400e3, -30.0, {{700.0, 60e3 * pre_emphasis(700.0)}});
    const std::vector<int16_t> iq = digitise(scene, 1.0);

    FmDemodulator demod(fs, kAudioRate);
    for (const bool wide : {false, true, false, true}) {
        if (wide)
            demod.set_channel(400e3, 200e3);
        else
            demod.set_channel(-600e3, 12.5e3);
        CHECK(demod.mode() == (wide ? FmMode::Wide : FmMode::Narrow));
        CHECK_EQ(demod.decimation(), wide ? 2 : 24);
        const std::vector<float> audio = demodulate(demod, iq, 16384);
        CHECK(audio.size() > 18'000);
        if (audio.size() < 18'000)
            continue;
        if (wide)
            CHECK_NEAR(measure_tone(audio, 7'200, 9'600, 700.0).amplitude, 60.0 / 75.0, 0.015);
        else
            CHECK_NEAR(measure_tone(audio, 7'200, 9'600, 1000.0).amplitude, 0.4, 0.012);
    }
}

TEST(listener_plays_the_simulators_broadcast_station)
{
    Options options;
    options.source.kind        = "sim";
    options.source.sample_rate = 2.4e6;
    options.source.frequency   = 98e6;
    options.source.gain        = SimSource::kReferenceGainDb;

    Pipeline pipeline(options);
    pipeline.start();
    Listener& listener = pipeline.listener();
    CHECK(listener.supports(FmMode::Wide));

    const SimSource::BroadcastStation station    = SimSource::broadcast_station(2.4e6);
    const double                      station_hz = 98e6 + station.offset_hz;
    const double                      amplitude  = station.deviation_hz / FmDemodulator::kWideDeviation;

    listener.set_volume(1.0f);
    listener.set_bandwidth(200e3);
    listener.set_deemphasis(station.deemphasis_s);
    // Off by 30 kHz, as after a click on the side of the station's hump.
    listener.tune(station_hz + 30e3, 50e3);
    {
        FakeSoundCard card(listener.audio(), 48000.0, 480);
        listener.set_enabled(true);
        wait_ms(1500);

        const ListenStatus status = listener.status();
        CHECK(status.mode == FmMode::Wide);
        CHECK_NEAR(status.bandwidth, 200e3, 1e-6);
        CHECK(status.in_band);
        CHECK(!status.searching);
        CHECK_NEAR(status.frequency, station_hz, 1500.0);
        CHECK(status.squelch_open);
        CHECK_NEAR(status.level_dbfs, station.level_dbfs, 1.0);
        CHECK_NEAR(status.audio_rms, amplitude / std::sqrt(2.0), 0.05);
        CHECK_EQ(status.underruns, uint64_t{0});
        CHECK_EQ(status.dropped_iq, uint64_t{0});

        const std::vector<float> played = card.take();
        CHECK(played.size() > 48'000);
        if (played.size() > 48'000) {
            const std::vector<float> recent(played.end() - 28'800, played.end());
            CHECK(share_of_tune(recent, amplitude, kBroadcastNotes) > 0.7);
            CHECK(measure_tone(recent, 0, 28'800, 19000.0).amplitude < 0.001);  // no pilot
        }

        // Narrowband again, on the voice channel: the mode follows the width.
        listener.set_bandwidth(12.5e3);
        listener.tune(98e6 + SimSource::voice_channel(2.4e6).offset_hz);
        wait_ms(600);
        CHECK(listener.status().mode == FmMode::Narrow);
        CHECK_NEAR(listener.status().level_dbfs, SimSource::voice_channel(2.4e6).level_dbfs, 1.0);
        listener.set_enabled(false);
    }
    const AcquisitionStats stats = pipeline.acquisition().stats();
    CHECK_EQ(stats.lost_samples, uint64_t{0});
    CHECK_EQ(stats.ring_dropped, uint64_t{0});
    pipeline.stop();
}
