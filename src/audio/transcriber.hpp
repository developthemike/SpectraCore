// Speech to text for the voice decoded from DMR and P25, with whisper.cpp:
// each over the listener hands over (see VoiceOvers) is transcribed on a
// thread of its own and listed with who spoke, when and where.
//
//   audio thread ──VoiceOver──▶ queue ──▶ transcriber thread (Whisper) ──▶ transcripts ──▶ window, log, file
//
// The model - one of OpenAI's Whisper models converted for whisper.cpp, such
// as ggml-base.en.bin - is loaded on that thread too, so starting up is not
// held back.  Overs that come faster than they can be transcribed wait, up
// to kMaxWaiting; beyond that the oldest are dropped.  Everything runs on the
// computer: no audio leaves it.
//
// whisper.cpp is optional (SPECTRO_WITH_WHISPER=ON).  Without it available()
// is false, and a Transcriber only reports that it cannot work.
//
// Thread safe.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "audio/voice_overs.hpp"

namespace spectro {

struct Transcript {
    uint64_t    serial    = 0;    // 1, 2, ... in the order they were transcribed
    double      start     = 0.0;  // when the over began: seconds since 1970
    double      seconds   = 0.0;  // of speech
    double      frequency = 0.0;  // Hz
    std::string system;           // "DMR", "P25" or "P25 Phase 2"
    int         slot      = -1;   // DMR time slot or Phase 2 slot, 0 or 1; -1 for P25 Phase 1
    std::string destination;      // "TG 200", or the radio called; empty if not known
    uint32_t    source    = 0;    // the radio speaking; 0 if not known
    int         part      = 0;    // of a long over
    std::string text;
    float       confidence = 0.0f;  // the mean probability Whisper gave its words, 0 ... 1
};

// "TG 2601 from 21", "TG 2601", "from 21", or "" when neither is known.
std::string transcript_speaker(const Transcript& t);
// "P25 Phase 2 slot 1", "DMR slot 2", "P25".
std::string transcript_channel(const Transcript& t);
// One line: "2026-11-17 21:04:33  154.08750 MHz  P25 Phase 2 slot 1  TG 2601 from 21  (3.2 s): text"
std::string format_transcript(const Transcript& t);

// Whisper's text as shown: without the notes it makes of what is not speech
// ("[BLANK_AUDIO]", "(static)", "*beep*", music signs), nor a word said over
// and over (a loop it got stuck in), its spaces tidied; "" if nothing is left.
std::string clean_transcript(std::string_view text);
// Speech from the vocoders, 8 kHz, as Whisper takes it: 16 kHz, at a steady
// level, and at least a second long.
std::vector<float> speech_for_whisper(const float* speech, size_t count);

struct TranscriberStatus {
    enum class State { Loading, Ready, Failed };
    State       state = State::Loading;
    std::string model;          // the model's file name
    std::string error;          // why it failed
    bool        enabled = true;
    size_t      waiting = 0;    // overs in the queue
    bool        busy    = false;  // transcribing one now
    uint64_t    transcribed = 0;  // overs that gave text
    uint64_t    silent      = 0;  // overs in which nothing was made out
    uint64_t    dropped     = 0;  // overs let go: too many waiting
    double      speed       = 0.0;  // seconds of speech transcribed per second, lately; 0: none yet
};

class Transcriber final : public VoiceOverSink {
public:
    struct Settings {
        std::string model;            // a Whisper model for whisper.cpp (ggml .bin file)
        std::string language = "en";  // of the speech; "auto" to find out each time
        std::string file;             // also append each transcript here; empty: no file
        int         threads  = 0;     // for Whisper; 0: half the CPU's, up to 8
    };

    static constexpr size_t kMaxWaiting = 16;
    static constexpr size_t kKept       = 500;     // transcripts kept for since()
    static constexpr float  kQuietRms   = 0.003f;  // quieter speech is taken for silence (-50 dBFS)

    // Whether this build can transcribe (built with whisper.cpp).
    static bool available();

    // Starts the thread, which loads the model.
    explicit Transcriber(Settings settings);
    ~Transcriber() override;
    Transcriber(const Transcriber&)            = delete;
    Transcriber& operator=(const Transcriber&) = delete;

    // Off: overs are let go, untranscribed.
    void set_enabled(bool enabled);

    // VoiceOverSink: from any thread; never waits for a transcription.
    void voice_over(VoiceOver&& over) override;

    TranscriberStatus status() const;
    // The transcripts after `serial`, oldest first (only the last kKept).
    std::vector<Transcript> since(uint64_t serial) const;
    // Waits until nothing is left to transcribe, up to `timeout` seconds; false if there still is.
    bool wait_idle(double timeout) const;

private:
    struct Engine;

    void run();
    void finish(const VoiceOver& over, std::string text, float confidence);

    const Settings settings_;
    std::FILE*     file_ = nullptr;

    mutable std::mutex              mutex_;
    mutable std::condition_variable wake_;  // something to transcribe, or stop
    mutable std::condition_variable idle_;  // the queue has run empty
    std::deque<VoiceOver>           queue_;
    std::deque<Transcript>          done_;
    TranscriberStatus               status_;
    uint64_t                        serial_ = 0;
    bool                            stop_   = false;
    std::atomic<bool>               abort_{false};  // stop_, for Whisper to see mid-transcription

    std::unique_ptr<Engine> engine_;
    std::thread             thread_;
};

}  // namespace spectro
