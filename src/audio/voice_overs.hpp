// Cutting the decoded digital voice into overs - one radio's transmission,
// from its first voice frame to its last - for what wants each one whole:
// speech to text (see Transcriber).
//
//   vocoder speech ──push()──▶ VoiceOvers ──an over ends──▶ VoiceOverSink
//
// There are five streams of speech: DMR's two time slots, P25 Phase 1 and
// P25 Phase 2's two slots.  An over ends when the next call starts on its
// stream, when another radio is heard speaking on it, when no speech has come
// for kGapSeconds, or when the channel changes.  One that goes on for
// kMaxSeconds is handed over in parts.  Overs shorter than kMinSeconds - a
// press of the push-to-talk button - are let go.
//
// Not thread safe: the listener's audio thread calls everything, and the sink
// is called from that thread too (it must not keep it waiting).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spectro {

struct VoiceOver {
    double             start     = 0.0;  // when it began: seconds since 1970 (wall clock)
    double             frequency = 0.0;  // Hz
    std::string        system;           // "DMR", "P25" or "P25 Phase 2"
    int                slot      = -1;   // DMR time slot or Phase 2 slot, 0 or 1; -1 for P25 Phase 1
    std::string        destination;      // "TG 200", or the radio called; empty if not known
    uint32_t           source    = 0;    // the radio speaking; 0 if not known
    int                part      = 0;    // a long over's parts after the first: 1, 2, ...
    std::vector<float> speech;           // at VoiceOvers::kSampleRate, -1 ... 1

    double seconds() const;
};

class VoiceOverSink {
public:
    virtual ~VoiceOverSink() = default;
    virtual void voice_over(VoiceOver&& over) = 0;
};

class VoiceOvers {
public:
    enum Stream { DmrSlot1, DmrSlot2, P25Phase1, P25Phase2Slot1, P25Phase2Slot2, kStreams };

    static constexpr double kSampleRate = 8000.0;  // the vocoders'
    static constexpr double kGapSeconds = 0.6;     // without speech: the over has ended
    static constexpr double kMaxSeconds = 28.0;    // Whisper hears 30 s at a time
    static constexpr double kMinSeconds = 0.5;

    // With no sink nothing is kept.  Changing it lets go of the overs under way.
    void           set_sink(VoiceOverSink* sink);
    VoiceOverSink* sink() const { return sink_; }

    // The channel the speech comes from (Hz).  Another one ends every over.
    void set_channel(double frequency);
    // Who is speaking on a stream, as far as its decoder knows - before the
    // stream's speech, and as it becomes known: the over under way takes it,
    // and a different caller than its own ends it.
    void identify(Stream stream, const std::string& destination, uint32_t source);
    // Speech of a stream.  `first`: the first frame of a call.
    void push(Stream stream, const float* speech, size_t count, bool first);
    // The time now, in seconds of signal (any clock that goes on): ends the
    // overs that have gone quiet.
    void advance(double now);
    // Ends every over now.
    void flush();

    bool open(Stream stream) const { return streams_[stream].open; }

private:
    struct State {
        bool        open = false;
        double      last = 0.0;  // when its speech last came
        std::string destination;
        uint32_t    source = 0;
        VoiceOver   over;
    };

    void end(Stream stream);
    void begin(Stream stream, int part, double start);

    VoiceOverSink* sink_      = nullptr;
    double         frequency_ = 0.0;
    double         now_       = 0.0;
    State          streams_[kStreams];
};

}  // namespace spectro
