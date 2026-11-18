// Receives P25 Phase 1 (C4FM, TIA-102.BAAA): finds the frames in the
// instantaneous frequency of an FM channel, reads their network ID, follows
// voice calls - who calls whom, whether encrypted - and passes on their IMBE
// voice frames.
//
//   frequency ──Fsk4Front──▶ frame sync ──▶ symbol clock ──▶ NID (NAC, frame type)
//                                                            │
//      LDU1: 9 voice frames, link control (talkgroup, source) ◀┤
//      LDU2: 9 voice frames, encryption sync (algorithm)       ◀┤
//      header, terminators: a call starts or ends              ◀┤
//      trunking signalling: a control channel                  ◀┘
//
// Each frame starts with a frame sync, which sets the symbol clock; the clock
// then runs on into the next frame, so that a frame whose sync is damaged is
// still read.  Not decoded: the header's own contents, most trunking
// messages, and packet data.  Phase 2 (TDMA) is P25Phase2Decoder's.
//
// Not thread safe: one thread calls everything.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "dsp/fsk4_front.hpp"
#include "dsp/p25_coding.hpp"
#include "dsp/p25_trunking.hpp"

namespace spectro {

struct P25Call {
    bool     active    = false;  // in progress
    bool     voice     = false;  // voice frames are arriving
    bool     encrypted = false;
    bool     ids_known = false;  // destination and source below are valid
    bool     group     = true;   // to a talkgroup, else to one radio
    uint32_t destination = 0;
    uint32_t source      = 0;
};

struct P25Status {
    bool     present  = false;  // P25 decoded within the last second
    int      nac      = -1;     // network access code, 12 bits; -1 = not known yet
    int      duid     = -1;     // type of the last frame (p25::Duid); -1 = none yet
    bool     control  = false;  // trunking signalling: a control channel
    P25Site  site;              // what a control channel says about its system
    std::vector<P25Grant> grants;  // the voice calls it granted, most recent first
    uint64_t tsbks    = 0;      // trunking blocks decoded
    double   time     = 0.0;    // seconds of signal seen: the clock of the grants' times
    P25Call  call;              // the call in progress; after it, the last one
    float    level_hz = 0.0f;   // deviation of the outer symbols
    uint64_t frames       = 0;  // frames decoded
    uint64_t voice_frames = 0;  // 20 ms voice frames passed on
};

struct P25Voice {
    p25::ImbeFrame frame;
    bool           first = false;  // the first frame of a call: start the vocoder afresh
};

// "Group call to TG 100 from 2001", "Encrypted call", "Control channel", "Idle", ...
std::string describe_p25(const P25Status& status);
// "TG 100" or "2001": who the call is to, if known.
std::string p25_destination(const P25Call& call);

class P25Decoder {
public:
    static constexpr double kSymbolRate = Fsk4Front::kSymbolRate;

    explicit P25Decoder(double sample_rate);

    void reset();

    // Instantaneous frequency in Hz, `sample_rate` samples per second.  The
    // voice frames of a call that is not encrypted are appended to `voice`.
    void process(const float* frequency, size_t count, double sample_rate, std::vector<P25Voice>& voice);

    const P25Status& status() const { return status_; }

private:
    void found_sync(const Fsk4Front::Sync& sync);
    void anchor(const Fsk4Front::Sync& sync);
    void run_clock(std::vector<P25Voice>& voice);
    bool read_nid();
    bool read_tsbk();
    void decode_frame(std::vector<P25Voice>& voice);
    void voice_frames(const std::vector<uint8_t>& bits, std::vector<P25Voice>& voice);
    void end_call();
    double seconds_since(uint64_t sample) const;

    Fsk4Front front_;
    double    sps_;
    double    rate_;

    // Symbol clock
    bool                 locked_    = false;
    double               next_time_ = 0.0;  // of the next symbol to take
    size_t               position_  = 0;    // of that symbol in the frame, status symbols counted
    size_t               length_    = 0;    // of this frame, in dibits; 0 until its NID is read
    uint8_t              duid_      = 0;
    double               level_ = 0.0, dc_ = 0.0;
    std::vector<uint8_t> dibits_;
    std::vector<uint8_t> bits_;  // of this frame, without its status symbols
    size_t               tsbks_read_ = 0;
    double               timing_error_   = 0.0;  // summed over a block of symbols
    double               previous_value_ = 0.0;
    size_t               timing_count_   = 0;

    uint64_t last_valid_   = 0;  // sample: the last frame whose NID decoded
    uint64_t last_voice_   = 0;  // sample: the last LDU or header
    uint64_t last_control_ = 0;  // sample: the last trunking block
    bool     valid_seen_   = false;
    bool     started_output_ = false;

    P25Status status_;
};

}  // namespace spectro
