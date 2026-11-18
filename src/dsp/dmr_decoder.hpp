// Receives DMR (ETSI TS 102 361): finds the bursts in the instantaneous
// frequency of an FM channel, decodes them, follows the calls on both time
// slots, and passes on the voice frames of one call to be heard.
//
//   frequency ──matched filter──▶ sync search ──▶ symbol clock ──▶ 144-symbol frames
//                                                                      │
//         CACH: which slot ◀───────────────────────────────────────────┤
//         data burst: slot type, voice header / terminator (who calls whom)
//         voice burst: three 20 ms AMBE+2 frames, EMB (encrypted or not),
//                      fragments of the link control
//
// The symbol clock starts at a sync pattern and then runs on its own,
// corrected by each sync pattern that comes along and by the timing of the
// symbols themselves, so that voice bursts B-F, which have none, are found
// too.  Repeaters (base stations) and radios talking directly are both
// understood.
//
// Not thread safe: one thread calls everything.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "dsp/dmr_coding.hpp"
#include "dsp/fsk4_front.hpp"

namespace spectro {

// A call on one time slot.
struct DmrCall {
    bool     active    = false;  // in progress
    bool     voice     = false;  // voice frames are arriving
    bool     encrypted = false;
    bool     ids_known = false;  // destination and source below are valid
    bool     group     = true;   // to a talkgroup, else to one radio
    uint32_t destination = 0;
    uint32_t source      = 0;
};

struct DmrStatus {
    bool     present    = false;  // DMR decoded within the last second
    bool     repeater   = false;  // a base station's downlink, else radios directly
    int      color_code = -1;     // 0-15, -1 = not known yet
    DmrCall  slot[2];             // time slots 1 and 2; after a call, the last one
    int      playing    = -1;     // the slot whose voice is passed on, -1 = none
    int      data_type[2] = {-1, -1};  // of the last data burst on each slot (dmr::DataType)
    float    level_hz   = 0.0f;   // deviation of the outer symbols
    uint64_t frames       = 0;    // 30 ms frames taken in
    uint64_t data_bursts  = 0;    // that decoded as data
    uint64_t voice_bursts = 0;    // and as voice
    uint64_t voice_frames = 0;    // 20 ms voice frames passed on
};

// "Group call to TG 91 from 3100123", "Encrypted call", "Idle", "Control (CSBK)", ...
std::string describe_dmr_slot(const DmrStatus& status, int slot);
// "TG 91" or "3100123": who the call on a slot is to, if known.
std::string dmr_destination(const DmrCall& call);

struct DmrVoice {
    dmr::AmbeFrame frame;
    int            slot  = 0;
    bool           first = false;  // the first frame of a call: start the vocoder afresh
};

class DmrDecoder {
public:
    static constexpr double kSymbolRate = 4800.0;

    // sample_rate: of the frequency samples (it may change a little later).
    explicit DmrDecoder(double sample_rate);

    void reset();

    // Instantaneous frequency in Hz, `sample_rate` samples per second.  Voice
    // frames of the call being listened to are appended to `voice`.
    void process(const float* frequency, size_t count, double sample_rate, std::vector<DmrVoice>& voice);

    const DmrStatus& status() const { return status_; }

private:
    struct Slot {
        DmrCall        call;
        uint64_t       last_activity = 0;  // frame
        int            voice_position = -1;  // 0 = burst A ... 5 = F, -1 = not in a superframe
        bool           started_output = false;
        std::array<uint8_t, 128> embedded{};
        int            embedded_have = 0;  // fragments collected
    };

    void   found_sync(const Fsk4Front::Sync& sync);
    void   anchor(const Fsk4Front::Sync& sync);
    void   run_clock(std::vector<DmrVoice>& voice);
    void   decode_frame(std::vector<DmrVoice>& voice);
    bool   voice_burst(int slot, const uint8_t* burst, bool burst_a, int emb_errors, uint8_t emb,
                       std::vector<DmrVoice>& voice);
    bool   data_burst(int slot, const uint8_t* burst);
    void   end_call(int slot);
    void   lose_lock();

    Fsk4Front front_;  // matched filter, history, sync search
    double    sps_;    // samples per symbol, now

    // Symbol clock
    bool     locked_ = false;
    double   next_time_ = 0.0;  // of the next symbol to take
    size_t   position_  = 0;    // of that symbol in the frame
    double   level_ = 0.0, dc_ = 0.0;
    std::array<uint8_t, dmr::kFrameSymbols> dibits_{};
    double   timing_error_ = 0.0;  // summed over the frame
    double   level_error_ = 0.0, dc_error_ = 0.0;
    int      outer_count_ = 0;
    double   previous_value_ = 0.0;

    // Frames
    uint64_t frame_count_      = 0;
    uint64_t last_valid_frame_ = 0;
    uint64_t last_base_frame_  = 0;
    bool     valid_seen_       = false;  // since the clock was set
    int      last_slot_        = 1;
    std::array<Slot, 2> slots_;

    DmrStatus status_;
};

}  // namespace spectro
