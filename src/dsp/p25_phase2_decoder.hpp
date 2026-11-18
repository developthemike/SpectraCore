// Receives P25 Phase 2 voice channels (TDMA, TIA-102.BBAC): finds the
// timeslots in the instantaneous frequency of an FM channel, works out where
// in the superframe each is, and for each of the two logical channels
// ("slots") follows the call - who calls whom, encrypted or not - and passes
// on the AMBE+2 voice frames of one.
//
//   frequency ──Fsk4Front (6000 Bd)──▶ S-ISCH ──▶ timeslots of 180 dibits ──▶ ISCH: where in the superframe
//                                                     │
//          descrambled with the system's sequence ◀───┤ burst type (DUID)
//          4V / 2V: voice frames      FACCH / SACCH: MAC PDUs (call start, end, who)
//
// The scrambling depends on the system (WACN, system ID, NAC), which its
// control channel says.  Given the systems heard (set_systems()), the decoder
// tries them in turn on the signalling until one checks.  Otherwise it works
// the system out from a signalling burst that came through without errors
// (p25p2::recover_system()), and takes it once another burst agrees; until
// then it can tell there is a Phase 2 call, but not what is said.
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
#include "dsp/p25_decoder.hpp"
#include "dsp/p25_phase2_coding.hpp"

namespace spectro {

struct P25Phase2Status {
    bool     present  = false;  // Phase 2 timeslots found within the last second
    bool     keyed    = false;  // the system is known: voice and signalling can be read
    p25p2::System system;       // which, if keyed
    bool     recovered = false; // keyed, with the system worked out from the channel itself
    P25Call  slot[2];           // the call on each logical channel (after it, the last one)
    int      playing  = -1;     // the slot whose voice is passed on, -1 = none
    float    level_hz = 0.0f;
    uint64_t timeslots    = 0;
    uint64_t macs         = 0;  // signalling PDUs decoded
    uint64_t voice_frames = 0;  // passed on
};

struct P25Phase2Voice {
    dmr::AmbeFrame frame;
    int            slot  = 0;
    bool           first = false;  // the first frame of a call: start the vocoder afresh
};

// "Slot 1: group call to TG 200 from 3001", ...
std::string describe_p25_phase2_slot(const P25Phase2Status& status, int slot);

class P25Phase2Decoder {
public:
    explicit P25Phase2Decoder(double sample_rate);

    void reset();
    // The systems whose scrambling to try (from control channels heard).
    void set_systems(const std::vector<p25p2::System>& systems);
    // Pass on only this slot's voice (0 or 1); -1: the first to speak.
    void set_slot(int slot) { wanted_slot_ = slot; }

    void process(const float* frequency, size_t count, double sample_rate, std::vector<P25Phase2Voice>& voice);

    const P25Phase2Status& status() const { return status_; }

private:
    struct Slot {
        P25Call  call;
        uint64_t last_activity    = 0;  // sample
        bool     started_output   = false;
        bool     encryption_known = false;
        // Voice heard before the signalling said whether the call is encrypted.
        std::vector<P25Phase2Voice> held;
    };
    // A timeslot as received, kept until it can be read.
    struct Timeslot {
        std::array<uint8_t, p25p2::kSlotDibits> dibits{};
        int  number   = -1;     // in the superframe, -1 = not known yet
        uint64_t serial = 0;    // counts the timeslots received
        bool good     = false;  // its ISCH decoded
        bool readable = false;  // good, or just one bad among good ones
    };

    void found_sync(const Fsk4Front::Sync& sync);
    void run_clock(std::vector<P25Phase2Voice>& voice);
    void decode_timeslot(std::vector<P25Phase2Voice>& voice);
    void read_timeslot(const Timeslot& t, std::vector<P25Phase2Voice>& voice);
    void note_unreadable(const Timeslot& t);
    int  burst_type(const Timeslot& t) const;
    bool try_systems(const Timeslot& t);
    void key(const p25p2::System& system, const std::vector<uint8_t>& mask, bool recovered);
    void lose_lock();
    void apply_mac(int channel, const std::vector<uint8_t>& pdu, std::vector<P25Phase2Voice>& voice);
    void output(Slot& s, P25Phase2Voice v, std::vector<P25Phase2Voice>& voice);
    void start_call(int channel);
    void end_call(int channel);
    double seconds_since(uint64_t sample) const;

    Fsk4Front front_;
    double    sps_;
    double    rate_;

    bool     locked_    = false;
    double   next_time_ = 0.0;
    size_t   position_  = 0;  // dibit in the timeslot
    int      timeslot_  = -1; // in the superframe, -1 = not known yet
    int      misses_    = 0;  // timeslots in a row whose ISCH did not decode
    double   level_ = 0.0, dc_ = 0.0;
    std::array<uint8_t, p25p2::kSlotDibits> dibits_{};
    double   timing_error_ = 0.0, previous_value_ = 0.0;
    size_t   timing_count_ = 0;

    std::vector<p25p2::System> systems_;
    std::vector<std::vector<uint8_t>> masks_;  // the scrambling of each
    // Systems worked out from a burst, waiting for another to agree.
    struct Candidate {
        p25p2::System        system;
        std::vector<uint8_t> mask;
        uint64_t             from = 0;  // the timeslot it came from
    };
    std::vector<Candidate> candidates_;
    uint64_t               serial_ = 0;
    std::vector<uint8_t>       mask_;  // the scrambling, if keyed
    // Up to a superframe of timeslots received before the system or where in
    // the superframe was known: read once both are, so a call's first bursts
    // are not lost while the decoder finds its feet.
    std::vector<Timeslot>      pending_;
    int                        wanted_slot_ = -1;
    std::array<Slot, 2>        slots_;
    uint64_t                   last_valid_ = 0;
    bool                       valid_seen_ = false;

    P25Phase2Status status_;
};

}  // namespace spectro
