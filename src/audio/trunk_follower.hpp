// Following a P25 trunked system's calls: listening to its control channel,
// tune to the voice channel of each call it grants - and, for Phase 2, the
// call's slot on it - and back to the control channel when the call is over.
//
//   control channel ──grant (TG 200, channel 2-00F)──▶ voice channel, slot 2 ──call ends──▶ control channel
//
// The follower only decides; it reads what the listener makes of the channel
// it is on, and says where to tune next (update()).  Calls are followed one at
// a time: the most recently granted that can be listened to - not encrypted,
// and on a channel whose frequency the control channel has given.
//
// Given a list of control channels it can also hunt for the system (hunt()):
// try each in turn until one is heard to be a control channel, follow that
// one's calls, and when it is lost - it goes quiet, or the site moves its
// control channel to another of its frequencies - try them again, starting
// with the next:
//
//   try 153.3375 ──nothing──▶ try 153.4375 ──control channel──▶ follow its calls ──lost──▶ try 154.2375 ...
//
// Tuning anywhere else by hand takes the follower off the system, and ends a
// hunt; it picks up again on whichever control channel the listener comes to.
//
// Not thread safe: one thread (the user interface's) calls everything.
#pragma once

#include <limits>
#include <string>
#include <vector>

#include "dsp/p25_decoder.hpp"
#include "dsp/p25_phase2_decoder.hpp"
#include "dsp/p25_trunking.hpp"

namespace spectro {

// Where to tune the listener, if anywhere.
struct FollowAction {
    bool   tune      = false;
    double frequency = 0.0;  // Hz
    int    slot      = -1;   // P25 Phase 2: the slot to hear, 0 or 1; -1 = whichever speaks
};

class TrunkFollower {
public:
    enum class State {
        Off,      // not following
        Waiting,  // following, but not on a control channel
        Hunting,  // trying the control channels of the list, one after another
        Control,  // on the control channel, waiting for a call
        Voice,    // on a call's voice channel
    };

    static constexpr double kFreshSeconds    = 1.0;  // a grant repeated this recently is a call in progress
    static constexpr double kNoCallSeconds   = 2.0;  // on a voice channel without finding the call: back
    static constexpr double kHangSeconds     = 1.0;  // after a call, for a reply on the same channel
    static constexpr double kSettleSeconds   = 1.0;  // for a retune to show in what the listener reports
    static constexpr double kAfterCallSeconds = 1.0; // a call heard ending is not followed again for this long
    static constexpr double kNotFoundSeconds = 5.0;  // nor one not found, for this long
    // Hunting: how long to listen on each control channel of the list (a
    // control channel sends all the time, and is recognised within a fraction
    // of a second), and how long the control channel may go unheard - after
    // the decoder's own second without it - before it counts as lost.
    static constexpr double kTrySeconds      = 1.5;
    static constexpr double kLostSeconds     = 2.0;

    void set_enabled(bool enabled);
    bool enabled() const { return state_ != State::Off; }

    // The control channels to hunt through, in Hz, in the order to try them.
    void                       set_control_channels(std::vector<double> hz);
    const std::vector<double>& control_channels() const { return channels_; }

    // Hunting, with a list of control channels: turns following on; while not
    // on a control channel, tries those of the list in turn until one is
    // heard; and once the control channel is lost, tries them again from the
    // next.  Returns where to tune first, if anywhere: nowhere if the listener,
    // at `frequency` (NaN: not known), is on the follower's control channel or
    // call already.  Tuning elsewhere by hand, stop_hunting() or turning
    // following off ends it.
    FollowAction hunt(double frequency, double now);
    void         stop_hunting();
    bool         hunting() const { return hunting_; }
    // Hunting: moves on to the next control channel of the list.
    FollowAction skip(double now);

    // Once per frame: where the listener is settled (NaN while it is not -
    // searching, out of band, not listening), what it decodes there, and the
    // time in seconds.
    FollowAction update(double frequency, const P25Status& p25, const P25Phase2Status& p25p2, double now);

    State           state() const { return state_; }
    double          control_frequency() const { return control_hz_; }
    double          voice_frequency() const { return target_hz_; }
    double          trying_frequency() const { return trying_hz_; }  // the control channel tried (State::Hunting)
    // Which of the list the control channel tried or followed is; -1 if none.
    int             listed_index() const { return listed_; }
    const P25Grant& call() const { return call_; }  // the call followed (State::Voice)
    // "Following TG 200 on 152.31250 MHz, slot 2", "Waiting for a call on 153.43750 MHz", ...
    std::string describe() const;

private:
    struct Recent {
        uint32_t destination;
        uint16_t channel;
        double   until;
    };

    bool         follows(const P25Grant& grant, const P25Status& p25, double now) const;
    FollowAction back_to_control(double now, bool call_heard);
    FollowAction try_next(double now);
    void         found_control(double frequency, double now);
    void         tuned_away();
    int          list_index(double hz) const;

    State               state_       = State::Off;
    std::vector<double> channels_;           // the control channels to hunt through
    bool                hunting_     = false;
    int                 listed_      = -1;   // into channels_
    double              trying_hz_   = 0.0;  // the control channel tried
    double              arrived_at_  = std::numeric_limits<double>::quiet_NaN();  // when the listener got there
    double              control_hz_  = 0.0;  // the control channel, where the listener found it
    double              control_seen_ = 0.0; // when it was last heard to be one
    double              target_hz_   = 0.0;  // the voice channel tuned to
    double              tuned_at_    = 0.0;  // when the last retune was asked for
    P25Grant            call_;
    bool                call_heard_  = false;
    double              last_heard_  = 0.0;
    std::vector<Recent> recent_;             // calls followed lately, not to go back for
};

}  // namespace spectro
