// The channels DMR or P25 voice has been heard on, to go back to them: what
// the listener decodes is looked at every so often, and each channel where a
// voice call came through gets an entry - when it was last heard, how many
// calls, and who the last one was between.
//
// Not thread safe: one thread (the user interface's) calls everything.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dsp/dmr_decoder.hpp"
#include "dsp/p25_decoder.hpp"
#include "dsp/p25_phase2_decoder.hpp"

namespace spectro {

struct HeardChannel {
    double      frequency  = 0.0;   // Hz, where the receiver settled on the carrier
    double      last_heard = 0.0;   // seconds, on the caller's clock
    int         calls      = 0;
    std::string system;             // "DMR", "P25" or "P25 Phase 2", of the last call
    int         color_code = -1;    // DMR
    int         nac        = -1;    // P25: network access code
    int         slot       = 0;     // DMR, P25 Phase 2: of the last call, 0 = (time) slot 1
    bool        encrypted  = false; // the last call was
    std::string talkgroup;          // of the last call: "TG 1705", or a radio's ID; empty = not known
    std::string last_call;          // e.g. "Group call to TG 1705 from 1001"
};

class HeardChannels {
public:
    // Within this of each other, two frequencies are the same channel.
    static constexpr double kSameChannelHz = 6.25e3;

    // What the receiver makes of `frequency` at time `now` (seconds).  Pass a
    // NaN frequency while it is not settled on a channel.
    void update(double frequency, const DmrStatus& dmr, const P25Status& p25, double now,
                const P25Phase2Status& p25p2 = P25Phase2Status{});

    // By frequency, lowest first.
    const std::vector<HeardChannel>& channels() const { return channels_; }
    void                             clear();

private:
    HeardChannel& entry(double frequency);

    std::vector<HeardChannel> channels_;
    double                    at_         = 0.0;  // the channel the calls below are on
    // Per DMR slot, P25, and P25 Phase 2 slot: a call already counted, and
    // its caller (0 = not known yet).
    static constexpr int      kCalls      = 5;
    bool                      in_call_[kCalls] = {};
    uint32_t                  source_[kCalls]  = {};
};

}  // namespace spectro
