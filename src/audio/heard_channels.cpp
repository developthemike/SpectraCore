#include "audio/heard_channels.hpp"

#include <algorithm>
#include <cmath>

namespace spectro {

void HeardChannels::update(double frequency, const DmrStatus& dmr, const P25Status& p25, double now,
                           const P25Phase2Status& p25p2)
{
    // Another channel, or none: the calls being followed are not there.
    const bool decoded = dmr.present || p25.present || p25p2.present;
    if (std::isnan(frequency) || !decoded || std::fabs(frequency - at_) > 1e3) {
        for (int k = 0; k < kCalls; ++k) {
            in_call_[k] = false;
            source_[k]  = 0;
        }
        at_ = std::isnan(frequency) ? 0.0 : frequency;
    }
    if (std::isnan(frequency) || !decoded)
        return;

    // The two DMR slots, P25, and the two P25 Phase 2 slots.
    struct Call {
        bool        voice;
        bool        ids_known;
        uint32_t    source;
        bool        encrypted;
        std::string destination, description;
    };
    const auto phase2 = [&](int s) {
        const P25Call& c = p25p2.slot[s];
        return Call{p25p2.present && c.active && c.voice, c.ids_known, c.source, c.encrypted, p25_destination(c),
                    describe_p25_phase2_slot(p25p2, s)};
    };
    const Call calls[kCalls] = {
        {dmr.present && dmr.slot[0].active && dmr.slot[0].voice, dmr.slot[0].ids_known, dmr.slot[0].source,
         dmr.slot[0].encrypted, dmr_destination(dmr.slot[0]), describe_dmr_slot(dmr, 0)},
        {dmr.present && dmr.slot[1].active && dmr.slot[1].voice, dmr.slot[1].ids_known, dmr.slot[1].source,
         dmr.slot[1].encrypted, dmr_destination(dmr.slot[1]), describe_dmr_slot(dmr, 1)},
        {p25.present && p25.call.active && p25.call.voice, p25.call.ids_known, p25.call.source, p25.call.encrypted,
         p25_destination(p25.call), describe_p25(p25)},
        phase2(0),
        phase2(1),
    };
    for (int k = 0; k < kCalls; ++k) {
        const Call& call = calls[k];
        if (!call.voice) {
            in_call_[k] = false;
            source_[k]  = 0;
            continue;
        }
        HeardChannel& e = entry(frequency);
        // A call is counted once: when it starts, or when another radio
        // takes over without a gap.  (Its caller may only become known
        // during the call, which does not make it another one.)
        const bool other_caller = call.ids_known && source_[k] != 0 && call.source != source_[k];
        const bool new_call     = !in_call_[k] || other_caller;
        if (new_call)
            ++e.calls;
        in_call_[k] = true;
        if (call.ids_known)
            source_[k] = call.source;

        e.last_heard = now;
        e.encrypted  = call.encrypted;
        e.last_call  = call.description;
        if (k < 2) {
            e.system     = "DMR";
            e.color_code = dmr.color_code;
            e.slot       = k;
        } else if (k == 2) {
            e.system = "P25";
            e.nac    = p25.nac;
        } else {
            e.system = "P25 Phase 2";
            e.slot   = k - 3;
            if (p25p2.keyed)
                e.nac = p25p2.system.nac;
        }
        if (call.ids_known)
            e.talkgroup = call.destination;
        else if (new_call)
            e.talkgroup.clear();  // not the last call's
    }
}

HeardChannel& HeardChannels::entry(double frequency)
{
    for (HeardChannel& e : channels_) {
        if (std::fabs(e.frequency - frequency) < kSameChannelHz)
            return e;
    }
    const auto at = std::lower_bound(channels_.begin(), channels_.end(), frequency,
                                     [](const HeardChannel& e, double f) { return e.frequency < f; });
    HeardChannel fresh;
    fresh.frequency = frequency;
    return *channels_.insert(at, fresh);
}

void HeardChannels::clear()
{
    channels_.clear();
    for (int k = 0; k < kCalls; ++k) {
        in_call_[k] = false;
        source_[k]  = 0;
    }
}

}  // namespace spectro
