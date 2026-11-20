// What a P25 trunked system's control channel says (TIA-102.AABC): who the
// system is, its channel plans - how a channel number becomes a frequency,
// and whether its voice channels are TDMA (Phase 2) - and the voice calls it
// grants, on which channel.
//
// Only the messages that matter for listening are read; the rest is ignored.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "dsp/p25_coding.hpp"

namespace spectro {

// One of a site's 16 channel plans ("identifiers").
struct P25ChannelPlan {
    bool   known      = false;
    double base_hz    = 0.0;  // channel 0
    double spacing_hz = 0.0;
    double tx_offset_hz = 0.0;  // of the radios' transmit frequency
    int    slots      = 1;      // per carrier: 1 FDMA (Phase 1); 2 (or 4) TDMA (Phase 2)
};

struct P25Site {
    int  wacn   = -1;  // wide area communications network, 20 bits; -1 = not heard yet
    int  system = -1;  // 12 bits
    int  rfss   = -1;
    int  site   = -1;
    double control_hz = 0.0;  // the control channel, as the site announces it
    std::array<P25ChannelPlan, 16> plans{};

    bool phase1() const;  // an FDMA channel plan: voice in Phase 1
    bool phase2() const;  // a TDMA channel plan: voice in Phase 2
    // "Phase 1", "Phase 2", "Phase 1 and 2"; empty while no plan is known.
    std::string voice_phases() const;
    // The frequency of a channel (4-bit plan, 12-bit number), 0 if its plan
    // is not known yet; and its TDMA slot, -1 for FDMA.
    double frequency(uint16_t channel, int* slot = nullptr) const;
};

// A voice call the control channel has granted a channel.
struct P25Grant {
    uint16_t channel     = 0;
    double   frequency   = 0.0;  // Hz, 0 = the channel's plan not heard yet
    int      slot        = -1;   // TDMA slot (Phase 2), -1 = FDMA (Phase 1)
    bool     group       = true;
    uint32_t destination = 0;    // talkgroup, or radio
    uint32_t source      = 0;    // 0 = not given
    bool     encrypted   = false;
    bool     emergency   = false;
    double   time        = 0.0;  // seconds, when last granted
};

// "TG 100", or a radio's ID.
std::string p25_grant_destination(const P25Grant& grant);

// Applies one trunking signalling block: updates `site`, and records the
// grant it makes, if any, in `grants` (most recent first, at most
// kMaxGrants, one per destination).  Returns true if it was understood.
constexpr size_t kMaxGrants = 8;
bool apply_tsbk(const p25::Tsbk& block, P25Site& site, std::vector<P25Grant>& grants, double now);

// Blocks as a control channel sends them (for the simulator and tests); the
// CRC is filled in when they are encoded.
namespace tsbk {
p25::Tsbk network_status(uint32_t wacn, uint16_t system, uint16_t channel);
p25::Tsbk rfss_status(uint16_t system, uint8_t rfss, uint8_t site, uint16_t channel);
// 700/800/900 MHz channel plan (FDMA), or VHF/UHF.
p25::Tsbk channel_plan(uint8_t id, double base_hz, double spacing_hz, double tx_offset_hz);
p25::Tsbk channel_plan_vhf_uhf(uint8_t id, double base_hz, double spacing_hz, double tx_offset_hz);
// A TDMA plan: `channel_type` 3 = two slots per carrier.
p25::Tsbk channel_plan_tdma(uint8_t id, uint8_t channel_type, double base_hz, double spacing_hz, double tx_offset_hz);
p25::Tsbk group_grant(uint16_t channel, uint16_t talkgroup, uint32_t source, uint8_t options = 0);
p25::Tsbk group_grant_update(uint16_t channel1, uint16_t talkgroup1, uint16_t channel2, uint16_t talkgroup2);
p25::Tsbk unit_grant(uint16_t channel, uint32_t target, uint32_t source);
// Marks the last block of a TSDU.
p25::Tsbk last(p25::Tsbk block);
}  // namespace tsbk

}  // namespace spectro
