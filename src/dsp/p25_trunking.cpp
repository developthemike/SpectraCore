#include "dsp/p25_trunking.hpp"

#include <algorithm>
#include <cmath>

namespace spectro {

namespace {

// Opcodes (with the standard manufacturer ID, 0).
constexpr uint8_t kGroupVoiceGrant       = 0x00;
constexpr uint8_t kGroupVoiceGrantUpdate = 0x02;
constexpr uint8_t kGroupVoiceGrantUpdateExplicit = 0x03;
constexpr uint8_t kUnitVoiceGrant        = 0x04;
constexpr uint8_t kUnitVoiceGrantUpdate  = 0x06;
constexpr uint8_t kChannelPlanTdma       = 0x33;
constexpr uint8_t kChannelPlanVhfUhf     = 0x34;
constexpr uint8_t kRfssStatus            = 0x3A;
constexpr uint8_t kNetworkStatus         = 0x3B;
constexpr uint8_t kChannelPlan           = 0x3D;

// Slots per carrier for each TDMA channel type.
constexpr int kTdmaSlots[16] = {1, 1, 1, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

// The block as a 96-bit number, first byte highest, split in two.
struct Bits96 {
    uint64_t high = 0;  // bytes 0-3, as bits 95-64
    uint64_t low  = 0;  // bytes 4-11, as bits 63-0
    explicit Bits96(const p25::Tsbk& b)
    {
        for (size_t i = 0; i < 4; ++i)
            high = (high << 8) | b[i];
        for (size_t i = 4; i < 12; ++i)
            low = (low << 8) | b[i];
    }
    // `count` bits ending at bit `shift` (as OP25 writes it: (tsbk >> shift) & mask).
    uint64_t get(int shift, int count) const
    {
        uint64_t v = 0;
        for (int i = count - 1; i >= 0; --i) {
            const int bit = shift + i;
            const uint64_t b = bit >= 64 ? (high >> (bit - 64)) & 1u : (low >> bit) & 1u;
            v = (v << 1) | b;
        }
        return v;
    }
};

void set_bits(p25::Tsbk& b, int shift, int count, uint64_t value)
{
    for (int i = 0; i < count; ++i) {
        const int    bit  = shift + i;  // 0 = least significant of byte 11
        const size_t byte = static_cast<size_t>(11 - bit / 8);
        const auto   mask = static_cast<uint8_t>(1u << (bit % 8));
        if ((value >> i) & 1u)
            b[byte] = static_cast<uint8_t>(b[byte] | mask);
        else
            b[byte] = static_cast<uint8_t>(b[byte] & ~mask);
    }
}

p25::Tsbk block(uint8_t opcode)
{
    p25::Tsbk b{};
    b[0] = opcode & 0x3Fu;
    return b;
}

void record(std::vector<P25Grant>& grants, P25Grant g)
{
    // One entry per destination, the most recent first.
    grants.erase(std::remove_if(grants.begin(), grants.end(),
                                [&](const P25Grant& o) { return o.group == g.group && o.destination == g.destination; }),
                 grants.end());
    grants.insert(grants.begin(), g);
    if (grants.size() > kMaxGrants)
        grants.resize(kMaxGrants);
}

}  // namespace

bool P25Site::phase1() const
{
    return std::any_of(plans.begin(), plans.end(), [](const P25ChannelPlan& p) { return p.known && p.slots == 1; });
}

bool P25Site::phase2() const
{
    return std::any_of(plans.begin(), plans.end(), [](const P25ChannelPlan& p) { return p.known && p.slots > 1; });
}

std::string P25Site::voice_phases() const
{
    if (phase1() && phase2())
        return "Phase 1 and 2";
    return phase1() ? "Phase 1" : phase2() ? "Phase 2" : "";
}

double P25Site::frequency(uint16_t channel, int* slot) const
{
    const P25ChannelPlan& plan   = plans[(channel >> 12) & 0xFu];
    const unsigned        number = channel & 0xFFFu;
    if (slot)
        *slot = plan.known && plan.slots > 1 ? static_cast<int>(number % static_cast<unsigned>(plan.slots)) : -1;
    if (!plan.known)
        return 0.0;
    return plan.base_hz + plan.spacing_hz * static_cast<double>(number / static_cast<unsigned>(std::max(plan.slots, 1)));
}

std::string p25_grant_destination(const P25Grant& grant)
{
    return (grant.group ? "TG " : "") + std::to_string(grant.destination);
}

bool apply_tsbk(const p25::Tsbk& b, P25Site& site, std::vector<P25Grant>& grants, double now)
{
    const Bits96  t(b);
    const uint8_t opcode = b[0] & 0x3Fu;
    const uint8_t mfid   = b[1];
    if (mfid != 0 && mfid != 1)
        return false;  // a manufacturer's own

    const auto grant = [&](uint16_t channel, bool group, uint32_t destination, uint32_t source, uint8_t options) {
        P25Grant g;
        g.channel     = channel;
        g.frequency   = site.frequency(channel, &g.slot);
        g.group       = group;
        g.destination = destination;
        g.source      = source;
        g.encrypted   = (options & 0x40u) != 0;
        g.emergency   = (options & 0x80u) != 0;
        g.time        = now;
        // An update without a source keeps the one the grant gave.
        for (const P25Grant& o : grants) {
            if (o.group == group && o.destination == destination && source == 0) {
                g.source    = o.source;
                g.encrypted = g.encrypted || o.encrypted;
            }
        }
        record(grants, g);
    };

    switch (opcode) {
    case kGroupVoiceGrant:
        grant(static_cast<uint16_t>(t.get(56, 16)), true, static_cast<uint32_t>(t.get(40, 16)),
              static_cast<uint32_t>(t.get(16, 24)), static_cast<uint8_t>(t.get(72, 8)));
        return true;
    case kGroupVoiceGrantUpdate:
        grant(static_cast<uint16_t>(t.get(64, 16)), true, static_cast<uint32_t>(t.get(48, 16)), 0, 0);
        if (t.get(32, 16) != t.get(64, 16) || t.get(16, 16) != t.get(48, 16))
            grant(static_cast<uint16_t>(t.get(32, 16)), true, static_cast<uint32_t>(t.get(16, 16)), 0, 0);
        return true;
    case kGroupVoiceGrantUpdateExplicit:
        grant(static_cast<uint16_t>(t.get(48, 16)), true, static_cast<uint32_t>(t.get(16, 16)), 0,
              static_cast<uint8_t>(t.get(72, 8)));
        return true;
    case kUnitVoiceGrant:
    case kUnitVoiceGrantUpdate:
        grant(static_cast<uint16_t>(t.get(64, 16)), false, static_cast<uint32_t>(t.get(40, 24)),
              static_cast<uint32_t>(t.get(16, 24)), 0);
        return true;
    case kChannelPlanVhfUhf:
    case kChannelPlan:
    case kChannelPlanTdma: {
        P25ChannelPlan& plan = site.plans[t.get(76, 4)];
        const double    spacing = static_cast<double>(t.get(48, 10)) * 125.0;
        double          offset  = 0.0;
        if (opcode == kChannelPlan) {
            const auto o = t.get(58, 9);
            offset       = static_cast<double>(o & 0xFFu) * 250e3 * ((o >> 8) & 1u ? 1.0 : -1.0);
        } else {
            const auto o = t.get(58, 14);
            offset       = static_cast<double>(o & 0x1FFFu) * spacing * ((o >> 13) & 1u ? 1.0 : -1.0);
        }
        plan.known        = true;
        plan.base_hz      = static_cast<double>(t.get(16, 32)) * 5.0;
        plan.spacing_hz   = spacing;
        plan.tx_offset_hz = offset;
        plan.slots        = opcode == kChannelPlanTdma ? kTdmaSlots[t.get(72, 4)] : 1;
        return true;
    }
    case kRfssStatus:
        site.system = static_cast<int>(t.get(56, 12));
        site.rfss   = static_cast<int>(t.get(48, 8));
        site.site   = static_cast<int>(t.get(40, 8));
        if (const double f = site.frequency(static_cast<uint16_t>(t.get(24, 16))); f > 0.0)
            site.control_hz = f;
        return true;
    case kNetworkStatus:
        site.wacn   = static_cast<int>(t.get(52, 20));
        site.system = static_cast<int>(t.get(40, 12));
        return true;
    default:
        return false;
    }
}

namespace tsbk {

p25::Tsbk network_status(uint32_t wacn, uint16_t system, uint16_t channel)
{
    p25::Tsbk b = block(kNetworkStatus);
    set_bits(b, 52, 20, wacn);
    set_bits(b, 40, 12, system);
    set_bits(b, 24, 16, channel);
    return b;
}

p25::Tsbk rfss_status(uint16_t system, uint8_t rfss, uint8_t site, uint16_t channel)
{
    p25::Tsbk b = block(kRfssStatus);
    set_bits(b, 56, 12, system);
    set_bits(b, 48, 8, rfss);
    set_bits(b, 40, 8, site);
    set_bits(b, 24, 16, channel);
    return b;
}

namespace {

p25::Tsbk plan(uint8_t opcode, uint8_t id, double base_hz, double spacing_hz, double tx_offset_hz)
{
    p25::Tsbk b = block(opcode);
    set_bits(b, 76, 4, id);
    const auto spacing = static_cast<uint64_t>(std::lround(spacing_hz / 125.0));
    set_bits(b, 48, 10, spacing);
    set_bits(b, 16, 32, static_cast<uint64_t>(std::llround(base_hz / 5.0)));
    if (opcode == kChannelPlan) {
        const auto o = static_cast<uint64_t>(std::lround(std::fabs(tx_offset_hz) / 250e3));
        set_bits(b, 58, 9, (tx_offset_hz > 0.0 ? 0x100u : 0u) | (o & 0xFFu));
        set_bits(b, 67, 9, 100);  // bandwidth: 12.5 kHz in 125 Hz units
    } else {
        const auto o = spacing > 0 ? static_cast<uint64_t>(std::lround(std::fabs(tx_offset_hz) / spacing_hz)) : 0;
        set_bits(b, 58, 14, (tx_offset_hz > 0.0 ? 0x2000u : 0u) | (o & 0x1FFFu));
    }
    return b;
}

}  // namespace

p25::Tsbk channel_plan(uint8_t id, double base_hz, double spacing_hz, double tx_offset_hz)
{
    return plan(kChannelPlan, id, base_hz, spacing_hz, tx_offset_hz);
}

p25::Tsbk channel_plan_vhf_uhf(uint8_t id, double base_hz, double spacing_hz, double tx_offset_hz)
{
    p25::Tsbk b = plan(kChannelPlanVhfUhf, id, base_hz, spacing_hz, tx_offset_hz);
    set_bits(b, 72, 4, 5);  // 12.5 kHz
    return b;
}

p25::Tsbk channel_plan_tdma(uint8_t id, uint8_t channel_type, double base_hz, double spacing_hz, double tx_offset_hz)
{
    p25::Tsbk b = plan(kChannelPlanTdma, id, base_hz, spacing_hz, tx_offset_hz);
    set_bits(b, 72, 4, channel_type);
    return b;
}

p25::Tsbk group_grant(uint16_t channel, uint16_t talkgroup, uint32_t source, uint8_t options)
{
    p25::Tsbk b = block(kGroupVoiceGrant);
    set_bits(b, 72, 8, options);
    set_bits(b, 56, 16, channel);
    set_bits(b, 40, 16, talkgroup);
    set_bits(b, 16, 24, source);
    return b;
}

p25::Tsbk group_grant_update(uint16_t channel1, uint16_t talkgroup1, uint16_t channel2, uint16_t talkgroup2)
{
    p25::Tsbk b = block(kGroupVoiceGrantUpdate);
    set_bits(b, 64, 16, channel1);
    set_bits(b, 48, 16, talkgroup1);
    set_bits(b, 32, 16, channel2);
    set_bits(b, 16, 16, talkgroup2);
    return b;
}

p25::Tsbk unit_grant(uint16_t channel, uint32_t target, uint32_t source)
{
    p25::Tsbk b = block(kUnitVoiceGrant);
    set_bits(b, 64, 16, channel);
    set_bits(b, 40, 24, target);
    set_bits(b, 16, 24, source);
    return b;
}

p25::Tsbk last(p25::Tsbk b)
{
    b[0] = static_cast<uint8_t>(b[0] | 0x80u);
    return b;
}

}  // namespace tsbk

}  // namespace spectro
