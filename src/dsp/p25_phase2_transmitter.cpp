#include "dsp/p25_phase2_transmitter.hpp"

#include <algorithm>

namespace spectro {

namespace {

int symbol_for(uint8_t dibit)
{
    static constexpr int kSymbols[4] = {+1, +3, -1, -3};
    return kSymbols[dibit & 3u];
}

}  // namespace

P25Phase2Transmitter::P25Phase2Transmitter(p25p2::System system) : system_(system), mask_(p25p2::scrambling(system)) {}

void P25Phase2Transmitter::start_call(int slot, uint16_t talkgroup, uint32_t source, int superframes, VoiceSource voice,
                                      bool encrypted)
{
    Call& c     = calls_[static_cast<size_t>(slot & 1)];
    c.active    = true;
    c.started   = false;
    c.left      = std::max(1, superframes);
    c.talkgroup = talkgroup;
    c.source    = source;
    c.encrypted = encrypted;
    c.voice     = std::move(voice);
}

void P25Phase2Transmitter::build_superframe()
{
    symbols_.clear();
    pos_ = 0;
    if (!calls_[0].active && !calls_[1].active)
        return;
    bool ending[2] = {calls_[0].active && calls_[0].left == 0, calls_[1].active && calls_[1].left == 0};

    for (int t = 0; t < static_cast<int>(p25p2::kSlots); ++t) {
        std::array<uint8_t, p25p2::kSlotDibits> d{};
        // ISCH
        const bool     sync = t == 2 || t == 3 || t == 6 || t == 7 || t == 10 || t == 11;
        const uint64_t isch = sync ? p25p2::kSync : p25p2::isch_encode(p25p2::isch_value(t));
        for (size_t i = 0; i < 20; ++i)
            d[i] = static_cast<uint8_t>((isch >> (38 - 2 * i)) & 3u);

        const int channel = p25p2::kChannelOf[t];
        Call&     c       = calls_[static_cast<size_t>(channel)];
        uint8_t   burst   = p25p2::kFacch;
        if (t >= 10) {
            burst = p25p2::kSacch;
            const auto pdu = c.active && !ending[channel]
                                 ? p25p2::mac_active(false, c.talkgroup, c.source, c.encrypted ? 0x40 : 0x00)
                                 : p25p2::mac_idle(false);
            p25p2::encode_mac(pdu, false, d.data());
        } else {
            const int position = t / 2;  // the channel's 0th ... 4th timeslot of the superframe
            if (!c.active) {
                p25p2::encode_mac(p25p2::mac_idle(true), true, d.data());
            } else if (ending[channel]) {
                const auto pdu = position == 0 ? p25p2::mac_end_ptt(true, system_.nac, c.talkgroup, c.source)
                                               : p25p2::mac_idle(true);
                p25p2::encode_mac(pdu, true, d.data());
            } else if (!c.started && position == 0) {
                p25p2::encode_mac(p25p2::mac_ptt(true, c.talkgroup, c.source, c.encrypted ? 0x84 : 0x80), true, d.data());
                c.started = true;
            } else {
                burst              = position == 4 ? p25p2::k2V : p25p2::k4V;
                const size_t count = burst == p25p2::k4V ? 4 : 2;
                for (size_t f = 0; f < count; ++f) {
                    const dmr::AmbeFrame frame = c.voice ? c.voice() : dmr::ambe_silence();
                    uint8_t              bits[72];
                    dmr::ambe_interleave(frame, bits);
                    for (size_t k = 0; k < 36; ++k)
                        d[p25p2::kVoiceAt[f] + k] = static_cast<uint8_t>((bits[2 * k] << 1) | bits[2 * k + 1]);
                }
            }
        }
        // Scramble all but the ISCH and the DUID, then the DUID.
        const size_t base = static_cast<size_t>(t) * p25p2::kSlotDibits;
        for (size_t i = 20; i < p25p2::kSlotDibits; ++i)
            d[i] = static_cast<uint8_t>(d[i] ^ mask_[base + i - 10]);
        const uint8_t duid = p25p2::duid_codeword(burst);
        for (size_t k = 0; k < 4; ++k)
            d[p25p2::kDuidAt[k]] = static_cast<uint8_t>((duid >> (6 - 2 * k)) & 3u);
        for (uint8_t dibit : d)
            symbols_.push_back(static_cast<int8_t>(symbol_for(dibit)));
    }

    for (int c = 0; c < 2; ++c) {
        Call& call = calls_[static_cast<size_t>(c)];
        if (ending[c])
            call = {};
        else if (call.active)
            --call.left;
    }
}

bool P25Phase2Transmitter::next_symbol(int& symbol)
{
    if (pos_ >= symbols_.size()) {
        build_superframe();
        if (symbols_.empty())
            return false;
    }
    symbol = symbols_[pos_++];
    return true;
}

}  // namespace spectro
