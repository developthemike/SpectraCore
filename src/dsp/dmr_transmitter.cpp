#include "dsp/dmr_transmitter.hpp"

#include <algorithm>

namespace spectro {

namespace {

constexpr int kHeaders     = 2;  // voice LC headers before the voice
constexpr int kTerminators = 2;

template <typename T>
void put_bits(uint8_t* out, T value, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        out[i] = static_cast<uint8_t>((value >> (count - 1 - i)) & 1u);
}

}  // namespace

DmrTransmitter::DmrTransmitter(uint8_t color_code) : color_code_(color_code & 0xFu)
{
    // An idle burst's payload is filler, which repeaters make pseudo-random
    // (so that the signal does not sit on one frequency).
    std::array<uint8_t, 96> fill{};
    uint32_t                state = 0x1D4C3B2Au;
    for (uint8_t& b : fill) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        b = static_cast<uint8_t>(state >> 31);
    }
    idle_ = dmr::bptc19696_encode(fill);
}

void DmrTransmitter::start_call(int slot, const dmr::LinkControl& lc, int superframes, VoiceSource voice)
{
    Call& c     = calls_[static_cast<size_t>(slot & 1)];
    c.stage     = Stage::Header;
    c.lc        = lc;
    c.remaining = kHeaders;
    c.voice     = std::move(voice);
    c.embedded  = dmr::encode_embedded_lc(lc);
    c.burst     = -std::max(1, superframes) * 6;  // voice bursts to go, counting up to 0
}

void DmrTransmitter::data_burst(uint8_t* burst, uint8_t type, const std::array<uint8_t, 196>& payload) const
{
    std::copy(payload.begin(), payload.begin() + 98, burst);
    std::copy(payload.begin() + 98, payload.end(), burst + 166);
    const uint32_t slot_type = dmr::golay2087_encode(static_cast<uint8_t>((color_code_ << 4) | type));
    put_bits(burst + 98, slot_type >> 10, 10);
    put_bits(burst + 156, slot_type & 0x3FFu, 10);
    put_bits(burst + 108, dmr::sync_pattern(dmr::Sync::BsData), 48);
}

void DmrTransmitter::voice_burst(uint8_t* burst, Call& call)
{
    // Three voice frames around the middle.
    uint8_t bits[3][72];
    for (auto& b : bits) {
        const dmr::AmbeFrame frame = call.voice ? call.voice() : dmr::ambe_silence();
        dmr::ambe_interleave(frame, b);
    }
    std::copy(bits[0], bits[0] + 72, burst);
    std::copy(bits[1], bits[1] + 36, burst + 72);
    std::copy(bits[1] + 36, bits[1] + 72, burst + 156);
    std::copy(bits[2], bits[2] + 72, burst + 192);

    const int position = ((call.burst % 6) + 6) % 6;  // 0 = A ... 5 = F
    if (position == 0) {
        put_bits(burst + 108, dmr::sync_pattern(dmr::Sync::BsVoice), 48);
        return;
    }
    // B-E carry the link control a quarter at a time; F carries nothing.
    static constexpr uint8_t kLcss[6] = {0, dmr::kFirst, dmr::kContinuation, dmr::kContinuation, dmr::kLast,
                                         dmr::kSingle};
    const uint8_t  lcss    = kLcss[position];
    const uint8_t  privacy = call.lc.encrypted() ? 1 : 0;
    const uint16_t emb     = dmr::qr1676_encode(static_cast<uint8_t>((color_code_ << 3) | (privacy << 2) | lcss));
    put_bits(burst + 108, emb >> 8, 8);
    put_bits(burst + 148, emb & 0xFFu, 8);
    if (position <= 4)
        std::copy(call.embedded.begin() + 32 * (position - 1), call.embedded.begin() + 32 * position, burst + 116);
    else
        std::fill(burst + 116, burst + 148, uint8_t{0});
}

int DmrTransmitter::next_frame(std::array<int8_t, dmr::kFrameSymbols>& symbols)
{
    slot_   = 1 - slot_;
    Call& c = calls_[static_cast<size_t>(slot_)];

    uint8_t bits[2 * dmr::kFrameSymbols] = {};
    uint8_t* cach  = bits;
    uint8_t* burst = bits + dmr::kCachBits;

    // CACH: the TACT says which slot follows (the short link control is left empty).
    const bool    busy = in_call(1 - slot_);
    const uint8_t tact = dmr::hamming743_encode(static_cast<uint8_t>((busy ? 8 : 0) | (slot_ << 2)));
    static constexpr size_t kTact[7] = {0, 4, 8, 12, 14, 18, 22};
    for (size_t i = 0; i < 7; ++i)
        cach[kTact[i]] = static_cast<uint8_t>((tact >> (6 - i)) & 1u);

    switch (c.stage) {
    case Stage::Idle: data_burst(burst, dmr::kIdle, idle_); break;
    case Stage::Header:
        data_burst(burst, dmr::kVoiceLcHeader, dmr::encode_full_lc(c.lc, dmr::kVoiceLcHeaderMask));
        if (--c.remaining == 0)
            c.stage = Stage::Voice;
        break;
    case Stage::Voice:
        voice_burst(burst, c);
        if (++c.burst == 0) {
            c.stage     = Stage::Terminator;
            c.remaining = kTerminators;
        }
        break;
    case Stage::Terminator:
        data_burst(burst, dmr::kTerminatorWithLc, dmr::encode_full_lc(c.lc, dmr::kTerminatorMask));
        if (--c.remaining == 0) {
            c.stage = Stage::Idle;
            c.voice = nullptr;
        }
        break;
    }

    for (size_t i = 0; i < dmr::kFrameSymbols; ++i)
        symbols[i] = static_cast<int8_t>(dmr::symbol_for_dibit(static_cast<uint8_t>((bits[2 * i] << 1) | bits[2 * i + 1])));
    return slot_;
}

int DmrTransmitter::next_symbol()
{
    if (frame_pos_ == dmr::kFrameSymbols) {
        next_frame(frame_);
        frame_pos_ = 0;
    }
    return frame_[frame_pos_++];
}

}  // namespace spectro
