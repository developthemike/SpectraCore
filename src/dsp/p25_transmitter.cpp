#include "dsp/p25_transmitter.hpp"

#include <algorithm>

namespace spectro {

namespace {

constexpr int     kTerminators = 2;
constexpr uint8_t kStatus      = 3;  // status symbols: "inbound channel idle"

template <typename T>
void put_bits(std::vector<uint8_t>& out, T value, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        out.push_back(static_cast<uint8_t>((value >> (count - 1 - i)) & 1u));
}

int symbol_for(uint8_t dibit)
{
    static constexpr int kSymbols[4] = {+1, +3, -1, -3};
    return kSymbols[dibit & 3u];
}

}  // namespace

P25Transmitter::P25Transmitter(uint16_t nac) : nac_(nac & 0xFFFu) {}

void P25Transmitter::start_call(const p25::LinkControl& lc, int ldus, VoiceSource voice)
{
    lc_            = lc;
    voice_         = std::move(voice);
    ldus_left_     = std::max(2, ldus);
    header_        = true;
    terminators_   = kTerminators;
    next_is_ldu1_  = true;
}

void P25Transmitter::control(BlockSource blocks, int blocks_per_tsdu)
{
    blocks_          = std::move(blocks);
    blocks_per_tsdu_ = std::clamp(blocks_per_tsdu, 1, 3);
}

// Frame sync, NID, then the payload, with a status symbol after every 35.
// The frame is as long as its type says, or holds `data_bits` (then up to
// and with the status symbol that follows them, if one is due).
void P25Transmitter::frame(uint8_t duid, const std::vector<uint8_t>& payload, size_t data_bits)
{
    std::vector<uint8_t> bits;
    put_bits(bits, p25::kFrameSync, 48);
    put_bits(bits, p25::nid_encode(nac_, duid), 64);
    bits.insert(bits.end(), payload.begin(), payload.end());

    size_t length = p25::frame_dibits(duid);
    if (data_bits > 0) {
        length = 0;
        for (size_t data = 0; data < data_bits / 2; ++length)
            data += p25::is_status_dibit(length) ? 0 : 1;
        if (p25::is_status_dibit(length))
            ++length;
    }
    frame_.clear();
    size_t b = 0;
    for (size_t d = 0; d < length; ++d) {
        uint8_t dibit = kStatus;
        if (!p25::is_status_dibit(d)) {
            const uint8_t hi = b < bits.size() ? bits[b] : 0;  // past the payload: null bits
            const uint8_t lo = b + 1 < bits.size() ? bits[b + 1] : 0;
            dibit            = static_cast<uint8_t>((hi << 1) | lo);
            b += 2;
        }
        frame_.push_back(static_cast<int8_t>(symbol_for(dibit)));
    }
    pos_ = 0;
}

void P25Transmitter::build_next_frame()
{
    if (header_) {
        // The header's contents (Golay and Reed-Solomon coded) are left as
        // zeros: the decoder only notes that a call starts.
        frame(p25::kHdu, {});
        header_ = false;
        return;
    }
    if (ldus_left_ > 0) {
        const bool ldu1 = next_is_ldu1_;
        std::vector<uint8_t> payload(p25::kLduBits - 112, 0);
        const auto put = [&](size_t at, const uint8_t* src, size_t count) {
            std::copy(src, src + count, payload.begin() + static_cast<long>(at - 112));
        };
        for (const size_t start : p25::kVoiceStart) {
            const p25::ImbeFrame f = voice_ ? voice_() : p25::ImbeFrame{};
            uint8_t              v[144];
            p25::imbe_interleave(f, v);
            put(start, v, 144);
        }
        std::array<uint8_t, 240> link{};
        if (ldu1) {
            link = p25::encode_link_control(lc_);
        } else {
            p25::EncryptionSync es;
            if (lc_.encrypted()) {
                es.algorithm = 0x84;  // AES-256
                es.key       = 1;
                es.mi        = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x00};
            }
            link = p25::encode_encryption_sync(es);
        }
        for (size_t i = 0; i < 6; ++i)
            put(p25::kLinkStart[i], link.data() + 40 * i, 40);
        // Low-speed data: zeros, a codeword of its code.
        frame(ldu1 ? p25::kLdu1 : p25::kLdu2, payload);
        next_is_ldu1_ = !ldu1;
        --ldus_left_;
        return;
    }
    if (terminators_ > 0) {
        frame(p25::kTdu, std::vector<uint8_t>(28, 0));
        --terminators_;
        if (terminators_ == 0)
            voice_ = nullptr;
        return;
    }
    if (blocks_) {
        std::vector<uint8_t> payload;
        for (int i = 0; i < blocks_per_tsdu_; ++i) {
            p25::Tsbk block = blocks_();
            block[0]        = static_cast<uint8_t>((block[0] & 0x7Fu) | (i + 1 == blocks_per_tsdu_ ? 0x80u : 0u));
            const auto bits = p25::encode_tsbk(block);
            payload.insert(payload.end(), bits.begin(), bits.end());
        }
        frame(p25::kTsdu, payload, 112 + payload.size());
        return;
    }
    frame_.clear();
    pos_ = 0;
}

bool P25Transmitter::next_symbol(int& symbol)
{
    if (pos_ >= frame_.size()) {
        if (!header_ && ldus_left_ == 0 && terminators_ == 0 && !blocks_) {
            frame_.clear();
            pos_ = 0;
            return false;
        }
        build_next_frame();
        if (frame_.empty())
            return false;
    }
    symbol = frame_[pos_++];
    return true;
}

}  // namespace spectro
