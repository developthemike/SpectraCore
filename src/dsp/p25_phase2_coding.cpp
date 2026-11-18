#include "dsp/p25_phase2_coding.hpp"

#include <algorithm>
#include <array>
#include <bit>

#include "dsp/p25_coding.hpp"

namespace spectro::p25p2 {

bool is_scrambled(uint8_t burst)
{
    return burst == k4V || burst == k2V || burst == kSacch || burst == kLcch || burst == kFacch;
}

// ---- DUID: (8,4,4) ----------------------------------------------------------------

namespace {

// The parity nibble of each information bit, from the least significant.
constexpr uint8_t kDuidParity[4] = {0x7, 0xE, 0xB, 0xD};

}  // namespace

uint8_t duid_codeword(uint8_t burst)
{
    uint8_t parity = 0;
    for (int k = 0; k < 4; ++k)
        if ((burst >> k) & 1u)
            parity ^= kDuidParity[k];
    return static_cast<uint8_t>(((burst & 0xFu) << 4) | parity);
}

int duid_decode(uint8_t codeword)
{
    int best = 9, second = 9, type = -1;
    for (int v = 0; v < 16; ++v) {
        const int d = std::popcount(static_cast<unsigned>(codeword ^ duid_codeword(static_cast<uint8_t>(v))));
        if (d < best) {
            second = best;
            best   = d;
            type   = v;
        } else if (d < second) {
            second = d;
        }
    }
    return best < second ? type : -1;  // one error mended, two detected
}

// ---- ISCH: (40,9,16), an offset and seven generator rows ----------------------------

namespace {

constexpr uint64_t kIschOffset  = 0x184229D461;
constexpr uint64_t kIschRows[7] = {0x343D8597, 0x58CBAA4E, 0x9DA3A171, 0x9048D9B72, 0x20807F7FF, 0xC00DED18E, 0x100F4B1758};
constexpr int      kMaxIschErrors = 7;

}  // namespace

uint64_t isch_encode(unsigned value)
{
    uint64_t word = kIschOffset;
    for (int k = 0; k < 7; ++k)
        if ((value >> k) & 1u)
            word ^= kIschRows[k];
    return word;
}

int isch_decode(uint64_t word)
{
    word &= 0xFFFFFFFFFFu;
    if (std::popcount(word ^ kSync) <= kMaxIschErrors)
        return -2;
    int best = kMaxIschErrors + 1, value = -1;
    for (unsigned v = 0; v < 128; ++v) {
        const int d = std::popcount(word ^ isch_encode(v));
        if (d < best) {
            best  = d;
            value = static_cast<int>(v);
        }
    }
    return value;
}

// value = channel << 5 | location << 3 | counters; timeslot = location * 4 + channel.
int isch_slot(unsigned value)
{
    const unsigned channel = (value >> 5) & 3u, location = (value >> 3) & 3u;
    return static_cast<int>(location * 4 + channel);
}

unsigned isch_value(int slot)
{
    return static_cast<unsigned>(((slot % 4) << 5) | ((slot / 4) << 3));
}

// ---- Scrambling ----------------------------------------------------------------------

std::vector<uint8_t> scrambling(const System& s)
{
    // The seed, WACN | system | NAC (44 bits), through a fixed matrix: output
    // bit j (from the most significant) is the sum of input bits j, j-4,
    // j-9, j-15, j-20 and j-34.
    const uint64_t seed = (uint64_t{s.wacn & 0xFFFFFu} << 24) | (uint64_t{s.system & 0xFFFu} << 12) | (s.nac & 0xFFFu);
    uint64_t       reg  = 0;
    static constexpr int kTaps[6] = {0, 4, 9, 15, 20, 34};
    for (int j = 0; j < 44; ++j) {
        unsigned bit = 0;
        for (int d : kTaps)
            if (j - d >= 0)
                bit ^= (seed >> (43 - (j - d))) & 1u;
        reg |= uint64_t{bit} << (43 - j);
    }
    // Then six shift registers of 4, 5, 6, 5, 14 and 10 bits, fed back from
    // the first one's top bit, which is the output.
    std::vector<uint8_t> bits(2 * kSlots * kSlotDibits);
    for (uint8_t& out : bits) {
        out = static_cast<uint8_t>((reg >> 43) & 1u);
        uint64_t s1 = (reg >> 40) & 0xF, s2 = (reg >> 35) & 0x1F, s3 = (reg >> 29) & 0x3F, s4 = (reg >> 24) & 0x1F,
                 s5 = (reg >> 10) & 0x3FFF, s6 = reg & 0x3FF;
        const uint64_t c1 = (s1 >> 3) & 1u;
        const uint64_t x1 = c1 ^ ((s2 >> 4) & 1u), x2 = c1 ^ ((s3 >> 5) & 1u), x3 = c1 ^ ((s4 >> 4) & 1u),
                       x4 = c1 ^ ((s5 >> 13) & 1u), x5 = c1 ^ ((s6 >> 9) & 1u);
        s1  = ((s1 << 1) & 0xF) | x1;
        s2  = ((s2 << 1) & 0x1F) | x2;
        s3  = ((s3 << 1) & 0x3F) | x3;
        s4  = ((s4 << 1) & 0x1F) | x4;
        s5  = ((s5 << 1) & 0x3FFF) | x5;
        s6  = ((s6 << 1) & 0x3FF) | c1;
        reg = (s1 << 40) | (s2 << 35) | (s3 << 29) | (s4 << 24) | (s5 << 10) | s6;
    }
    std::vector<uint8_t> dibits(kSlots * kSlotDibits);
    for (size_t i = 0; i < dibits.size(); ++i)
        dibits[i] = static_cast<uint8_t>((bits[2 * i] << 1) | bits[2 * i + 1]);
    return dibits;
}

// ---- CRC-12: x^12 + x^11 + x^7 + x^4 + x^2 + x + 1, inverted -------------------------------

uint16_t crc12(const uint8_t* bits, size_t count)
{
    uint16_t crc = 0;
    for (size_t i = 0; i < count; ++i) {
        const unsigned feedback = ((crc >> 11) & 1u) ^ (bits[i] & 1u);
        crc                     = static_cast<uint16_t>((crc << 1) & 0xFFFu);
        if (feedback)
            crc ^= 0x897u;
    }
    return static_cast<uint16_t>(crc ^ 0xFFFu);
}

// ---- Signalling ------------------------------------------------------------------------

namespace {

// The data dibits of a FACCH and of a SACCH, as runs of a timeslot's dibits.
struct Run {
    size_t first, count;
};
constexpr Run kFacchRuns[4] = {{21, 36}, {58, 31}, {110, 32}, {143, 36}};  // 135 dibits: 45 hexbits
constexpr Run kSacchRuns[3] = {{21, 36}, {58, 84}, {143, 36}};             // 156 dibits: 52 hexbits

// The Reed-Solomon (63,35) codeword: FACCH hexbits 9..53 are sent (0..8
// shortened, 54..62 punctured), SACCH 5..56.
struct Layout {
    size_t first, count, data;  // sent hexbits from `first`; `data` of them carry the PDU and CRC
};
constexpr Layout kFacchLayout = {9, 45, 26};
constexpr Layout kSacchLayout = {5, 52, 30};

template <size_t N>
void gather(const uint8_t* slot, const Run (&runs)[N], std::vector<uint8_t>& bits)
{
    for (const Run& r : runs)
        for (size_t i = 0; i < r.count; ++i) {
            bits.push_back(static_cast<uint8_t>((slot[r.first + i] >> 1) & 1u));
            bits.push_back(static_cast<uint8_t>(slot[r.first + i] & 1u));
        }
}

template <size_t N>
void scatter(const std::vector<uint8_t>& bits, const Run (&runs)[N], uint8_t* slot)
{
    size_t b = 0;
    for (const Run& r : runs)
        for (size_t i = 0; i < r.count; ++i, b += 2)
            slot[r.first + i] = static_cast<uint8_t>((bits[b] << 1) | bits[b + 1]);
}

}  // namespace

bool decode_mac(const uint8_t* slot, bool facch, std::vector<uint8_t>& pdu)
{
    std::vector<uint8_t> bits;
    if (facch)
        gather(slot, kFacchRuns, bits);
    else
        gather(slot, kSacchRuns, bits);
    const Layout& layout = facch ? kFacchLayout : kSacchLayout;

    uint8_t h[63] = {};
    for (size_t i = 0; i < layout.count; ++i) {
        unsigned v = 0;
        for (size_t k = 0; k < 6; ++k)
            v = (v << 1) | bits[6 * i + k];
        h[layout.first + i] = static_cast<uint8_t>(v);
    }
    std::vector<size_t> erasures;
    for (size_t i = 0; i < 63; ++i)
        if (i < layout.first || i >= layout.first + layout.count)
            erasures.push_back(i);
    if (p25::rs63_decode(h, 28, erasures) < 0)
        return false;

    std::vector<uint8_t> data;
    for (size_t i = 0; i < layout.data; ++i)
        for (size_t k = 0; k < 6; ++k)
            data.push_back(static_cast<uint8_t>((h[layout.first + i] >> (5 - k)) & 1u));
    const size_t payload = data.size() - 12;
    uint16_t     crc     = 0;
    for (size_t i = payload; i < data.size(); ++i)
        crc = static_cast<uint16_t>((crc << 1) | data[i]);
    if (crc != crc12(data.data(), payload))
        return false;
    pdu.assign(payload / 8, 0);
    for (size_t i = 0; i < payload; ++i)
        pdu[i / 8] = static_cast<uint8_t>(pdu[i / 8] | (data[i] << (7 - i % 8)));
    return true;
}

void encode_mac(const std::vector<uint8_t>& pdu, bool facch, uint8_t* slot)
{
    const Layout&        layout = facch ? kFacchLayout : kSacchLayout;
    std::vector<uint8_t> data;
    for (size_t i = 0; i < 8 * pdu.size(); ++i)
        data.push_back(static_cast<uint8_t>((pdu[i / 8] >> (7 - i % 8)) & 1u));
    const uint16_t crc = crc12(data.data(), data.size());
    for (int k = 11; k >= 0; --k)
        data.push_back(static_cast<uint8_t>((crc >> k) & 1u));
    uint8_t h[63] = {};
    for (size_t i = 0; i < layout.data; ++i) {
        unsigned v = 0;
        for (size_t k = 0; k < 6; ++k)
            v = (v << 1) | data[6 * i + k];
        h[layout.first + i] = static_cast<uint8_t>(v);
    }
    p25::rs_encode(h, 63, 28);
    std::vector<uint8_t> bits;
    for (size_t i = 0; i < layout.count; ++i)
        for (size_t k = 0; k < 6; ++k)
            bits.push_back(static_cast<uint8_t>((h[layout.first + i] >> (5 - k)) & 1u));
    if (facch)
        scatter(bits, kFacchRuns, slot);
    else
        scatter(bits, kSacchRuns, slot);
}

// ---- Working out the system from a signalling burst ------------------------------------

namespace {

// A vector over GF(2) of up to 320 bits.
using Bits = std::array<uint64_t, 5>;

bool bit_of(const Bits& v, size_t i) { return (v[i / 64] >> (i % 64)) & 1u; }
void flip(Bits& v, size_t i) { v[i / 64] ^= uint64_t{1} << (i % 64); }
bool dot(const Bits& a, const Bits& b)
{
    uint64_t x = 0;
    for (size_t w = 0; w < a.size(); ++w)
        x ^= a[w] & b[w];
    return std::popcount(x) & 1;
}

// The bits of a burst's signalling, in the order they are sent.
Bits burst_bits(const uint8_t* slot, bool facch)
{
    std::vector<uint8_t> bits;
    if (facch)
        gather(slot, kFacchRuns, bits);
    else
        gather(slot, kSacchRuns, bits);
    Bits v{};
    for (size_t i = 0; i < bits.size(); ++i)
        if (bits[i])
            flip(v, i);
    return v;
}

// For one kind of burst: the bits sent for an all-zero PDU, the parity checks
// every valid burst satisfies (relative to that), and per timeslot, for each
// system bit, the bits of the scrambling it flips.
struct Checks {
    size_t                                       length = 0;  // bits sent
    Bits                                         zero{};
    std::vector<Bits>                            parity;
    std::array<std::array<Bits, 44>, kSlots>     scrambled{};
};

Checks make_checks(bool facch)
{
    Checks c;
    const size_t payload = 8 * (facch ? kFacchBytes : kSacchBytes);
    c.length             = 2 * (facch ? 135 : 156);
    std::array<uint8_t, kSlotDibits> slot{};
    encode_mac(std::vector<uint8_t>(payload / 8, 0), facch, slot.data());
    c.zero = burst_bits(slot.data(), facch);

    // The code's generator rows, each PDU bit's effect, brought to reduced
    // row echelon form; the parity checks are its null space.
    std::vector<Bits> rows;
    for (size_t k = 0; k < payload; ++k) {
        std::vector<uint8_t> pdu(payload / 8, 0);
        pdu[k / 8] = static_cast<uint8_t>(0x80u >> (k % 8));
        encode_mac(pdu, facch, slot.data());
        Bits row = burst_bits(slot.data(), facch);
        for (size_t w = 0; w < row.size(); ++w)
            row[w] ^= c.zero[w];
        rows.push_back(row);
    }
    std::vector<size_t> pivots;
    size_t              rank = 0;
    for (size_t col = 0; col < c.length && rank < rows.size(); ++col) {
        size_t r = rank;
        while (r < rows.size() && !bit_of(rows[r], col))
            ++r;
        if (r == rows.size())
            continue;
        std::swap(rows[rank], rows[r]);
        for (size_t i = 0; i < rows.size(); ++i)
            if (i != rank && bit_of(rows[i], col))
                for (size_t w = 0; w < rows[i].size(); ++w)
                    rows[i][w] ^= rows[rank][w];
        pivots.push_back(col);
        ++rank;
    }
    for (size_t col = 0; col < c.length; ++col) {
        if (std::find(pivots.begin(), pivots.end(), col) != pivots.end())
            continue;
        Bits check{};
        flip(check, col);
        for (size_t i = 0; i < rank; ++i)
            if (bit_of(rows[i], col))
                flip(check, pivots[i]);
        c.parity.push_back(check);
    }

    // The scrambling is linear in the system's bits: each bit's own sequence.
    for (size_t j = 0; j < 44; ++j) {
        const uint64_t       seed = uint64_t{1} << j;
        const System         s{static_cast<uint32_t>(seed >> 24), static_cast<uint16_t>((seed >> 12) & 0xFFFu),
                               static_cast<uint16_t>(seed & 0xFFFu)};
        const auto           mask = scrambling(s);
        for (size_t t = 0; t < kSlots; ++t) {
            std::array<uint8_t, kSlotDibits> m{};
            for (size_t d = 10; d < kSlotDibits; ++d)
                m[d] = mask[t * kSlotDibits + d - 10];
            c.scrambled[t][j] = burst_bits(m.data(), facch);
        }
    }
    return c;
}

const Checks& checks(bool facch)
{
    static const Checks facch_checks = make_checks(true);
    static const Checks sacch_checks = make_checks(false);
    return facch ? facch_checks : sacch_checks;
}

}  // namespace

std::optional<System> recover_system(const uint8_t* slot_dibits, int timeslot, bool facch)
{
    if (timeslot < 0 || timeslot >= static_cast<int>(kSlots))
        return std::nullopt;
    const Checks& c        = checks(facch);
    Bits          received = burst_bits(slot_dibits, facch);
    for (size_t w = 0; w < received.size(); ++w)
        received[w] ^= c.zero[w];
    // Each parity check h: h . (received + scrambling(system)) = 0, i.e.
    // sum over the system's bits j of (h . scrambled_j) s_j = h . received.
    // One equation a row: bits 0..43 the coefficients, bit 44 the right side.
    std::vector<uint64_t> eq;
    eq.reserve(c.parity.size());
    for (const Bits& h : c.parity) {
        uint64_t row = 0;
        for (size_t j = 0; j < 44; ++j)
            if (dot(h, c.scrambled[static_cast<size_t>(timeslot)][j]))
                row |= uint64_t{1} << j;
        if (dot(h, received))
            row |= uint64_t{1} << 44;
        eq.push_back(row);
    }
    size_t rank = 0;
    for (size_t j = 0; j < 44; ++j) {
        size_t r = rank;
        while (r < eq.size() && !((eq[r] >> j) & 1u))
            ++r;
        if (r == eq.size())
            return std::nullopt;  // not determined: should not happen
        std::swap(eq[rank], eq[r]);
        for (size_t i = 0; i < eq.size(); ++i)
            if (i != rank && ((eq[i] >> j) & 1u))
                eq[i] ^= eq[rank];
        ++rank;
    }
    // Left over: equations with no unknowns, which an error-free burst satisfies.
    for (size_t i = rank; i < eq.size(); ++i)
        if (eq[i] != 0)
            return std::nullopt;
    uint64_t seed = 0;
    for (size_t j = 0; j < 44; ++j)
        if ((eq[j] >> 44) & 1u)
            seed |= uint64_t{1} << j;
    return System{static_cast<uint32_t>(seed >> 24), static_cast<uint16_t>((seed >> 12) & 0xFFFu),
                  static_cast<uint16_t>(seed & 0xFFFu)};
}

// ---- MAC PDUs ---------------------------------------------------------------------------

uint8_t mac_opcode(const std::vector<uint8_t>& pdu) { return pdu.empty() ? 0 : static_cast<uint8_t>(pdu[0] >> 5); }

std::optional<MacCall> mac_call(const std::vector<uint8_t>& pdu)
{
    if (pdu.size() < kFacchBytes)
        return std::nullopt;
    MacCall    call;
    const auto u24 = [&](size_t i) { return (uint32_t{pdu[i]} << 16) | (uint32_t{pdu[i + 1]} << 8) | pdu[i + 2]; };
    switch (mac_opcode(pdu)) {
    case kMacPtt:
    case kMacEndPtt:
        call.ids_known   = true;
        call.source      = u24(13);
        call.destination = (uint32_t{pdu[16]} << 8) | pdu[17];
        if (mac_opcode(pdu) == kMacPtt) {
            call.encryption_known = true;
            call.encrypted        = pdu[10] != 0x80 && pdu[10] != 0;
        }
        return call;
    case kMacActive:
    case kMacHangtime: {
        // The messages it carries: a voice channel user one names the call.
        size_t at = 1;
        while (at < pdu.size()) {
            const uint8_t op = pdu[at];
            size_t        length = 0;
            if (op == 0x01 && at + 7 <= pdu.size()) {  // group voice channel user (abbreviated)
                call.ids_known   = true;
                call.group       = true;
                call.destination = (uint32_t{pdu[at + 2]} << 8) | pdu[at + 3];
                call.source      = u24(at + 4);
                length           = 7;
            } else if (op == 0x21 && at + 14 <= pdu.size()) {  // ... (extended)
                call.ids_known   = true;
                call.group       = true;
                call.destination = (uint32_t{pdu[at + 2]} << 8) | pdu[at + 3];
                call.source      = u24(at + 4);
                length           = 14;
            } else if (op == 0x02 && at + 8 <= pdu.size()) {  // unit to unit voice channel user
                call.ids_known   = true;
                call.group       = false;
                call.destination = u24(at + 2);
                call.source      = u24(at + 5);
                length           = 8;
            } else {
                break;  // a message of another kind, or nothing: its length is not known here
            }
            call.encryption_known = true;
            call.encrypted        = (pdu[at + 1] & 0x40u) != 0;
            at += length;
            if (call.ids_known)
                break;
        }
        if (!call.ids_known)
            return std::nullopt;
        return call;
    }
    default:
        return std::nullopt;
    }
}

namespace {

std::vector<uint8_t> pdu_for(bool facch, uint8_t opcode)
{
    std::vector<uint8_t> pdu(facch ? kFacchBytes : kSacchBytes, 0);
    pdu[0] = static_cast<uint8_t>(opcode << 5);
    return pdu;
}

}  // namespace

std::vector<uint8_t> mac_ptt(bool facch, uint16_t group, uint32_t source, uint8_t algorithm)
{
    std::vector<uint8_t> pdu = pdu_for(facch, kMacPtt);
    pdu[10]                  = algorithm;
    pdu[13]                  = static_cast<uint8_t>(source >> 16);
    pdu[14]                  = static_cast<uint8_t>(source >> 8);
    pdu[15]                  = static_cast<uint8_t>(source);
    pdu[16]                  = static_cast<uint8_t>(group >> 8);
    pdu[17]                  = static_cast<uint8_t>(group);
    return pdu;
}

std::vector<uint8_t> mac_end_ptt(bool facch, uint16_t nac, uint16_t group, uint32_t source)
{
    std::vector<uint8_t> pdu = mac_ptt(facch, group, source, 0);
    pdu[0]                   = static_cast<uint8_t>(kMacEndPtt << 5);
    pdu[1]                   = static_cast<uint8_t>((nac >> 8) & 0xFu);
    pdu[2]                   = static_cast<uint8_t>(nac);
    pdu[10]                  = 0;
    return pdu;
}

std::vector<uint8_t> mac_active(bool facch, uint16_t group, uint32_t source, uint8_t options)
{
    std::vector<uint8_t> pdu = pdu_for(facch, kMacActive);
    pdu[1]                   = 0x01;
    pdu[2]                   = options;
    pdu[3]                   = static_cast<uint8_t>(group >> 8);
    pdu[4]                   = static_cast<uint8_t>(group);
    pdu[5]                   = static_cast<uint8_t>(source >> 16);
    pdu[6]                   = static_cast<uint8_t>(source >> 8);
    pdu[7]                   = static_cast<uint8_t>(source);
    return pdu;
}

std::vector<uint8_t> mac_idle(bool facch) { return pdu_for(facch, kMacIdle); }

}  // namespace spectro::p25p2
