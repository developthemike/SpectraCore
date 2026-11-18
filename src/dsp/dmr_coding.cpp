#include "dsp/dmr_coding.hpp"

#include <bit>

namespace spectro::dmr {

namespace {

// ---- Small linear codes ------------------------------------------------------
//
// Each is given by the parity bits that each information bit contributes (the
// generator matrix); they are short enough to decode by comparing the word
// received with every codeword, which corrects as many errors as the code can.

// Golay (20,8,8): the parity of information bits 0 (least significant) to 7.
constexpr uint16_t kGolayParity[8] = {0x8EB, 0x93E, 0xA97, 0xDC6, 0x367, 0x6CD, 0xD99, 0x3DA};
// Quadratic residue (16,7,6).
constexpr uint16_t kQrParity[7] = {0x073, 0x0E5, 0x1C9, 0x1E2, 0x1B7, 0x11E, 0x04F};

template <size_t N>
uint32_t parity_of(uint32_t info, const uint16_t (&rows)[N])
{
    uint32_t p = 0;
    for (size_t k = 0; k < N; ++k) {
        if ((info >> k) & 1u)
            p ^= rows[k];
    }
    return p;
}

// The information whose codeword is nearest to `received`, if no more than
// `correctable` bits away.
template <typename Encode>
int nearest(uint32_t received, unsigned info_values, int correctable, Encode encode, uint8_t& info)
{
    int best = 1000;
    for (unsigned v = 0; v < info_values; ++v) {
        const int d = std::popcount(received ^ encode(static_cast<uint8_t>(v)));
        if (d < best) {
            best = d;
            info = static_cast<uint8_t>(v);
        }
    }
    return best <= correctable ? best : -1;
}

// ---- Hamming codes -------------------------------------------------------------

// A Hamming code over bits d[0..k) followed by its parity bits: parity bit p
// is the sum of the data bits in mask p (bit j = d[j]).
struct Hamming {
    int      k;
    int      parities;
    uint16_t masks[5];
};
// (15,11,3) for the rows of BPTC (196,96), (13,9,3) for its columns, (16,11,4)
// for the embedded link control.
constexpr Hamming kHamming15113 = {11, 4, {0x1AF, 0x35E, 0x6BC, 0x4D7, 0}};
constexpr Hamming kHamming1393  = {9, 4, {0x06B, 0x0D7, 0x1AF, 0x135, 0}};
constexpr Hamming kHamming16114 = {11, 5, {0x1AF, 0x35E, 0x6BC, 0x4D7, 0x765}};

uint8_t data_parity(const uint8_t* d, const Hamming& h, int p)
{
    uint8_t v = 0;
    for (int j = 0; j < h.k; ++j) {
        if ((h.masks[p] >> j) & 1u)
            v ^= d[j];
    }
    return v;
}

void hamming_encode(uint8_t* d, const Hamming& h)
{
    for (int p = 0; p < h.parities; ++p)
        d[h.k + p] = data_parity(d, h, p);
}

// Corrects one error.  Returns the errors corrected (0 or 1), or -1.
// d[i * stride] is bit i.
int hamming_correct(uint8_t* d, size_t stride, const Hamming& h)
{
    uint8_t bits[16];
    const int n = h.k + h.parities;
    for (int i = 0; i < n; ++i)
        bits[i] = d[static_cast<size_t>(i) * stride];
    unsigned syndrome = 0;
    for (int p = 0; p < h.parities; ++p) {
        if (data_parity(bits, h, p) != bits[h.k + p])
            syndrome |= 1u << p;
    }
    if (syndrome == 0)
        return 0;
    int wrong = -1;
    for (int p = 0; p < h.parities; ++p) {
        if (syndrome == (1u << p))
            wrong = h.k + p;
    }
    for (int j = 0; j < h.k && wrong < 0; ++j) {
        unsigned column = 0;
        for (int p = 0; p < h.parities; ++p)
            column |= ((h.masks[p] >> j) & 1u) << p;
        if (column == syndrome)
            wrong = j;
    }
    if (wrong < 0)
        return -1;
    d[static_cast<size_t>(wrong) * stride] ^= 1u;
    return 1;
}

bool hamming_valid(const uint8_t* d, size_t stride, const Hamming& h)
{
    uint8_t bits[16];
    for (int i = 0; i < h.k + h.parities; ++i)
        bits[i] = d[static_cast<size_t>(i) * stride];
    for (int p = 0; p < h.parities; ++p) {
        if (data_parity(bits, h, p) != bits[h.k + p])
            return false;
    }
    return true;
}

// ---- GF(256), x^8 + x^4 + x^3 + x^2 + 1 ----------------------------------------

uint8_t gf_multiply(uint8_t a, uint8_t b)
{
    uint8_t product = 0;
    while (b) {
        if (b & 1u)
            product ^= a;
        const bool carry = (a & 0x80u) != 0;
        a                = static_cast<uint8_t>(a << 1);
        if (carry)
            a ^= 0x1Du;
        b >>= 1;
    }
    return product;
}

// ---- Bits -----------------------------------------------------------------------

template <size_t N>
void put_bits(uint8_t* out, uint64_t value)
{
    for (size_t i = 0; i < N; ++i)
        out[i] = static_cast<uint8_t>((value >> (N - 1 - i)) & 1u);
}

uint32_t get_bits(const uint8_t* in, size_t n)
{
    uint32_t v = 0;
    for (size_t i = 0; i < n; ++i)
        v = (v << 1) | in[i];
    return v;
}

}  // namespace

// ---- Bursts -------------------------------------------------------------------

uint64_t sync_pattern(Sync sync)
{
    switch (sync) {
    case Sync::BsVoice: return 0x755FD7DF75F7;
    case Sync::BsData: return 0xDFF57D75DF5D;
    case Sync::MsVoice: return 0x7F7D5DD57DFD;
    case Sync::MsData: return 0xD5D7F77FD757;
    case Sync::Ts1Voice: return 0x5D577F7757FF;
    case Sync::Ts1Data: return 0xF7FDD5DDFD55;
    case Sync::Ts2Voice: return 0x7DFFD5F55D5F;
    case Sync::Ts2Data: return 0xD7557F5FF7F5;
    case Sync::None: break;
    }
    return 0;
}

bool is_voice(Sync sync)
{
    return sync == Sync::BsVoice || sync == Sync::MsVoice || sync == Sync::Ts1Voice || sync == Sync::Ts2Voice;
}

bool from_base_station(Sync sync) { return sync == Sync::BsVoice || sync == Sync::BsData; }

int symbol_for_dibit(uint8_t dibit)
{
    static constexpr int kSymbols[4] = {+1, +3, -1, -3};
    return kSymbols[dibit & 3u];
}

uint8_t dibit_for_symbol(int symbol)
{
    switch (symbol) {
    case 3: return 1;
    case 1: return 0;
    case -1: return 2;
    default: return 3;
    }
}

// ---- Codes ------------------------------------------------------------------------

uint32_t golay2087_encode(uint8_t info) { return (uint32_t{info} << 12) | parity_of(info, kGolayParity); }

int golay2087_decode(uint32_t codeword, uint8_t& info)
{
    return nearest(codeword & 0xFFFFFu, 256, 3, golay2087_encode, info);
}

uint16_t qr1676_encode(uint8_t info)
{
    info &= 0x7Fu;
    return static_cast<uint16_t>((uint32_t{info} << 9) | parity_of(info, kQrParity));
}

int qr1676_decode(uint16_t codeword, uint8_t& info) { return nearest(codeword, 128, 2, qr1676_encode, info); }

// TACT: AT, TC, LCSS (2 bits), then H0 = AT+TC+LS1, H1 = TC+LS1+LS0, H2 = AT+TC+LS0.
uint8_t hamming743_encode(uint8_t info)
{
    const unsigned at = (info >> 3) & 1u, tc = (info >> 2) & 1u, ls1 = (info >> 1) & 1u, ls0 = info & 1u;
    const unsigned h0 = at ^ tc ^ ls1, h1 = tc ^ ls1 ^ ls0, h2 = at ^ tc ^ ls0;
    return static_cast<uint8_t>(((info & 0xFu) << 3) | (h0 << 2) | (h1 << 1) | h2);
}

int hamming743_decode(uint8_t codeword, uint8_t& info) { return nearest(codeword & 0x7Fu, 16, 1, hamming743_encode, info); }

// The generator polynomial is (x - a)(x - a^2)(x - a^3) = x^3 + 14 x^2 + 56 x + 64.
void rs129_parity(const uint8_t data[9], uint8_t parity[3])
{
    uint8_t r2 = 0, r1 = 0, r0 = 0;  // the remainder, r2 the highest power
    for (int i = 0; i < 9; ++i) {
        const uint8_t feedback = data[i] ^ r2;
        r2                     = r1 ^ gf_multiply(14, feedback);
        r1                     = r0 ^ gf_multiply(56, feedback);
        r0                     = gf_multiply(64, feedback);
    }
    parity[0] = r2;
    parity[1] = r1;
    parity[2] = r0;
}

bool rs129_check(const uint8_t codeword[12])
{
    uint8_t parity[3];
    rs129_parity(codeword, parity);
    return parity[0] == codeword[9] && parity[1] == codeword[10] && parity[2] == codeword[11];
}

// BPTC (196,96).  Before interleaving the bits form a matrix of 13 rows of 15
// after one unused bit: m[1 + 15 r + c].  Rows 0-8 carry the data in columns
// 0-10 (row 0 from column 3: three reserved bits come first) and a Hamming
// (15,11) parity in columns 11-14; rows 9-12 are the Hamming (13,9) parity
// of each column - and so, the codes being linear, rows of the row code too,
// which the decoder makes use of.  Bit a of the matrix is sent as bit
// (181 a) mod 196.
namespace {

constexpr size_t bptc_position(size_t a) { return (a * 181) % 196; }

size_t bptc_data_index(size_t i)  // matrix index of data bit i
{
    if (i < 8)
        return 4 + i;
    const size_t row = 1 + (i - 8) / 11, column = (i - 8) % 11;
    return 1 + 15 * row + column;
}

}  // namespace

std::array<uint8_t, 196> bptc19696_encode(const std::array<uint8_t, 96>& data)
{
    std::array<uint8_t, 196> m{};
    for (size_t i = 0; i < 96; ++i)
        m[bptc_data_index(i)] = data[i];
    for (size_t r = 0; r < 9; ++r)
        hamming_encode(&m[1 + 15 * r], kHamming15113);
    for (size_t c = 0; c < 15; ++c) {
        uint8_t column[13];
        for (size_t r = 0; r < 9; ++r)
            column[r] = m[1 + 15 * r + c];
        hamming_encode(column, kHamming1393);
        for (size_t r = 9; r < 13; ++r)
            m[1 + 15 * r + c] = column[r];
    }
    std::array<uint8_t, 196> out{};
    for (size_t a = 0; a < 196; ++a)
        out[bptc_position(a)] = m[a];
    return out;
}

int bptc19696_decode(const std::array<uint8_t, 196>& in, std::array<uint8_t, 96>& data)
{
    std::array<uint8_t, 196> m{};
    for (size_t a = 0; a < 196; ++a)
        m[a] = in[bptc_position(a)];

    // Columns and rows in turn, until nothing more changes.
    int corrected = 0;
    for (int pass = 0; pass < 5; ++pass) {
        int fixed = 0;
        for (size_t c = 0; c < 15; ++c) {
            const int e = hamming_correct(&m[1 + c], 15, kHamming1393);
            fixed += e > 0 ? e : 0;
        }
        for (size_t r = 0; r < 13; ++r) {
            const int e = hamming_correct(&m[1 + 15 * r], 1, kHamming15113);
            fixed += e > 0 ? e : 0;
        }
        corrected += fixed;
        if (fixed == 0)
            break;
    }
    for (size_t c = 0; c < 15; ++c) {
        if (!hamming_valid(&m[1 + c], 15, kHamming1393))
            return -1;
    }
    for (size_t r = 0; r < 13; ++r) {
        if (!hamming_valid(&m[1 + 15 * r], 1, kHamming15113))
            return -1;
    }
    for (size_t i = 0; i < 96; ++i)
        data[i] = m[bptc_data_index(i)];
    return corrected;
}

// ---- Link control ----------------------------------------------------------------

std::array<uint8_t, 9> LinkControl::bytes() const
{
    return {static_cast<uint8_t>(flco & 0x3Fu),
            feature_set,
            options,
            static_cast<uint8_t>(destination >> 16),
            static_cast<uint8_t>(destination >> 8),
            static_cast<uint8_t>(destination),
            static_cast<uint8_t>(source >> 16),
            static_cast<uint8_t>(source >> 8),
            static_cast<uint8_t>(source)};
}

LinkControl LinkControl::from_bytes(const std::array<uint8_t, 9>& b)
{
    LinkControl lc;
    lc.flco        = b[0] & 0x3Fu;
    lc.feature_set = b[1];
    lc.options     = b[2];
    lc.destination = (uint32_t{b[3]} << 16) | (uint32_t{b[4]} << 8) | b[5];
    lc.source      = (uint32_t{b[6]} << 16) | (uint32_t{b[7]} << 8) | b[8];
    return lc;
}

bool operator==(const LinkControl& a, const LinkControl& b) { return a.bytes() == b.bytes(); }

std::array<uint8_t, 196> encode_full_lc(const LinkControl& lc, uint32_t mask)
{
    uint8_t                codeword[12];
    const std::array<uint8_t, 9> b = lc.bytes();
    for (size_t i = 0; i < 9; ++i)
        codeword[i] = b[i];
    rs129_parity(codeword, codeword + 9);
    codeword[9] ^= static_cast<uint8_t>(mask >> 16);
    codeword[10] ^= static_cast<uint8_t>(mask >> 8);
    codeword[11] ^= static_cast<uint8_t>(mask);
    std::array<uint8_t, 96> bits{};
    for (size_t i = 0; i < 12; ++i)
        put_bits<8>(&bits[8 * i], codeword[i]);
    return bptc19696_encode(bits);
}

bool decode_full_lc(const std::array<uint8_t, 196>& in, uint32_t mask, LinkControl& lc)
{
    std::array<uint8_t, 96> bits{};
    if (bptc19696_decode(in, bits) < 0)
        return false;
    uint8_t codeword[12];
    for (size_t i = 0; i < 12; ++i)
        codeword[i] = static_cast<uint8_t>(get_bits(&bits[8 * i], 8));
    codeword[9] ^= static_cast<uint8_t>(mask >> 16);
    codeword[10] ^= static_cast<uint8_t>(mask >> 8);
    codeword[11] ^= static_cast<uint8_t>(mask);
    if (!rs129_check(codeword))
        return false;
    std::array<uint8_t, 9> b{};
    for (size_t i = 0; i < 9; ++i)
        b[i] = codeword[i];
    lc = LinkControl::from_bytes(b);
    return true;
}

// Embedded link control.  A matrix of 8 rows of 16: the 72 bits fill columns
// 0-10 of rows 0 and 1 and columns 0-9 of rows 2-6, whose column 10 holds the
// checksum (the sum of the 9 bytes, modulo 31), most significant bit first;
// columns 11-15 of rows 0-6 are a Hamming (16,11) parity, and row 7 the even
// parity of each column.  Matrix bit (16 a) mod 127 is sent as bit a, and bit
// 127 last.
namespace {

size_t embedded_data_index(size_t i)  // matrix index of link-control bit i
{
    if (i < 22)
        return 16 * (i / 11) + i % 11;
    return 16 * (2 + (i - 22) / 10) + (i - 22) % 10;
}

size_t embedded_position(size_t a) { return a == 127 ? 127 : (16 * a) % 127; }

}  // namespace

std::array<uint8_t, 128> encode_embedded_lc(const LinkControl& lc)
{
    const std::array<uint8_t, 9> b = lc.bytes();
    unsigned                     sum = 0;
    for (uint8_t v : b)
        sum += v;
    const unsigned checksum = sum % 31;

    std::array<uint8_t, 128> m{};
    for (size_t i = 0; i < 72; ++i)
        m[embedded_data_index(i)] = static_cast<uint8_t>((b[i / 8] >> (7 - i % 8)) & 1u);
    for (size_t r = 2; r < 7; ++r)
        m[16 * r + 10] = static_cast<uint8_t>((checksum >> (6 - r)) & 1u);
    for (size_t r = 0; r < 7; ++r)
        hamming_encode(&m[16 * r], kHamming16114);
    for (size_t c = 0; c < 16; ++c) {
        uint8_t parity = 0;
        for (size_t r = 0; r < 7; ++r)
            parity ^= m[16 * r + c];
        m[112 + c] = parity;
    }
    std::array<uint8_t, 128> out{};
    for (size_t a = 0; a < 128; ++a)
        out[a] = m[embedded_position(a)];
    return out;
}

bool decode_embedded_lc(const std::array<uint8_t, 128>& in, LinkControl& lc)
{
    std::array<uint8_t, 128> m{};
    for (size_t a = 0; a < 128; ++a)
        m[embedded_position(a)] = in[a];
    for (size_t r = 0; r < 7; ++r) {
        if (hamming_correct(&m[16 * r], 1, kHamming16114) < 0)
            return false;
    }
    for (size_t c = 0; c < 16; ++c) {
        uint8_t parity = 0;
        for (size_t r = 0; r < 8; ++r)
            parity ^= m[16 * r + c];
        if (parity)
            return false;
    }
    std::array<uint8_t, 9> b{};
    for (size_t i = 0; i < 72; ++i)
        b[i / 8] = static_cast<uint8_t>(b[i / 8] | (m[embedded_data_index(i)] << (7 - i % 8)));
    unsigned checksum = 0, sum = 0;
    for (size_t r = 2; r < 7; ++r)
        checksum = (checksum << 1) | m[16 * r + 10];
    for (uint8_t v : b)
        sum += v;
    if (sum % 31 != checksum)
        return false;
    lc = LinkControl::from_bytes(b);
    return true;
}

// ---- Voice -------------------------------------------------------------------------

namespace {

// Where the j-th of the 72 bits, the codewords one after another, lives in
// the frame: C0 bits 23..0, C1 22..0, C2 10..0, C3 13..0.
void codeword_bit(size_t j, size_t& row, size_t& column)
{
    if (j < 24) {
        row    = 0;
        column = 23 - j;
    } else if (j < 47) {
        row    = 1;
        column = 22 - (j - 24);
    } else if (j < 58) {
        row    = 2;
        column = 10 - (j - 47);
    } else {
        row    = 3;
        column = 13 - (j - 58);
    }
}

// The codeword bit sent as bit k.
size_t interleaved(size_t k) { return (k % 4) * 18 + k / 4; }

}  // namespace

bool operator==(const AmbeFrame& a, const AmbeFrame& b) { return a.c == b.c; }

void ambe_deinterleave(const uint8_t bits[72], AmbeFrame& frame)
{
    frame = {};
    for (size_t k = 0; k < 72; ++k) {
        size_t row = 0, column = 0;
        codeword_bit(interleaved(k), row, column);
        frame.c[row][column] = static_cast<char>(bits[k] & 1u);
    }
}

void ambe_interleave(const AmbeFrame& frame, uint8_t bits[72])
{
    for (size_t k = 0; k < 72; ++k) {
        size_t row = 0, column = 0;
        codeword_bit(interleaved(k), row, column);
        bits[k] = static_cast<uint8_t>(frame.c[row][column] & 1);
    }
}

AmbeFrame ambe_silence()
{
    // As repeaters fill a voice burst that has nothing to say.
    static constexpr uint8_t kSilence[9] = {0xB9, 0xE8, 0x81, 0x52, 0x61, 0x73, 0x00, 0x2A, 0x6B};
    uint8_t                  bits[72];
    for (size_t k = 0; k < 72; ++k)
        bits[k] = static_cast<uint8_t>((kSilence[k / 8] >> (7 - k % 8)) & 1u);
    AmbeFrame frame;
    ambe_deinterleave(bits, frame);
    return frame;
}

}  // namespace spectro::dmr
