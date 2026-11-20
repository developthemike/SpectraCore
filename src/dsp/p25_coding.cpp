#include "dsp/p25_coding.hpp"

#include <algorithm>
#include <bit>
#include <vector>

namespace spectro::p25 {

size_t frame_dibits(uint8_t duid)
{
    switch (duid) {
    case kHdu: return 396;
    case kTdu: return 72;
    case kLdu1:
    case kLdu2: return 864;
    case kTsdu: return 360;  // up to three trunking blocks
    case kTdulc: return 216;
    default: return 0;
    }
}

// ---- NID: BCH (63,16,23) ----------------------------------------------------------

namespace {

constexpr uint64_t kBchGenerator = 06331141367235453;  // degree 47
constexpr int      kMaxNidErrors = 8;                  // of the 11 the code could correct: fewer false NIDs

uint64_t bch_codeword(uint16_t info)
{
    uint64_t r = uint64_t{info} << 47;
    for (int bit = 62; bit >= 47; --bit) {
        if ((r >> bit) & 1u)
            r ^= kBchGenerator << (bit - 47);
    }
    return (uint64_t{info} << 47) | r;
}

bool has_parity_bit(uint8_t duid) { return duid == kLdu1 || duid == kLdu2; }

}  // namespace

uint64_t nid_encode(uint16_t nac, uint8_t duid)
{
    const auto info = static_cast<uint16_t>(((nac & 0xFFFu) << 4) | (duid & 0xFu));
    return (bch_codeword(info) << 1) | (has_parity_bit(duid) ? 1u : 0u);
}

int nid_decode(uint64_t nid, uint16_t& nac, uint8_t& duid)
{
    // The nearest of the 65536 codewords, visited in Gray-code order so that
    // each differs from the last by one generator row.
    static const std::array<uint64_t, 16> rows = [] {
        std::array<uint64_t, 16> r{};
        for (int k = 0; k < 16; ++k)
            r[static_cast<size_t>(k)] = bch_codeword(static_cast<uint16_t>(1u << k));
        return r;
    }();
    const uint64_t received = nid >> 1;
    uint64_t       codeword = 0;
    int            best     = 64;
    uint16_t       best_info = 0;
    for (uint32_t i = 0; i < 65536; ++i) {
        if (i > 0)
            codeword ^= rows[static_cast<size_t>(std::countr_zero(i))];
        const int d = std::popcount(codeword ^ received);
        if (d < best) {
            best      = d;
            best_info = static_cast<uint16_t>(i ^ (i >> 1));
        }
    }
    if (best > kMaxNidErrors)
        return -1;
    nac  = static_cast<uint16_t>(best_info >> 4);
    duid = static_cast<uint8_t>(best_info & 0xFu);
    return best;
}

// ---- Hamming (10,6,3) -------------------------------------------------------------

namespace {

// The parity each data bit contributes, from the most significant down.
constexpr uint8_t kHammingParity[6] = {14, 13, 11, 7, 3, 12};

}  // namespace

uint16_t hamming1063_encode(uint8_t hexbit)
{
    uint8_t parity = 0;
    for (int k = 0; k < 6; ++k) {
        if ((hexbit >> (5 - k)) & 1u)
            parity ^= kHammingParity[k];
    }
    return static_cast<uint16_t>(((hexbit & 0x3Fu) << 4) | parity);
}

int hamming1063_decode(uint16_t codeword, uint8_t& hexbit)
{
    int best = 11;
    for (unsigned v = 0; v < 64; ++v) {
        const int d = std::popcount(static_cast<unsigned>(codeword ^ hamming1063_encode(static_cast<uint8_t>(v))) & 0x3FFu);
        if (d < best) {
            best   = d;
            hexbit = static_cast<uint8_t>(v);
        }
    }
    return best <= 1 ? best : -1;
}

// ---- Reed-Solomon over GF(64) --------------------------------------------------------

namespace {

// GF(64) with x^6 + x + 1; the generator's roots are a^1 ... a^parity.
struct Gf64 {
    uint8_t exp[126];
    int     log[64];
    Gf64()
    {
        unsigned x = 1;
        for (int i = 0; i < 63; ++i) {
            exp[i] = exp[i + 63] = static_cast<uint8_t>(x);
            log[x]               = i;
            x <<= 1;
            if (x & 0x40u)
                x ^= 0x43u;
        }
        log[0] = -1;
    }
    uint8_t mul(uint8_t a, uint8_t b) const { return a && b ? exp[log[a] + log[b]] : 0; }
    uint8_t div(uint8_t a, uint8_t b) const { return a ? exp[(log[a] - log[b] + 63) % 63] : 0; }
    uint8_t pow(int e) const { return exp[((e % 63) + 63) % 63]; }
};

const Gf64& gf()
{
    static const Gf64 field;
    return field;
}

// Generator polynomial, highest power first (g[0] = 1).
std::vector<uint8_t> generator(size_t parity)
{
    const Gf64&          f = gf();
    std::vector<uint8_t> g{1};
    for (size_t i = 1; i <= parity; ++i) {
        std::vector<uint8_t> next(g.size() + 1, 0);
        const uint8_t        root = f.pow(static_cast<int>(i));
        for (size_t j = 0; j < g.size(); ++j) {
            next[j] ^= g[j];
            next[j + 1] ^= f.mul(g[j], root);
        }
        g = next;
    }
    return g;
}

// p(x) at x, coefficients lowest power first.
uint8_t evaluate(const std::vector<uint8_t>& p, uint8_t x)
{
    const Gf64& f = gf();
    uint8_t     v = 0;
    for (size_t i = p.size(); i-- > 0;)
        v = static_cast<uint8_t>(f.mul(v, x) ^ p[i]);
    return v;
}

}  // namespace

void rs_encode(uint8_t* h, size_t n, size_t parity)
{
    const Gf64&                f = gf();
    const std::vector<uint8_t> g = generator(parity);
    std::vector<uint8_t>       r(parity, 0);
    for (size_t i = 0; i < n - parity; ++i) {
        const uint8_t feedback = static_cast<uint8_t>(h[i] ^ r[0]);
        for (size_t j = 0; j + 1 < parity; ++j)
            r[j] = static_cast<uint8_t>(r[j + 1] ^ f.mul(feedback, g[j + 1]));
        r[parity - 1] = f.mul(feedback, g[parity]);
    }
    std::copy(r.begin(), r.end(), h + (n - parity));
}

int rs_decode(uint8_t* h, size_t n, size_t parity)
{
    const Gf64& f = gf();
    // Syndromes S_j = c(a^j), j = 1 .. parity; h[0] is the highest power.
    std::vector<uint8_t> s(parity, 0);
    bool                 errors = false;
    for (size_t j = 0; j < parity; ++j) {
        const uint8_t x = f.pow(static_cast<int>(j + 1));
        uint8_t       v = 0;
        for (size_t i = 0; i < n; ++i)
            v = static_cast<uint8_t>(f.mul(v, x) ^ (h[i] & 0x3Fu));
        s[j] = v;
        errors |= v != 0;
    }
    if (!errors)
        return 0;

    // Berlekamp-Massey: the error locator, lowest power first.
    std::vector<uint8_t> lambda{1}, b{1};
    size_t               l = 0, m = 1;
    uint8_t              bd = 1;
    for (size_t r = 0; r < parity; ++r) {
        uint8_t d = s[r];
        for (size_t i = 1; i <= l && i < lambda.size(); ++i)
            d ^= f.mul(lambda[i], s[r - i]);
        if (d == 0) {
            ++m;
            continue;
        }
        std::vector<uint8_t> t = lambda;
        const uint8_t        c = f.div(d, bd);
        if (lambda.size() < b.size() + m)
            lambda.resize(b.size() + m, 0);
        for (size_t i = 0; i < b.size(); ++i)
            lambda[i + m] ^= f.mul(c, b[i]);
        if (2 * l <= r) {
            l  = r + 1 - l;
            b  = t;
            bd = d;
            m  = 1;
        } else {
            ++m;
        }
    }
    while (lambda.size() > 1 && lambda.back() == 0)
        lambda.pop_back();
    if (l == 0 || 2 * l > parity || lambda.size() - 1 != l)
        return -1;

    // The error evaluator, and the locator's derivative.
    std::vector<uint8_t> omega(parity, 0);
    for (size_t i = 0; i < parity; ++i)
        for (size_t j = 0; j < lambda.size() && i + j < parity; ++j)
            omega[i + j] ^= f.mul(s[i], lambda[j]);
    std::vector<uint8_t> derivative(lambda.size() > 1 ? lambda.size() - 1 : 1, 0);
    for (size_t i = 1; i < lambda.size(); i += 2)
        derivative[i - 1] = lambda[i];

    // Chien search over the positions there are, then Forney's formula.
    size_t found = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t x_inverse = f.pow(-static_cast<int>(n - 1 - i));
        if (evaluate(lambda, x_inverse) != 0)
            continue;
        const uint8_t denominator = evaluate(derivative, x_inverse);
        if (denominator == 0)
            return -1;
        h[i] = static_cast<uint8_t>((h[i] ^ f.div(evaluate(omega, x_inverse), denominator)) & 0x3Fu);
        ++found;
    }
    if (found != l)
        return -1;
    // Make sure.
    for (size_t j = 0; j < parity; ++j) {
        const uint8_t x = f.pow(static_cast<int>(j + 1));
        uint8_t       v = 0;
        for (size_t i = 0; i < n; ++i)
            v = static_cast<uint8_t>(f.mul(v, x) ^ h[i]);
        if (v != 0)
            return -1;
    }
    return static_cast<int>(found);
}

int rs63_decode(uint8_t* h, size_t parity, const std::vector<size_t>& erasures)
{
    const Gf64&  f = gf();
    const size_t n = 63;
    for (size_t e : erasures)
        if (e < n)
            h[e] = 0;
    // Syndromes S_1 ... S_parity.
    std::vector<uint8_t> s(parity, 0);
    bool                 errors = false;
    for (size_t j = 0; j < parity; ++j) {
        const uint8_t x = f.pow(static_cast<int>(j + 1));
        uint8_t       v = 0;
        for (size_t i = 0; i < n; ++i)
            v = static_cast<uint8_t>(f.mul(v, x) ^ (h[i] & 0x3Fu));
        s[j] = v;
        errors |= v != 0;
    }
    if (!errors)
        return 0;
    if (erasures.size() > parity)
        return -1;

    // The erasure locator, Gamma(x) = product of (1 + X_e x); it seeds
    // Berlekamp-Massey, which then finds the errors as well.
    std::vector<uint8_t> lambda{1};
    for (size_t e : erasures) {
        const uint8_t        x = f.pow(static_cast<int>(n - 1 - e));
        std::vector<uint8_t> next(lambda.size() + 1, 0);
        for (size_t i = 0; i < lambda.size(); ++i) {
            next[i] ^= lambda[i];
            next[i + 1] ^= f.mul(lambda[i], x);
        }
        lambda = next;
    }
    std::vector<uint8_t> b = lambda;
    const size_t         v = erasures.size();
    size_t               l = v;
    for (size_t k = v + 1; k <= parity; ++k) {
        uint8_t d = 0;
        for (size_t i = 0; i < lambda.size() && i < k; ++i)
            d ^= f.mul(lambda[i], s[k - 1 - i]);
        std::vector<uint8_t> xb(b.size() + 1, 0);
        std::copy(b.begin(), b.end(), xb.begin() + 1);
        if (d == 0) {
            b = xb;
            continue;
        }
        std::vector<uint8_t> t = lambda;
        if (t.size() < xb.size())
            t.resize(xb.size(), 0);
        for (size_t i = 0; i < xb.size(); ++i)
            t[i] ^= f.mul(d, xb[i]);
        if (2 * l <= k - 1 + v) {
            b.assign(lambda.size(), 0);
            for (size_t i = 0; i < lambda.size(); ++i)
                b[i] = f.div(lambda[i], d);
            l = k + v - l;
        } else {
            b = xb;
        }
        lambda = t;
    }
    while (lambda.size() > 1 && lambda.back() == 0)
        lambda.pop_back();
    if (lambda.size() - 1 != l || 2 * (l - v) + v > parity)
        return -1;

    std::vector<uint8_t> omega(parity, 0);
    for (size_t i = 0; i < parity; ++i)
        for (size_t j = 0; j < lambda.size() && i + j < parity; ++j)
            omega[i + j] ^= f.mul(s[i], lambda[j]);
    std::vector<uint8_t> derivative(lambda.size() > 1 ? lambda.size() - 1 : 1, 0);
    for (size_t i = 1; i < lambda.size(); i += 2)
        derivative[i - 1] = lambda[i];

    size_t found = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t x_inverse = f.pow(-static_cast<int>(n - 1 - i));
        if (evaluate(lambda, x_inverse) != 0)
            continue;
        const uint8_t denominator = evaluate(derivative, x_inverse);
        if (denominator == 0)
            return -1;
        h[i] = static_cast<uint8_t>((h[i] ^ f.div(evaluate(omega, x_inverse), denominator)) & 0x3Fu);
        ++found;
    }
    if (found != l)
        return -1;
    for (size_t j = 0; j < parity; ++j) {
        const uint8_t x = f.pow(static_cast<int>(j + 1));
        uint8_t       w = 0;
        for (size_t i = 0; i < n; ++i)
            w = static_cast<uint8_t>(f.mul(w, x) ^ h[i]);
        if (w != 0)
            return -1;
    }
    return static_cast<int>(found);
}

// ---- Link control and encryption sync -------------------------------------------

std::array<uint8_t, 9> LinkControl::bytes() const
{
    std::array<uint8_t, 9> b{};
    b[0] = lcf;
    b[1] = mfid;
    b[2] = options;
    if ((lcf & 0x3F) == kGroupVoice) {
        b[3] = 0;
        b[4] = static_cast<uint8_t>(destination >> 8);
        b[5] = static_cast<uint8_t>(destination);
    } else {
        b[3] = static_cast<uint8_t>(destination >> 16);
        b[4] = static_cast<uint8_t>(destination >> 8);
        b[5] = static_cast<uint8_t>(destination);
    }
    b[6] = static_cast<uint8_t>(source >> 16);
    b[7] = static_cast<uint8_t>(source >> 8);
    b[8] = static_cast<uint8_t>(source);
    return b;
}

LinkControl LinkControl::from_bytes(const std::array<uint8_t, 9>& b)
{
    LinkControl lc;
    lc.lcf     = b[0];
    lc.mfid    = b[1];
    lc.options = b[2];
    if ((lc.lcf & 0x3F) == kGroupVoice)
        lc.destination = (uint32_t{b[4]} << 8) | b[5];
    else
        lc.destination = (uint32_t{b[3]} << 16) | (uint32_t{b[4]} << 8) | b[5];
    lc.source = (uint32_t{b[6]} << 16) | (uint32_t{b[7]} << 8) | b[8];
    return lc;
}

namespace {

// Bytes to hexbits and back, most significant first.
template <size_t Bytes>
std::array<uint8_t, Bytes * 4 / 3> to_hexbits(const std::array<uint8_t, Bytes>& bytes)
{
    std::array<uint8_t, Bytes * 4 / 3> h{};
    for (size_t i = 0; i < h.size(); ++i) {
        unsigned v = 0;
        for (size_t k = 0; k < 6; ++k) {
            const size_t bit = 6 * i + k;
            v = (v << 1) | ((bytes[bit / 8] >> (7 - bit % 8)) & 1u);
        }
        h[i] = static_cast<uint8_t>(v);
    }
    return h;
}

template <size_t Bytes>
std::array<uint8_t, Bytes> from_hexbits(const uint8_t* h)
{
    std::array<uint8_t, Bytes> bytes{};
    for (size_t bit = 0; bit < 8 * Bytes; ++bit) {
        const unsigned v = (h[bit / 6] >> (5 - bit % 6)) & 1u;
        bytes[bit / 8]   = static_cast<uint8_t>(bytes[bit / 8] | (v << (7 - bit % 8)));
    }
    return bytes;
}

std::array<uint8_t, 240> hexwords_to_bits(const std::array<uint8_t, 24>& h)
{
    std::array<uint8_t, 240> bits{};
    for (size_t i = 0; i < 24; ++i) {
        const uint16_t cw = hamming1063_encode(h[i]);
        for (size_t k = 0; k < 10; ++k)
            bits[10 * i + k] = static_cast<uint8_t>((cw >> (9 - k)) & 1u);
    }
    return bits;
}

std::array<uint8_t, 24> bits_to_hexwords(const std::array<uint8_t, 240>& bits)
{
    std::array<uint8_t, 24> h{};
    for (size_t i = 0; i < 24; ++i) {
        uint16_t cw = 0;
        for (size_t k = 0; k < 10; ++k)
            cw = static_cast<uint16_t>((cw << 1) | bits[10 * i + k]);
        if (hamming1063_decode(cw, h[i]) < 0)
            h[i] = static_cast<uint8_t>(cw >> 4);  // as received: the Reed-Solomon code may mend it
    }
    return h;
}

}  // namespace

std::array<uint8_t, 240> encode_link_control(const LinkControl& lc)
{
    const auto              data = to_hexbits(lc.bytes());
    std::array<uint8_t, 24> h{};
    std::copy(data.begin(), data.end(), h.begin());
    rs_encode(h.data(), 24, 12);
    return hexwords_to_bits(h);
}

bool decode_link_control(const std::array<uint8_t, 240>& bits, LinkControl& lc)
{
    std::array<uint8_t, 24> h = bits_to_hexwords(bits);
    if (rs_decode(h.data(), 24, 12) < 0)
        return false;
    lc = LinkControl::from_bytes(from_hexbits<9>(h.data()));
    return true;
}

std::array<uint8_t, 240> encode_encryption_sync(const EncryptionSync& es)
{
    std::array<uint8_t, 12> bytes{};
    std::copy(es.mi.begin(), es.mi.end(), bytes.begin());
    bytes[9]                = es.algorithm;
    bytes[10]               = static_cast<uint8_t>(es.key >> 8);
    bytes[11]               = static_cast<uint8_t>(es.key);
    const auto              data = to_hexbits(bytes);
    std::array<uint8_t, 24> h{};
    std::copy(data.begin(), data.end(), h.begin());
    rs_encode(h.data(), 24, 8);
    return hexwords_to_bits(h);
}

bool decode_encryption_sync(const std::array<uint8_t, 240>& bits, EncryptionSync& es)
{
    std::array<uint8_t, 24> h = bits_to_hexwords(bits);
    if (rs_decode(h.data(), 24, 8) < 0)
        return false;
    const std::array<uint8_t, 12> bytes = from_hexbits<12>(h.data());
    std::copy(bytes.begin(), bytes.begin() + 9, es.mi.begin());
    es.algorithm = bytes[9];
    es.key       = static_cast<uint16_t>((bytes[10] << 8) | bytes[11]);
    return true;
}

// ---- Trunking signalling blocks ------------------------------------------------

namespace {

// The rate 1/2 trellis: the 4-bit codeword sent for each input dibit, by
// state (the previous input dibit).
constexpr uint8_t kTrellis[4][4] = {{0x2, 0xC, 0x1, 0xF}, {0xE, 0x0, 0xD, 0x3}, {0x9, 0x7, 0xA, 0x4}, {0x5, 0xB, 0x6, 0x8}};

// Codeword k (of 49) is sent from bit tsbk_position(k) on: the 196 bits are
// four columns (13, 12, 12 and 12 codewords) read one after another.
size_t tsbk_position(size_t k)
{
    static constexpr size_t kColumn[4] = {0, 52, 100, 148};
    return 4 * (k / 4) + kColumn[k % 4];
}

uint16_t crc_ccitt(const uint8_t* bytes, size_t count)
{
    uint16_t crc = 0;
    for (size_t i = 0; i < count; ++i) {
        crc ^= static_cast<uint16_t>(bytes[i] << 8);
        for (int b = 0; b < 8; ++b)
            crc = static_cast<uint16_t>(crc & 0x8000u ? (crc << 1) ^ 0x1021u : crc << 1);
    }
    return static_cast<uint16_t>(~crc);
}

}  // namespace

std::array<uint8_t, 196> encode_tsbk(Tsbk block)
{
    const uint16_t crc = crc_ccitt(block.data(), 10);
    block[10]          = static_cast<uint8_t>(crc >> 8);
    block[11]          = static_cast<uint8_t>(crc);
    std::array<uint8_t, 196> bits{};
    unsigned                 state = 0;
    for (size_t k = 0; k < 49; ++k) {
        const unsigned input    = k < 48 ? (block[k / 4] >> (6 - 2 * (k % 4))) & 3u : 0u;  // then a flushing 0
        const uint8_t  codeword = kTrellis[state][input];
        state                   = input;
        for (size_t b = 0; b < 4; ++b)
            bits[tsbk_position(k) + b] = static_cast<uint8_t>((codeword >> (3 - b)) & 1u);
    }
    return bits;
}

int decode_tsbk(const uint8_t bits[196], Tsbk& block)
{
    // Viterbi over 49 codewords; the path starts and ends in state 0.
    constexpr int kInfinity = 1 << 20;
    int           metric[4] = {0, kInfinity, kInfinity, kInfinity};
    uint8_t       from[49][4];
    for (size_t k = 0; k < 49; ++k) {
        unsigned received = 0;
        for (size_t b = 0; b < 4; ++b)
            received = (received << 1) | (bits[tsbk_position(k) + b] & 1u);
        int next[4] = {kInfinity, kInfinity, kInfinity, kInfinity};
        for (unsigned s = 0; s < 4; ++s) {
            if (metric[s] >= kInfinity)
                continue;
            for (unsigned input = 0; input < 4; ++input) {
                const int m = metric[s] + std::popcount(received ^ kTrellis[s][input]);
                if (m < next[input]) {
                    next[input]    = m;
                    from[k][input] = static_cast<uint8_t>(s);
                }
            }
        }
        std::copy(next, next + 4, metric);
    }
    if (metric[0] >= kInfinity)
        return -1;
    block.fill(0);
    unsigned state = 0;
    for (size_t k = 49; k-- > 0;) {
        if (k < 48)
            block[k / 4] = static_cast<uint8_t>(block[k / 4] | (state << (6 - 2 * (k % 4))));
        state = from[k][state];
    }
    const uint16_t crc = crc_ccitt(block.data(), 10);
    if (block[10] != static_cast<uint8_t>(crc >> 8) || block[11] != static_cast<uint8_t>(crc))
        return -1;
    return metric[0];
}

// ---- Voice ----------------------------------------------------------------------

namespace {

constexpr size_t kCodewordBits[8] = {23, 23, 23, 23, 15, 15, 15, 7};

// Where the j-th of the 144 bits, the codewords one after another, is sent.
size_t sent_position(size_t j)
{
    const size_t pair = j / 24, within = j % 24;
    const size_t column = within % 2 == 0 ? pair : pair + (pair % 2 == 0 ? 7 : 5);
    return 12 * (within / 2) + column;
}

template <typename Visit>
void each_bit(Visit visit)
{
    size_t j = 0;
    for (size_t c = 0; c < 8; ++c)
        for (size_t b = kCodewordBits[c]; b-- > 0;)
            visit(sent_position(j++), c, b);
}

}  // namespace

bool operator==(const ImbeFrame& a, const ImbeFrame& b) { return a.c == b.c; }

void imbe_deinterleave(const uint8_t bits[144], ImbeFrame& frame)
{
    frame = {};
    each_bit([&](size_t k, size_t c, size_t b) { frame.c[c][b] = static_cast<char>(bits[k] & 1u); });
}

void imbe_interleave(const ImbeFrame& frame, uint8_t bits[144])
{
    each_bit([&](size_t k, size_t c, size_t b) { bits[k] = static_cast<uint8_t>(frame.c[c][b] & 1); });
}

}  // namespace spectro::p25
