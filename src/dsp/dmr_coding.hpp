// The coding of DMR (ETSI TS 102 361-1): the error-correcting codes that
// protect each part of a burst, how bits are spread over the burst, and the
// link control that names who is calling whom.
//
// Bits are held one per byte (0 or 1), most significant first, in the order
// in which they are sent.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace spectro::dmr {

// ---- Bursts ---------------------------------------------------------------
//
// Every 30 ms carries 144 symbols (288 bits) at 4800 Bd: a burst of 132
// symbols, preceded on a repeater's downlink by the 12-symbol CACH, which says
// which of the two time slots the burst belongs to.  A burst is
//
//    bits   0..107   payload, first half
//         108..155   a sync pattern, or in a voice burst B-F the EMB and a
//                    fragment of embedded signalling
//         156..263   payload, second half
//
// A data burst gives up 20 of its payload bits next to the middle to its slot
// type: 98..107 and 156..165.

constexpr size_t kFrameSymbols = 144;  // CACH + burst: one time slot
constexpr size_t kCachBits     = 24;
constexpr size_t kBurstBits    = 264;
constexpr size_t kSyncStart    = 66;   // symbol in the frame where the sync pattern starts

// The 48-bit patterns in the middle of a burst.
enum class Sync : uint8_t {
    None,
    BsVoice,   // from a repeater (base station)
    BsData,
    MsVoice,   // from a radio (mobile station) to a repeater
    MsData,
    Ts1Voice,  // radio to radio, without a repeater ("direct mode"), slot 1
    Ts1Data,
    Ts2Voice,  // direct mode, slot 2
    Ts2Data,
};
constexpr size_t kSyncPatterns = 8;  // all but None
uint64_t sync_pattern(Sync sync);     // 48 bits, the first sent in bit 47
bool     is_voice(Sync sync);
bool     from_base_station(Sync sync);

// Symbols as sent: +3, +1, -1, -3 for the dibits 01, 00, 10, 11.
int     symbol_for_dibit(uint8_t dibit);
uint8_t dibit_for_symbol(int symbol);

// Data types, from a data burst's slot type.
enum DataType : uint8_t {
    kPiHeader        = 0,   // the call that follows is encrypted
    kVoiceLcHeader   = 1,   // a voice call starts
    kTerminatorWithLc = 2,  // and ends
    kCsbk            = 3,
    kMbcHeader       = 4,
    kMbcContinuation = 5,
    kDataHeader      = 6,
    kRate12Data      = 7,
    kRate34Data      = 8,
    kIdle            = 9,
    kRate1Data       = 10,
};

// ---- Error-correcting codes -----------------------------------------------
//
// The decoders return the number of bit errors corrected, or -1 if there were
// more than the code can correct.

// Golay (20,8): the slot type, colour code (4 bits) and data type (4 bits).
// The codeword holds the 8 information bits in its top bits.
uint32_t golay2087_encode(uint8_t info);
int      golay2087_decode(uint32_t codeword, uint8_t& info);

// Quadratic residue (16,7): the EMB of a voice burst, colour code (4 bits),
// privacy indicator (1) and link-control fragment type (2).
uint16_t qr1676_encode(uint8_t info);
int      qr1676_decode(uint16_t codeword, uint8_t& info);

// Hamming (7,4): the CACH's TACT - access type, time slot, fragment type (2).
uint8_t hamming743_encode(uint8_t info);
int     hamming743_decode(uint8_t codeword, uint8_t& info);

// Reed-Solomon (12,9) over GF(256): the checksum of a full link control.
void rs129_parity(const uint8_t data[9], uint8_t parity[3]);
bool rs129_check(const uint8_t codeword[12]);

// Block product turbo code BPTC (196,96): the payload of a data burst, 96 bits
// protected by Hamming codes over the rows and columns of a matrix, and
// interleaved.  `in` is the burst's 196 payload bits as sent.
std::array<uint8_t, 196> bptc19696_encode(const std::array<uint8_t, 96>& data);
int                      bptc19696_decode(const std::array<uint8_t, 196>& in, std::array<uint8_t, 96>& data);

// ---- Link control -----------------------------------------------------------

struct LinkControl {
    static constexpr uint8_t kGroupVoice   = 0;  // FLCO: a call to a talkgroup
    static constexpr uint8_t kPrivateVoice = 3;  // to one radio
    static constexpr uint8_t kPrivacy      = 0x40;  // service option: encrypted
    static constexpr uint8_t kEmergency    = 0x80;

    uint8_t  flco        = kGroupVoice;  // what it is about
    uint8_t  feature_set = 0;            // 0: standard
    uint8_t  options     = 0;            // service options
    uint32_t destination = 0;            // talkgroup or radio, 24 bits
    uint32_t source      = 0;            // radio, 24 bits

    bool voice() const { return flco == kGroupVoice || flco == kPrivateVoice; }
    bool group() const { return flco == kGroupVoice; }
    bool encrypted() const { return (options & kPrivacy) != 0; }

    std::array<uint8_t, 9> bytes() const;
    static LinkControl     from_bytes(const std::array<uint8_t, 9>& bytes);
};
bool operator==(const LinkControl& a, const LinkControl& b);

// The full link control of a voice LC header or a terminator: BPTC (196,96)
// over the 9 bytes and their Reed-Solomon checksum, which is masked so that
// the two kinds cannot be mistaken for each other.
constexpr uint32_t kVoiceLcHeaderMask = 0x969696;
constexpr uint32_t kTerminatorMask    = 0x999999;
std::array<uint8_t, 196> encode_full_lc(const LinkControl& lc, uint32_t mask);
bool                     decode_full_lc(const std::array<uint8_t, 196>& in, uint32_t mask, LinkControl& lc);

// The link control embedded in voice bursts B-E, 32 bits in each, for a
// receiver that missed the header: 72 bits, a 5-bit checksum and Hamming
// (16,11) rows with a parity row, 128 bits in all, read out by columns.
std::array<uint8_t, 128> encode_embedded_lc(const LinkControl& lc);
bool                     decode_embedded_lc(const std::array<uint8_t, 128>& in, LinkControl& lc);

// The fragment types of embedded signalling (LCSS), in the EMB.
enum Lcss : uint8_t { kSingle = 0, kFirst = 1, kLast = 2, kContinuation = 3 };

// ---- Voice --------------------------------------------------------------------

// One 20 ms AMBE+2 frame: 72 bits, as four codewords C0 (24 bits), C1 (23),
// C2 (11) and C3 (14) - the layout of mbelib's ambe_fr[4][24], in which
// c[0][23] is the first bit of C0.
struct AmbeFrame {
    std::array<std::array<char, 24>, 4> c{};
};
bool operator==(const AmbeFrame& a, const AmbeFrame& b);

// DMR sends the 72 bits interleaved: the codewords one after another, most
// significant bit first, written into 4 columns of 18 and read out by rows.
void ambe_deinterleave(const uint8_t bits[72], AmbeFrame& frame);
void ambe_interleave(const AmbeFrame& frame, uint8_t bits[72]);

// A frame of silence, as DMR equipment sends it.
AmbeFrame ambe_silence();

}  // namespace spectro::dmr
