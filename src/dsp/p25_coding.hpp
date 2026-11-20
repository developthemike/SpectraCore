// The coding of P25 Phase 1 (TIA-102.BAAA): frames, the codes protecting
// their parts, and the link control that says who is calling whom.
//
// Bits are held one per byte (0 or 1), most significant first, in the order
// in which they are sent.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace spectro::p25 {

// ---- Frames -----------------------------------------------------------------
//
// Every frame starts with the frame sync (24 symbols, all outer) and the
// network ID (NID): the network access code (NAC, 12 bits) and the data unit
// ID (DUID, 4 bits), protected by a BCH (63,16) code, and a parity bit.  One
// status symbol is inserted after every 35 others, counting from the start of
// the frame (dibits 35, 71, ...); the lengths below include them.
//
// A voice call is a header (HDU), then logical link data units LDU1 and LDU2
// in turn, each 180 ms: nine 20 ms IMBE voice frames, and between them the
// link control (LDU1) or the encryption sync (LDU2), and low-speed data.  A
// terminator (TDU, TDULC) ends it.

constexpr uint64_t kFrameSync = 0x5575F5FF77FF;  // 48 bits

enum Duid : uint8_t {
    kHdu   = 0x0,  // header data unit
    kTdu   = 0x3,  // terminator without link control
    kLdu1  = 0x5,
    kTsdu  = 0x7,  // trunking signalling (a control channel)
    kLdu2  = 0xA,
    kPdu   = 0xC,  // packet data
    kTdulc = 0xF,  // terminator with link control
};

// Frame length in dibits, status symbols included; 0 for one of no fixed length.
size_t frame_dibits(uint8_t duid);
constexpr size_t kNidEndDibit = 57;  // the NID is complete with this many dibits

constexpr bool is_status_dibit(size_t index) { return index % 36 == 35; }

// Positions in an LDU's bits once the status symbols are taken out.
constexpr size_t kLduBits        = 1680;
constexpr size_t kVoiceStart[9]  = {112, 256, 440, 624, 808, 992, 1176, 1360, 1536};  // 144 bits each
constexpr size_t kLinkStart[6]   = {400, 584, 768, 952, 1136, 1320};                  // 40 bits each
constexpr size_t kLowSpeedStart  = 1504;                                              // 32 bits

// ---- Codes ----------------------------------------------------------------------

// The NID: NAC and DUID, BCH (63,16) and the parity bit (1 for LDU1 and
// LDU2, 0 otherwise), 64 bits with the first sent in bit 63.
uint64_t nid_encode(uint16_t nac, uint8_t duid);
// Corrects up to 8 errors in the BCH codeword; returns the errors corrected,
// or -1.
int nid_decode(uint64_t nid, uint16_t& nac, uint8_t& duid);

// Hamming (10,6,3): one hexbit (6 bits) of link control or encryption sync.
uint16_t hamming1063_encode(uint8_t hexbit);
int      hamming1063_decode(uint16_t codeword, uint8_t& hexbit);

// Reed-Solomon over GF(64), shortened from RS(63, 63 - parity): `n` hexbits,
// the last `parity` of which are the parity.  Decoding corrects up to parity/2
// errors in place and returns how many, or -1.
void rs_encode(uint8_t* hexbits, size_t n, size_t parity);
int  rs_decode(uint8_t* hexbits, size_t n, size_t parity);
// The full RS(63, 63 - parity) with erasures (symbols known to be missing, as
// a punctured code's): corrects e errors and the erasures while 2e + erasures
// <= parity.  hexbits[0] is the highest power.
int rs63_decode(uint8_t hexbits[63], size_t parity, const std::vector<size_t>& erasures);

// ---- Link control and encryption sync -------------------------------------------

struct LinkControl {
    static constexpr uint8_t kGroupVoice   = 0x00;  // LCF: a call to a talkgroup
    static constexpr uint8_t kPrivateVoice = 0x03;  // to one radio
    static constexpr uint8_t kEncrypted    = 0x40;  // service option ("protected")
    static constexpr uint8_t kEmergency    = 0x80;

    uint8_t  lcf         = kGroupVoice;  // link control format
    uint8_t  mfid        = 0;            // manufacturer: 0 standard
    uint8_t  options     = 0;            // service options
    uint32_t destination = 0;            // talkgroup (16 bits) or radio (24 bits)
    uint32_t source      = 0;            // radio, 24 bits

    bool voice() const { return (lcf & 0x3F) == kGroupVoice || (lcf & 0x3F) == kPrivateVoice; }
    bool group() const { return (lcf & 0x3F) == kGroupVoice; }
    bool encrypted() const { return (options & kEncrypted) != 0; }

    std::array<uint8_t, 9> bytes() const;
    static LinkControl     from_bytes(const std::array<uint8_t, 9>& bytes);
};

struct EncryptionSync {
    static constexpr uint8_t kClear = 0x80;  // algorithm ID: no encryption

    std::array<uint8_t, 9> mi{};  // message indicator
    uint8_t                algorithm = kClear;
    uint16_t               key       = 0;

    bool encrypted() const { return algorithm != kClear && algorithm != 0; }
};

// The 240 bits an LDU carries between its voice frames: 24 Hamming-coded
// hexbits, RS (24,12,13) for link control (LDU1), RS (24,16,9) for the
// encryption sync (LDU2).
std::array<uint8_t, 240> encode_link_control(const LinkControl& lc);
bool                     decode_link_control(const std::array<uint8_t, 240>& bits, LinkControl& lc);
std::array<uint8_t, 240> encode_encryption_sync(const EncryptionSync& es);
bool                     decode_encryption_sync(const std::array<uint8_t, 240>& bits, EncryptionSync& es);

// ---- Trunking signalling blocks ------------------------------------------------
//
// A TSDU carries one to three trunking signalling blocks (TSBK), from bit 112
// (status symbols taken out) 196 bits apart: 12 bytes - last-block flag,
// protected flag, opcode, manufacturer, 8 bytes of arguments and a CRC - as
// 48 dibits through a rate 1/2 trellis code (4 states), interleaved.

using Tsbk = std::array<uint8_t, 12>;

// The 196 bits as sent.  encode_tsbk() fills in the CRC.
std::array<uint8_t, 196> encode_tsbk(Tsbk block);
// Viterbi decoding; returns the bit errors it went past, or -1 if the CRC fails.
int decode_tsbk(const uint8_t bits[196], Tsbk& block);

// ---- Voice ----------------------------------------------------------------------

// One 20 ms IMBE frame: 144 bits as eight codewords, C0-C3 of 23 bits, C4-C6
// of 15 and C7 of 7 - the layout of mbelib's imbe_fr[8][23], in which
// c[0][22] is the first bit of C0.
struct ImbeFrame {
    std::array<std::array<char, 23>, 8> c{};
};
bool operator==(const ImbeFrame& a, const ImbeFrame& b);

// P25 sends the 144 bits interleaved: the codewords one after another, most
// significant bit first, written in pairs of columns of a 12 x 12 matrix,
// (0,7), (1,6), (2,9), (3,8), (4,11), (5,10), each pair row by row, and the
// matrix read out by rows.
void imbe_deinterleave(const uint8_t bits[144], ImbeFrame& frame);
void imbe_interleave(const ImbeFrame& frame, uint8_t bits[144]);

}  // namespace spectro::p25
