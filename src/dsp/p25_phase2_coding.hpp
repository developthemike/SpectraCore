// The coding of P25 Phase 2 (TIA-102.BBAC), the TDMA voice channel as a base
// station sends it: 6000 symbols a second of differential QPSK, two logical
// channels ("slots") in turn.
//
//   superframe, 360 ms = 12 timeslots of 30 ms; a timeslot = 180 dibits:
//     dibits   0..19   ISCH: the sync word (timeslots 2, 3, 6, 7, 10, 11) or
//                      where in the superframe it is (the others)
//     dibits  20..179  a burst: four voice frames (4V), two (2V) and
//                      encryption sync, or signalling (FACCH, SACCH); its
//                      type (DUID) is spread over dibits 20, 57, 142 and 179
//   timeslots 0-9 alternate between the logical channels; 10 is channel 1's
//   SACCH, 11 channel 0's.
//
// Bursts are scrambled (all but the DUID) with a sequence that depends on the
// system: its WACN, system ID and NAC.  The voice frames are AMBE+2, 72 bits,
// interleaved as DMR's are.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace spectro::p25p2 {

constexpr double   kSymbolRate = 6000.0;
constexpr uint64_t kSync       = 0x575D57F7FF;  // the S-ISCH, 40 bits
constexpr size_t   kSlotDibits = 180;
constexpr size_t   kSlots      = 12;           // timeslots in a superframe

// The logical channel of each timeslot.
constexpr int kChannelOf[kSlots] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0};

enum Burst : uint8_t {
    k4V          = 0,
    kSacch       = 3,   // scrambled
    kLcch        = 4,
    k2V          = 6,
    kFacch       = 9,   // scrambled
    kSacchClear  = 12,
    kLcchClear   = 13,
    kFacchClear  = 15,
};
bool is_scrambled(uint8_t burst);

constexpr size_t kDuidAt[4]   = {20, 57, 142, 179};
constexpr size_t kVoiceAt[4]  = {21, 58, 106, 143};  // 36 dibits each; 2V has the first two
constexpr size_t kEssAt       = 94;                  // 4V: 12 dibits of encryption sync

uint8_t duid_codeword(uint8_t burst);  // (8,4,4) code
int     duid_decode(uint8_t codeword); // the burst type, or -1

// The I-ISCH carries 7 bits: the logical channel, where in the superframe,
// and counters; (40,9,16) code.
uint64_t isch_encode(unsigned value);
// The value, -2 for the S-ISCH (sync), -1 if neither is near enough.
int isch_decode(uint64_t word);
// Timeslot for an I-ISCH value, and the value to send in a timeslot (0, 1, 4,
// 5, 8 or 9).
int      isch_slot(unsigned value);
unsigned isch_value(int slot);

// A system's scrambling: 4320 bits a superframe, as dibits.  Timeslot t's
// dibit d (from 10 on) is scrambled with scrambling[t * 180 + d - 10].
struct System {
    uint32_t wacn   = 0;
    uint16_t system = 0;
    uint16_t nac    = 0;
    bool operator==(const System&) const = default;
};
std::vector<uint8_t> scrambling(const System& system);

uint16_t crc12(const uint8_t* bits, size_t count);

// Signalling: a MAC PDU of 18 bytes (FACCH) or 21 (SACCH), CRC-12 and the
// Reed-Solomon (63,35) code, shortened and punctured, in a timeslot's dibits
// (descrambled).
constexpr size_t kFacchBytes = 18, kSacchBytes = 21;
bool decode_mac(const uint8_t* slot_dibits, bool facch, std::vector<uint8_t>& pdu);
void encode_mac(const std::vector<uint8_t>& pdu, bool facch, uint8_t* slot_dibits);

// The system whose scrambling makes this scrambled FACCH or SACCH (on
// timeslot `timeslot` of the superframe) a valid one, worked out from the
// burst alone - if it came through without errors.
//
// The scrambling is a linear function of the system's 44 bits (over GF(2)),
// and so is the code a MAC PDU is sent in, give or take a constant: a valid
// burst satisfies its parity checks - 126 of them for a FACCH, 144 for a
// SACCH - and each of those is a linear equation in the 44 unknown bits of
// the system.  Solved, they give the system; an error in the burst almost
// always makes them contradict each other instead.  (Check the answer on
// another burst all the same.)
std::optional<System> recover_system(const uint8_t* slot_dibits, int timeslot, bool facch);

// MAC PDU opcodes (the top three bits of its first byte).
enum MacOpcode : uint8_t { kMacSignal = 0, kMacPtt = 1, kMacEndPtt = 2, kMacIdle = 3, kMacActive = 4, kMacHangtime = 6 };

// What a MAC PDU says about the call on its channel.
struct MacCall {
    bool     ids_known   = false;
    bool     group       = true;
    uint32_t destination = 0;
    uint32_t source      = 0;
    bool     encrypted   = false;
    bool     encryption_known = false;
};
uint8_t                mac_opcode(const std::vector<uint8_t>& pdu);
std::optional<MacCall> mac_call(const std::vector<uint8_t>& pdu);

// MAC PDUs as a base station sends them (for the simulator and tests).
std::vector<uint8_t> mac_ptt(bool facch, uint16_t group, uint32_t source, uint8_t algorithm);
std::vector<uint8_t> mac_end_ptt(bool facch, uint16_t nac, uint16_t group, uint32_t source);
std::vector<uint8_t> mac_active(bool facch, uint16_t group, uint32_t source, uint8_t options);
std::vector<uint8_t> mac_idle(bool facch);

}  // namespace spectro::p25p2
