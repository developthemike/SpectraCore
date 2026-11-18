// Produces a P25 Phase 2 voice channel as a base station sends it: on the air
// while either logical channel has a call - a MAC PTT, superframes of four 4V
// bursts and a 2V on its timeslots with MAC ACTIVE on its SACCH, a MAC END
// PTT - scrambled with the system's sequence.  Symbols (+3, +1, -1, -3) stand
// for phase steps of +-135 and +-45 degrees.  For the simulator and tests.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#include "dsp/dmr_coding.hpp"
#include "dsp/p25_phase2_coding.hpp"

namespace spectro {

class P25Phase2Transmitter {
public:
    using VoiceSource = std::function<dmr::AmbeFrame()>;

    explicit P25Phase2Transmitter(p25p2::System system);

    // A call on `slot` (0 or 1), `superframes` x 360 ms of voice.
    void start_call(int slot, uint16_t talkgroup, uint32_t source, int superframes, VoiceSource voice,
                    bool encrypted = false);
    bool on_air() const { return pos_ < symbols_.size() || calls_[0].active || calls_[1].active; }

    // The next symbol; false when off the air.
    bool next_symbol(int& symbol);

private:
    struct Call {
        bool        active     = false;
        bool        started    = false;  // its PTT has been sent
        int         left       = 0;      // superframes of voice to go; then its end
        uint16_t    talkgroup  = 0;
        uint32_t    source     = 0;
        bool        encrypted  = false;
        VoiceSource voice;
    };

    void build_superframe();

    p25p2::System        system_;
    std::vector<uint8_t> mask_;
    std::array<Call, 2>  calls_;
    std::vector<int8_t>  symbols_;
    size_t               pos_ = 0;
};

}  // namespace spectro
