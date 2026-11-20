// Produces DMR as a repeater sends it: a continuous stream of 30 ms frames,
// the time slots taking turns, idle bursts where nothing is going on, and voice
// calls - voice LC headers, superframes of six voice bursts with the link
// control embedded, and a terminator.  For the simulator and for tests.
#pragma once

#include <array>
#include <cstdint>
#include <functional>

#include "dsp/dmr_coding.hpp"

namespace spectro {

class DmrTransmitter {
public:
    // The voice, 20 ms at a time.
    using VoiceSource = std::function<dmr::AmbeFrame()>;

    explicit DmrTransmitter(uint8_t color_code = 1);

    // A call on `slot` (0 or 1) for `superframes` x 360 ms of voice.  An
    // encrypted call is flagged as one (its voice is whatever `voice` gives).
    void start_call(int slot, const dmr::LinkControl& lc, int superframes, VoiceSource voice);
    bool in_call(int slot) const { return calls_[static_cast<size_t>(slot)].stage != Stage::Idle; }

    // The next frame - CACH and burst - as symbols (+3, +1, -1, -3), and the
    // slot it belongs to.
    int next_frame(std::array<int8_t, dmr::kFrameSymbols>& symbols);
    // Or one symbol at a time.
    int next_symbol();

private:
    enum class Stage { Idle, Header, Voice, Terminator };
    struct Call {
        Stage            stage = Stage::Idle;
        dmr::LinkControl lc;
        int              remaining = 0;  // bursts left in this stage
        int              burst     = 0;  // voice: 0 = A ... 5 = F
        VoiceSource      voice;
        std::array<uint8_t, 128> embedded{};
    };

    void data_burst(uint8_t* burst, uint8_t type, const std::array<uint8_t, 196>& payload) const;
    void voice_burst(uint8_t* burst, Call& call);

    uint8_t             color_code_;
    std::array<uint8_t, 196> idle_{};  // an idle burst's payload
    int                 slot_ = 1;  // of the last frame
    std::array<Call, 2> calls_;
    std::array<int8_t, dmr::kFrameSymbols> frame_{};
    size_t              frame_pos_ = dmr::kFrameSymbols;
};

}  // namespace spectro
