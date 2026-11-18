// Produces P25 Phase 1 as a conventional radio sends it: on the air only for
// a call - a header, LDU1 and LDU2 in turn with the link control and
// encryption sync, then terminators.  Or as a trunked system's control
// channel: on the air all the time, trunking signalling blocks three to a
// TSDU.  For the simulator and for tests.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "dsp/p25_coding.hpp"

namespace spectro {

class P25Transmitter {
public:
    // The voice, 20 ms at a time.
    using VoiceSource = std::function<p25::ImbeFrame()>;
    // A control channel's next trunking block.
    using BlockSource = std::function<p25::Tsbk()>;

    explicit P25Transmitter(uint16_t nac = 0x293);

    // A call of `ldus` x 180 ms of voice.  An encrypted call (the link
    // control's service options say so) carries an algorithm in its
    // encryption sync too.
    void start_call(const p25::LinkControl& lc, int ldus, VoiceSource voice);
    // A control channel, with `blocks_per_tsdu` (1-3) blocks to a TSDU.
    void control(BlockSource blocks, int blocks_per_tsdu = 3);

    bool on_air() const { return pos_ < frame_.size() || header_ || ldus_left_ > 0 || terminators_ > 0 || blocks_; }

    // The next symbol (+3, +1, -1, -3); false when off the air.
    bool next_symbol(int& symbol);

private:
    void build_next_frame();
    void frame(uint8_t duid, const std::vector<uint8_t>& payload, size_t data_bits = 0);

    uint16_t             nac_;
    p25::LinkControl     lc_;
    VoiceSource          voice_;
    bool                 header_      = false;
    int                  ldus_left_   = 0;
    int                  terminators_ = 0;
    bool                 next_is_ldu1_ = true;
    BlockSource          blocks_;
    int                  blocks_per_tsdu_ = 3;
    std::vector<int8_t>  frame_;  // symbols of the frame being sent
    size_t               pos_ = 0;
};

}  // namespace spectro
