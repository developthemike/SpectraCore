#include "dsp/dmr_decoder.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace spectro {

namespace {

constexpr int      kMaxSyncBitErrors = 5;     // of 48, in a burst the clock has placed
constexpr int      kMaxLateSyncErrors = 12;   // where a voice sync is expected
constexpr uint64_t kLostAfterFrames  = 34;    // about a second without a burst that decodes
constexpr uint64_t kCallTimeoutFrames = 12;   // 360 ms without anything from a call
constexpr double   kTimingGain       = 0.5;
constexpr double   kLevelGain        = 0.3;

constexpr dmr::Sync kPatternTypes[dmr::kSyncPatterns] = {
    dmr::Sync::BsVoice,  dmr::Sync::BsData,  dmr::Sync::MsVoice,  dmr::Sync::MsData,
    dmr::Sync::Ts1Voice, dmr::Sync::Ts1Data, dmr::Sync::Ts2Voice, dmr::Sync::Ts2Data,
};

std::vector<uint64_t> sync_patterns()
{
    std::vector<uint64_t> patterns;
    for (dmr::Sync type : kPatternTypes)
        patterns.push_back(dmr::sync_pattern(type));
    return patterns;
}

uint32_t read_bits(const uint8_t* bits, size_t first, size_t count)
{
    uint32_t v = 0;
    for (size_t i = 0; i < count; ++i)
        v = (v << 1) | bits[first + i];
    return v;
}

}  // namespace

std::string dmr_destination(const DmrCall& call)
{
    if (!call.ids_known)
        return {};
    return (call.group ? "TG " : "") + std::to_string(call.destination);
}

std::string describe_dmr_slot(const DmrStatus& status, int slot)
{
    const DmrCall& call = status.slot[slot];
    if (call.active) {
        std::string text = call.encrypted ? "Encrypted " : "";
        text += call.ids_known ? (call.group ? "group call" : "private call") : "voice call";
        if (text[0] >= 'a')
            text[0] = static_cast<char>(text[0] - 'a' + 'A');
        if (call.ids_known)
            text += " to " + dmr_destination(call) + " from " + std::to_string(call.source);
        return text;
    }
    std::string text;
    switch (status.data_type[slot]) {
    case -1: text = "Nothing heard"; break;
    case dmr::kIdle: text = "Idle"; break;
    case dmr::kCsbk:
    case dmr::kMbcHeader:
    case dmr::kMbcContinuation: text = "Control (CSBK)"; break;
    case dmr::kVoiceLcHeader:
    case dmr::kTerminatorWithLc:
    case dmr::kPiHeader: text = "Idle"; break;  // a call's end
    default: text = "Data"; break;
    }
    if (call.ids_known)
        text += "; last call to " + dmr_destination(call) + " from " + std::to_string(call.source);
    return text;
}

// Enough history for a frame and a half, from a sync pattern found late.
DmrDecoder::DmrDecoder(double sample_rate)
    : front_(sample_rate, sync_patterns(), 1.5 * dmr::kFrameSymbols), sps_(front_.sps())
{
    reset();
}

void DmrDecoder::reset()
{
    front_.reset();
    sps_              = front_.sps();
    locked_           = false;
    frame_count_      = 0;
    last_valid_frame_ = 0;
    last_base_frame_  = 0;
    last_slot_        = 1;
    slots_            = {};
    status_           = {};
    valid_seen_       = false;
}

void DmrDecoder::process(const float* frequency, size_t count, double sample_rate, std::vector<DmrVoice>& voice)
{
    if (sample_rate > 0.0) {
        front_.set_sample_rate(sample_rate);
        sps_ = front_.sps();
    }
    Fsk4Front::Sync sync;
    for (size_t i = 0; i < count; ++i) {
        if (front_.push(frequency[i], sync))
            found_sync(sync);
        if (locked_)
            run_clock(voice);
    }
    status_.present = locked_ && valid_seen_;
}

void DmrDecoder::found_sync(const Fsk4Front::Sync& sync)
{
    if (!locked_) {
        anchor(sync);
        return;
    }
    // Where the clock would put the sync pattern of a frame.
    const double frame    = static_cast<double>(dmr::kFrameSymbols) * sps_;
    const double expected = next_time_ + (static_cast<double>(dmr::kSyncStart) - static_cast<double>(position_)) * sps_;
    double       delta    = sync.time - expected;
    delta -= frame * std::round(delta / frame);
    if (std::fabs(delta) <= 1.5 * sps_) {
        next_time_ += 0.5 * delta;
        level_ += 0.5 * (sync.level - level_);
        dc_ += 0.5 * (sync.dc - dc_);
    } else if (frame_count_ - last_valid_frame_ >= 2) {
        anchor(sync);  // the clock has gone astray
    }
}

void DmrDecoder::anchor(const Fsk4Front::Sync& sync)
{
    if (!locked_) {
        last_valid_frame_ = frame_count_;
        valid_seen_       = false;
    }
    locked_    = true;
    next_time_ = sync.time - static_cast<double>(dmr::kSyncStart) * sps_;
    position_  = 0;
    level_     = sync.level;
    dc_        = sync.dc;
    timing_error_ = level_error_ = dc_error_ = 0.0;
    outer_count_    = 0;
    previous_value_ = 0.0;
    // The frame must still be in the history.
    while (next_time_ - 0.5 * sps_ < front_.oldest())
        next_time_ += static_cast<double>(dmr::kFrameSymbols) * sps_;
}

void DmrDecoder::run_clock(std::vector<DmrVoice>& voice)
{
    const double latest = static_cast<double>(front_.written() - 1);
    while (locked_ && next_time_ + 0.5 * sps_ + 1.0 <= latest) {
        const double t      = next_time_;
        const double value  = front_.sample(t) - dc_;
        const int    symbol = Fsk4Front::decide(value, level_);
        dibits_[position_] = dmr::dibit_for_symbol(symbol);

        // Gardner's timing error: the value half way between two symbols
        // leans towards the later one when the clock is late.
        const double middle = front_.sample(t - 0.5 * sps_) - dc_;
        timing_error_ += (value - previous_value_) * middle;
        previous_value_ = value;
        dc_error_ += value - symbol * level_ / 3.0;
        if (symbol == 3 || symbol == -3) {
            level_error_ += std::fabs(value) - level_;
            ++outer_count_;
        }

        next_time_ += sps_;
        if (++position_ == dmr::kFrameSymbols) {
            decode_frame(voice);
            position_     = 0;
            timing_error_ = level_error_ = dc_error_ = 0.0;
            outer_count_  = 0;
        }
    }
}

void DmrDecoder::decode_frame(std::vector<DmrVoice>& voice)
{
    ++frame_count_;
    ++status_.frames;
    uint8_t bits[2 * dmr::kFrameSymbols];
    for (size_t i = 0; i < dmr::kFrameSymbols; ++i) {
        bits[2 * i]     = static_cast<uint8_t>(dibits_[i] >> 1);
        bits[2 * i + 1] = static_cast<uint8_t>(dibits_[i] & 1u);
    }
    const uint8_t* cach  = bits;
    const uint8_t* burst = bits + dmr::kCachBits;

    // What is in the middle of the burst.
    uint64_t middle = 0;
    for (size_t i = 108; i < 156; ++i)
        middle = (middle << 1) | burst[i];
    dmr::Sync sync        = dmr::Sync::None;
    int       sync_errors = 49;
    for (dmr::Sync type : kPatternTypes) {
        const int d = std::popcount(middle ^ dmr::sync_pattern(type));
        if (d < sync_errors) {
            sync_errors = d;
            sync        = type;
        }
    }
    const dmr::Sync nearest_sync = sync;
    if (sync_errors > kMaxSyncBitErrors)
        sync = dmr::Sync::None;
    if (dmr::from_base_station(sync))
        last_base_frame_ = frame_count_;
    const bool repeater = last_base_frame_ > 0 && frame_count_ - last_base_frame_ < kLostAfterFrames;

    // Which slot.  Slots alternate; a repeater also says which in the CACH
    // (a perfect Hamming code always decodes, so only an exact one is trusted).
    int slot = 1 - last_slot_;
    if (sync == dmr::Sync::Ts1Voice || sync == dmr::Sync::Ts1Data) {
        slot = 0;
    } else if (sync == dmr::Sync::Ts2Voice || sync == dmr::Sync::Ts2Data) {
        slot = 1;
    } else if (repeater) {
        const auto tact_word = static_cast<uint8_t>((cach[0] << 6) | (cach[4] << 5) | (cach[8] << 4) | (cach[12] << 3) |
                                                    (cach[14] << 2) | (cach[18] << 1) | cach[22]);
        uint8_t tact = 0;
        if (dmr::hamming743_decode(tact_word, tact) == 0)
            slot = (tact >> 2) & 1;
    } else if (sync == dmr::Sync::MsVoice || sync == dmr::Sync::MsData) {
        slot = 0;
    }
    last_slot_ = slot;
    Slot& s    = slots_[static_cast<size_t>(slot)];

    bool valid = false;
    if (dmr::is_voice(sync)) {
        valid = voice_burst(slot, burst, true, 0, 0, voice);
    } else if (sync != dmr::Sync::None) {
        valid = data_burst(slot, burst);
    } else if (s.call.active && s.voice_position == 5 && dmr::is_voice(nearest_sync) &&
               sync_errors <= kMaxLateSyncErrors) {
        voice_burst(slot, burst, true, 0, 0, voice);  // burst A was due: a damaged sync pattern
    } else {
        const auto emb_word = static_cast<uint16_t>((read_bits(burst, 108, 8) << 8) | read_bits(burst, 148, 8));
        uint8_t    emb      = 0;
        const int  errors   = dmr::qr1676_decode(emb_word, emb);
        const bool due      = s.call.active && s.voice_position >= 0 && s.voice_position < 5;
        const bool colour   = status_.color_code < 0 || (emb >> 3) == status_.color_code;
        // Without a superframe that expects it, only a perfect EMB counts.
        if (errors >= 0 && colour && (due || errors == 0))
            valid = voice_burst(slot, burst, false, errors, emb, voice);
        else if (due)
            voice_burst(slot, burst, false, -1, 0, voice);  // the voice frames have their own protection
    }

    if (valid) {
        valid_seen_       = true;
        last_valid_frame_ = frame_count_;
        // Fine-tune the clock and the levels from this frame's symbols.
        if (level_ > 0.0) {
            const double e = timing_error_ / (static_cast<double>(dmr::kFrameSymbols) * level_ * level_);
            next_time_ -= kTimingGain * std::clamp(e, -0.5, 0.5) * sps_;
        }
        dc_ += kLevelGain * dc_error_ / static_cast<double>(dmr::kFrameSymbols);
        if (outer_count_ > 16)
            level_ += kLevelGain * level_error_ / outer_count_;
    }

    for (int i = 0; i < 2; ++i) {
        if (slots_[static_cast<size_t>(i)].call.active &&
            frame_count_ - slots_[static_cast<size_t>(i)].last_activity > kCallTimeoutFrames)
            end_call(i);
        status_.slot[i] = slots_[static_cast<size_t>(i)].call;
    }
    status_.repeater = repeater;
    status_.level_hz = static_cast<float>(level_);
    if (frame_count_ - last_valid_frame_ >= kLostAfterFrames)
        lose_lock();
}

bool DmrDecoder::voice_burst(int slot, const uint8_t* burst, bool burst_a, int emb_errors, uint8_t emb,
                             std::vector<DmrVoice>& voice)
{
    Slot& s = slots_[static_cast<size_t>(slot)];
    if (!s.call.active) {
        s.call           = {};
        s.call.active    = true;
        s.started_output = false;
        s.embedded_have  = 0;
        s.voice_position = -1;
    }
    s.call.voice    = true;
    s.last_activity = frame_count_;
    ++status_.voice_bursts;

    if (burst_a) {
        s.voice_position = 0;
        s.embedded_have  = 0;
    } else if (emb_errors >= 0) {
        const int lcss = emb & 3;
        if ((emb >> 2) & 1)
            s.call.encrypted = true;
        status_.color_code = emb >> 3;
        if (s.voice_position >= 0 && s.voice_position < 5)
            ++s.voice_position;
        else
            s.voice_position = lcss == dmr::kFirst ? 1 : lcss == dmr::kLast ? 4 : lcss == dmr::kSingle ? 5 : 2;

        // The link control, a quarter at a time in bursts B-E.
        const auto fragment = [&](int index) {
            for (size_t i = 0; i < 32; ++i)
                s.embedded[static_cast<size_t>(32 * index) + i] = burst[116 + i];
        };
        if (lcss == dmr::kFirst) {
            fragment(0);
            s.embedded_have = 1;
        } else if (lcss == dmr::kContinuation && (s.embedded_have == 1 || s.embedded_have == 2)) {
            fragment(s.embedded_have);
            ++s.embedded_have;
        } else if (lcss == dmr::kLast && s.embedded_have == 3) {
            fragment(3);
            dmr::LinkControl lc;
            if (dmr::decode_embedded_lc(s.embedded, lc) && lc.voice()) {
                s.call.ids_known   = true;
                s.call.group       = lc.group();
                s.call.destination = lc.destination;
                s.call.source      = lc.source;
                if (lc.encrypted())
                    s.call.encrypted = true;
            }
            s.embedded_have = 0;
        } else {
            s.embedded_have = 0;
        }
    } else {
        if (s.voice_position >= 0 && s.voice_position < 5)
            ++s.voice_position;
        s.embedded_have = 0;
    }

    // The first call to speak is heard, until it ends.
    if (status_.playing < 0)
        status_.playing = slot;
    if (status_.playing == slot && !s.call.encrypted) {
        uint8_t bits[3][72];
        std::copy(burst, burst + 72, bits[0]);
        std::copy(burst + 72, burst + 108, bits[1]);
        std::copy(burst + 156, burst + 192, bits[1] + 36);
        std::copy(burst + 192, burst + 264, bits[2]);
        for (const auto& b : bits) {
            DmrVoice v;
            dmr::ambe_deinterleave(b, v.frame);
            v.slot           = slot;
            v.first          = !s.started_output;
            s.started_output = true;
            voice.push_back(v);
            ++status_.voice_frames;
        }
    }
    return burst_a || emb_errors >= 0;
}

bool DmrDecoder::data_burst(int slot, const uint8_t* burst)
{
    const uint32_t slot_type = (read_bits(burst, 98, 10) << 10) | read_bits(burst, 156, 10);
    uint8_t        info      = 0;
    if (dmr::golay2087_decode(slot_type, info) < 0)
        return false;
    status_.color_code = info >> 4;
    const int type     = info & 0xF;
    status_.data_type[slot] = type;
    ++status_.data_bursts;

    std::array<uint8_t, 196> payload{};
    std::copy(burst, burst + 98, payload.begin());
    std::copy(burst + 166, burst + 264, payload.begin() + 98);

    Slot&      s  = slots_[static_cast<size_t>(slot)];
    const auto start_call = [&] {
        s.call           = {};
        s.call.active    = true;
        s.started_output = false;
        s.voice_position = -1;
        s.embedded_have  = 0;
    };
    const auto take = [&](const dmr::LinkControl& lc) {
        s.call.ids_known   = true;
        s.call.group       = lc.group();
        s.call.destination = lc.destination;
        s.call.source      = lc.source;
        if (lc.encrypted())
            s.call.encrypted = true;
    };

    switch (type) {
    case dmr::kPiHeader:
        if (!s.call.active)
            start_call();
        s.call.encrypted = true;
        s.last_activity  = frame_count_;
        break;
    case dmr::kVoiceLcHeader: {
        dmr::LinkControl lc;
        if (dmr::decode_full_lc(payload, dmr::kVoiceLcHeaderMask, lc) && lc.voice()) {
            const bool other = s.call.ids_known && (lc.destination != s.call.destination || lc.source != s.call.source);
            if (!s.call.active || other) {
                if (other && status_.playing == slot)
                    status_.playing = -1;
                start_call();
            }
            take(lc);
            s.last_activity = frame_count_;
        }
        break;
    }
    case dmr::kTerminatorWithLc: {
        dmr::LinkControl lc;
        if (dmr::decode_full_lc(payload, dmr::kTerminatorMask, lc) && lc.voice() && !s.call.ids_known)
            take(lc);
        end_call(slot);
        break;
    }
    default:
        // Anything else on the slot: whatever call there was is over.
        if (s.call.active)
            end_call(slot);
        break;
    }
    return true;
}

void DmrDecoder::end_call(int slot)
{
    Slot& s          = slots_[static_cast<size_t>(slot)];
    s.call.active    = false;
    s.call.voice     = false;
    s.voice_position = -1;
    s.embedded_have  = 0;
    s.started_output = false;
    if (status_.playing == slot)
        status_.playing = -1;
    status_.slot[slot] = s.call;
}

void DmrDecoder::lose_lock()
{
    locked_     = false;
    valid_seen_ = false;
    slots_      = {};
    DmrStatus fresh;
    fresh.frames       = status_.frames;
    fresh.data_bursts  = status_.data_bursts;
    fresh.voice_bursts = status_.voice_bursts;
    fresh.voice_frames = status_.voice_frames;
    status_            = fresh;
}

}  // namespace spectro
