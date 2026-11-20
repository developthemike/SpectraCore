#include "dsp/p25_decoder.hpp"

#include <algorithm>
#include <cmath>

namespace spectro {

namespace {

constexpr double kLostAfterSeconds = 1.0;   // without a frame that decodes
constexpr double kCallTimeoutSeconds = 0.6; // without a voice frame, and no terminator
constexpr double kLevelGain = 0.5;          // of the frame sync, for the levels
constexpr double kSyncTimingGain = 0.8;     // a frame sync's timing, taken nearly at its word
constexpr size_t kTimingBlock = 36;         // symbols between timing corrections from the symbols
constexpr double kTimingGain = 0.3;

}  // namespace

std::string p25_destination(const P25Call& call)
{
    if (!call.ids_known)
        return {};
    return (call.group ? "TG " : "") + std::to_string(call.destination);
}

std::string describe_p25(const P25Status& status)
{
    const P25Call& call = status.call;
    if (call.active) {
        std::string text = call.encrypted ? "Encrypted " : "";
        text += call.ids_known ? (call.group ? "group call" : "private call") : "voice call";
        if (text[0] >= 'a')
            text[0] = static_cast<char>(text[0] - 'a' + 'A');
        if (call.ids_known)
            text += " to " + p25_destination(call) + " from " + std::to_string(call.source);
        return text;
    }
    std::string text = status.control ? "Control channel" : "Idle";
    if (call.ids_known)
        text += "; last call to " + p25_destination(call) + " from " + std::to_string(call.source);
    return text;
}

P25Decoder::P25Decoder(double sample_rate)
    : front_(sample_rate, {p25::kFrameSync}, 64.0), sps_(front_.sps()), rate_(sample_rate)
{
    dibits_.reserve(p25::frame_dibits(p25::kLdu1));
    reset();
}

void P25Decoder::reset()
{
    front_.reset();
    sps_    = front_.sps();
    locked_ = false;
    dibits_.clear();
    bits_.clear();
    tsbks_read_ = 0;
    position_ = length_ = 0;
    last_valid_ = last_voice_ = last_control_ = 0;
    valid_seen_     = false;
    started_output_ = false;
    status_         = {};
}

double P25Decoder::seconds_since(uint64_t sample) const
{
    return static_cast<double>(front_.written() - std::min(sample, front_.written())) / rate_;
}

void P25Decoder::process(const float* frequency, size_t count, double sample_rate, std::vector<P25Voice>& voice)
{
    if (sample_rate > 0.0) {
        front_.set_sample_rate(sample_rate);
        sps_  = front_.sps();
        rate_ = sample_rate;
    }
    Fsk4Front::Sync sync;
    for (size_t i = 0; i < count; ++i) {
        if (front_.push(frequency[i], sync))
            found_sync(sync);
        if (locked_)
            run_clock(voice);
    }

    if (status_.call.active && seconds_since(last_voice_) > kCallTimeoutSeconds)
        end_call();
    status_.control = valid_seen_ && last_control_ > 0 && seconds_since(last_control_) < kLostAfterSeconds;
    if (valid_seen_ && seconds_since(last_valid_) > kLostAfterSeconds) {
        // Gone: start afresh, but remember the last call.
        const P25Call last = status_.call;
        const uint64_t frames = status_.frames, voice_frames = status_.voice_frames, tsbks = status_.tsbks;
        P25Site               site   = status_.site;
        std::vector<P25Grant> grants = std::move(status_.grants);
        status_              = {};
        status_.call         = last;
        status_.site         = site;
        status_.grants       = std::move(grants);
        status_.tsbks        = tsbks;
        status_.call.active  = status_.call.voice = false;
        status_.frames       = frames;
        status_.voice_frames = voice_frames;
        valid_seen_          = false;
        locked_              = false;
    }
    status_.present = valid_seen_;
    status_.time    = static_cast<double>(front_.written()) / rate_;
}

void P25Decoder::found_sync(const Fsk4Front::Sync& sync)
{
    // The frame the clock is in: if this is its sync, a nudge; otherwise a
    // frame starts here.
    const double start = next_time_ - static_cast<double>(position_) * sps_;
    if (locked_ && std::fabs(sync.time - start) <= 0.6 * sps_) {
        next_time_ += kSyncTimingGain * (sync.time - start);
        level_ += kLevelGain * (sync.level - level_);
        dc_ += kLevelGain * (sync.dc - dc_);
        return;
    }
    anchor(sync);
}

void P25Decoder::anchor(const Fsk4Front::Sync& sync)
{
    locked_    = true;
    next_time_ = std::max(sync.time, front_.oldest() + 0.5 * sps_);
    position_  = 0;
    length_    = 0;
    tsbks_read_ = 0;
    level_     = sync.level;
    dc_        = sync.dc;
    dibits_.clear();
    bits_.clear();
    timing_error_ = previous_value_ = 0.0;
    timing_count_ = 0;
}

void P25Decoder::run_clock(std::vector<P25Voice>& voice)
{
    const double latest = static_cast<double>(front_.written() - 1);
    while (locked_ && next_time_ + 1.0 <= latest) {
        const double  value  = front_.sample(next_time_) - dc_;
        const int     symbol = Fsk4Front::decide(value, level_);
        const uint8_t dibit  = symbol == 3 ? 1 : symbol == 1 ? 0 : symbol == -1 ? 2 : 3;
        dibits_.push_back(dibit);
        if (!p25::is_status_dibit(position_)) {
            bits_.push_back(static_cast<uint8_t>(dibit >> 1));
            bits_.push_back(static_cast<uint8_t>(dibit & 1u));
        }

        // Gardner's timing error: the value half way between two symbols
        // leans towards the later one when the clock is late.  A frame is
        // long enough for the clocks to drift apart within it.
        const double middle = front_.sample(next_time_ - 0.5 * sps_) - dc_;
        timing_error_ += (value - previous_value_) * middle;
        previous_value_ = value;
        ++position_;
        next_time_ += sps_;
        if (++timing_count_ == kTimingBlock) {
            if (level_ > 0.0) {
                const double e = timing_error_ / (static_cast<double>(kTimingBlock) * level_ * level_);
                next_time_ -= kTimingGain * std::clamp(e, -0.5, 0.5) * sps_;
            }
            timing_error_ = 0.0;
            timing_count_ = 0;
        }

        if (position_ == p25::kNidEndDibit && !read_nid()) {
            locked_ = false;  // not a frame after all, or the clock has run on too far
            return;
        }
        // Trunking blocks are read as they arrive: a TSDU holds one to three.
        if (length_ > 0 && duid_ == p25::kTsdu && bits_.size() == 112 + 196 * (tsbks_read_ + 1)) {
            if (read_tsbk())  // the last block: the frame ends here (with a status symbol due)
                length_ = position_ + (p25::is_status_dibit(position_) ? 1 : 0);
        }
        if (length_ > 0 && position_ == length_) {
            decode_frame(voice);
            // The next frame follows at once.
            position_    = 0;
            length_      = 0;
            tsbks_read_  = 0;
            dibits_.clear();
            bits_.clear();
        }
    }
}

bool P25Decoder::read_nid()
{
    uint64_t nid = 0;
    for (size_t d = 24; d < p25::kNidEndDibit; ++d) {
        if (!p25::is_status_dibit(d))
            nid = (nid << 2) | dibits_[d];
    }
    uint16_t nac  = 0;
    uint8_t  duid = 0;
    if (p25::nid_decode(nid, nac, duid) < 0)
        return false;
    length_      = p25::frame_dibits(duid);
    duid_        = duid;
    status_.duid = duid;
    if (length_ == 0) {
        // Packet data: of no fixed length; wait for the next frame sync.
        status_.nac = nac;
        last_valid_ = front_.written();
        valid_seen_ = true;
        ++status_.frames;
        return false;
    }
    status_.nac = nac;
    return true;
}

void P25Decoder::decode_frame(std::vector<P25Voice>& voice)
{
    last_valid_ = front_.written();
    valid_seen_ = true;
    ++status_.frames;
    status_.level_hz = static_cast<float>(level_);

    const std::vector<uint8_t>& bits = bits_;  // the frame's, without its status symbols

    P25Call&   call       = status_.call;
    const auto start_call = [&] {
        call            = {};
        call.active     = true;
        started_output_ = false;
    };

    switch (duid_) {
    case p25::kHdu:
        if (!call.active)
            start_call();
        last_voice_ = front_.written();
        break;
    case p25::kLdu1:
    case p25::kLdu2: {
        if (!call.active)
            start_call();
        call.voice  = true;
        last_voice_ = front_.written();
        std::array<uint8_t, 240> link{};
        for (size_t b = 0; b < 6; ++b)
            std::copy(bits.begin() + static_cast<long>(p25::kLinkStart[b]),
                      bits.begin() + static_cast<long>(p25::kLinkStart[b] + 40), link.begin() + 40 * static_cast<long>(b));
        if (duid_ == p25::kLdu1) {
            p25::LinkControl lc;
            if (p25::decode_link_control(link, lc) && lc.voice() && lc.mfid <= 1) {
                const bool other = call.ids_known && (lc.destination != call.destination || lc.source != call.source);
                if (other)
                    start_call();  // another call without a terminator between
                call.active      = call.voice = true;
                call.ids_known   = true;
                call.group       = lc.group();
                call.destination = lc.destination;
                call.source      = lc.source;
                if (lc.encrypted())
                    call.encrypted = true;
            }
        } else {
            p25::EncryptionSync es;
            if (p25::decode_encryption_sync(link, es))
                call.encrypted = es.encrypted();
        }
        if (!call.encrypted)
            voice_frames(bits, voice);
        break;
    }
    case p25::kTdu:
    case p25::kTdulc:
        end_call();
        break;
    case p25::kTsdu:
        last_control_ = front_.written();
        break;
    default:
        break;
    }
}

// The latest trunking block of a TSDU.  Returns true if it was the last.
bool P25Decoder::read_tsbk()
{
    const size_t start = 112 + 196 * tsbks_read_;
    ++tsbks_read_;
    p25::Tsbk block{};
    if (p25::decode_tsbk(bits_.data() + start, block) < 0)
        return tsbks_read_ >= 3;
    last_control_ = front_.written();
    ++status_.tsbks;
    apply_tsbk(block, status_.site, status_.grants, static_cast<double>(front_.written()) / rate_);
    return (block[0] & 0x80u) != 0 || tsbks_read_ >= 3;
}

void P25Decoder::voice_frames(const std::vector<uint8_t>& bits, std::vector<P25Voice>& voice)
{
    for (const size_t start : p25::kVoiceStart) {
        P25Voice v;
        p25::imbe_deinterleave(bits.data() + start, v.frame);
        v.first         = !started_output_;
        started_output_ = true;
        voice.push_back(v);
        ++status_.voice_frames;
    }
}

void P25Decoder::end_call()
{
    status_.call.active = false;
    status_.call.voice  = false;
    started_output_     = false;
}

}  // namespace spectro
