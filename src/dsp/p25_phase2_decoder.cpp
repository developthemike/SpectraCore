#include "dsp/p25_phase2_decoder.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace spectro {

namespace {

constexpr double kLostAfterSeconds   = 1.0;
constexpr double kCallTimeoutSeconds = 0.8;   // without voice or signalling from a call
constexpr double kSyncTimingGain     = 0.8;
constexpr size_t kTimingBlock        = 36;
constexpr double kTimingGain         = 0.3;
constexpr double kLevelGain          = 0.5;
constexpr int    kMissesToLose       = 6;     // ISCHs in a row not decoded: the signal has gone
constexpr size_t kMaxHeld            = 18;    // a superframe's voice: the signalling comes in every one
constexpr size_t kMaxCandidates      = 4;     // systems worked out, waiting for a second burst to agree

// The frequency averaged over a symbol is the phase step over it: +-45 or
// +-135 degrees, +-750 or +-2250 Hz.  The pulse shaping can carry the phase on
// past +-180 degrees in between, which the discriminator then counts as a full
// turn more or less; a step is only known modulo a turn, 6000 Hz.
double wrap_step(double hz)
{
    return hz - p25p2::kSymbolRate * std::round(hz / p25p2::kSymbolRate);
}

}  // namespace

std::string describe_p25_phase2_slot(const P25Phase2Status& status, int slot)
{
    const P25Call& call = status.slot[slot];
    if (!call.active) {
        if (call.ids_known)
            return "Idle; last call to " + p25_destination(call) + (call.source ? " from " + std::to_string(call.source) : "");
        return "Idle";
    }
    std::string text = call.encrypted ? "Encrypted " : "";
    text += call.ids_known ? (call.group ? "group call" : "private call") : "voice call";
    if (text[0] >= 'a')
        text[0] = static_cast<char>(text[0] - 'a' + 'A');
    if (call.ids_known)
        text += " to " + p25_destination(call) + (call.source ? " from " + std::to_string(call.source) : "");
    if (!status.keyed)
        text += " (scrambled: system not known yet)";
    return text;
}

P25Phase2Decoder::P25Phase2Decoder(double sample_rate)
    : front_(sample_rate, {p25p2::kSync}, 2.0 * p25p2::kSlotDibits,
             Fsk4Front::Options{p25p2::kSymbolRate, 20, true}),
      sps_(front_.sps()),
      rate_(sample_rate)
{
    reset();
}

void P25Phase2Decoder::reset()
{
    front_.reset();
    sps_      = front_.sps();
    locked_   = false;
    position_ = 0;
    timeslot_ = -1;
    misses_   = 0;
    slots_    = {};
    mask_.clear();
    pending_.clear();
    candidates_.clear();
    last_valid_ = 0;
    valid_seen_ = false;
    status_     = {};
}

void P25Phase2Decoder::set_systems(const std::vector<p25p2::System>& systems)
{
    if (systems == systems_)
        return;
    systems_ = systems;
    masks_.clear();
    for (const p25p2::System& system : systems_)
        masks_.push_back(p25p2::scrambling(system));
    // A system already in use stays unless it is no longer offered (one
    // worked out from the channel stays regardless).
    if (status_.keyed && !status_.recovered &&
        std::find(systems_.begin(), systems_.end(), status_.system) == systems_.end()) {
        status_.keyed = false;
        mask_.clear();
    }
}

double P25Phase2Decoder::seconds_since(uint64_t sample) const
{
    return static_cast<double>(front_.written() - std::min(sample, front_.written())) / rate_;
}

void P25Phase2Decoder::process(const float* frequency, size_t count, double sample_rate,
                               std::vector<P25Phase2Voice>& voice)
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
    for (int c = 0; c < 2; ++c)
        if (slots_[static_cast<size_t>(c)].call.active &&
            seconds_since(slots_[static_cast<size_t>(c)].last_activity) > kCallTimeoutSeconds)
            end_call(c);
    if (valid_seen_ && seconds_since(last_valid_) > kLostAfterSeconds) {
        lose_lock();
        valid_seen_ = false;
        for (int c = 0; c < 2; ++c)
            end_call(c);
    }
    status_.present = valid_seen_;
    for (int c = 0; c < 2; ++c)
        status_.slot[c] = slots_[static_cast<size_t>(c)].call;
}

// An S-ISCH starts a timeslot.  It comes in six of the twelve, so the clock is
// set right often.
void P25Phase2Decoder::found_sync(const Fsk4Front::Sync& sync)
{
    const double start = next_time_ - static_cast<double>(position_) * sps_;
    if (locked_ && std::fabs(sync.time - start) <= 0.6 * sps_) {
        next_time_ += kSyncTimingGain * (sync.time - start);
        level_ += kLevelGain * (sync.level - level_);
        dc_ += kLevelGain * (sync.dc - dc_);
        return;
    }
    if (locked_ && position_ > 0) {
        // Out of step with the timeslots: start again from here, and work out
        // where in the superframe from the next I-ISCH.
        lose_lock();
    }
    locked_    = true;
    misses_    = 0;
    next_time_ = std::max(sync.time, front_.oldest() + 0.5 * sps_);
    position_  = 0;
    level_     = sync.level;
    dc_        = sync.dc;
    timing_error_ = previous_value_ = 0.0;
    timing_count_ = 0;
}

void P25Phase2Decoder::run_clock(std::vector<P25Phase2Voice>& voice)
{
    const double latest = static_cast<double>(front_.written() - 1);
    while (locked_ && next_time_ + 1.0 <= latest) {
        const double value  = wrap_step(front_.sample(next_time_) - dc_);
        const int    symbol = Fsk4Front::decide(value, level_);
        dibits_[position_]  = symbol == 3 ? 1 : symbol == 1 ? 0 : symbol == -1 ? 2 : 3;

        // As in the Phase 1 decoder: Gardner's timing error over blocks of symbols.
        const double middle = wrap_step(front_.sample(next_time_ - 0.5 * sps_) - dc_);
        timing_error_ += (value - previous_value_) * middle;
        previous_value_ = value;
        next_time_ += sps_;
        if (++timing_count_ == kTimingBlock) {
            if (level_ > 0.0)
                next_time_ -= kTimingGain * std::clamp(timing_error_ / (kTimingBlock * level_ * level_), -0.5, 0.5) * sps_;
            timing_error_ = 0.0;
            timing_count_ = 0;
        }
        if (++position_ == p25p2::kSlotDibits) {
            decode_timeslot(voice);
            position_ = 0;
        }
    }
}

void P25Phase2Decoder::lose_lock()
{
    locked_   = false;
    timeslot_ = -1;
    misses_   = 0;
    pending_.clear();
}

void P25Phase2Decoder::decode_timeslot(std::vector<P25Phase2Voice>& voice)
{
    // Where in the superframe: the I-ISCH says, the S-ISCH follows on.
    uint64_t isch = 0;
    for (size_t d = 0; d < 20; ++d)
        isch = (isch << 2) | dibits_[d];
    const int  value = p25p2::isch_decode(isch);
    const bool good  = value != -1;
    if (good) {
        misses_     = 0;
        last_valid_ = front_.written();
        valid_seen_ = true;
        ++status_.timeslots;
        status_.level_hz = static_cast<float>(level_);
    } else if (++misses_ >= kMissesToLose) {
        lose_lock();
        return;
    }
    const bool was_known = timeslot_ >= 0;
    if (value >= 0)
        timeslot_ = p25p2::isch_slot(static_cast<unsigned>(value));
    else if (timeslot_ >= 0)
        timeslot_ = (timeslot_ + 1) % static_cast<int>(p25p2::kSlots);
    if (timeslot_ >= static_cast<int>(p25p2::kSlots))
        timeslot_ = -1;

    Timeslot t;
    t.serial   = ++serial_;
    t.dibits   = dibits_;
    t.number   = timeslot_;
    t.good     = good;
    t.readable = good || misses_ == 1;
    pending_.push_back(t);
    if (pending_.size() > p25p2::kSlots)
        pending_.erase(pending_.begin());
    if (timeslot_ < 0)
        return;
    if (!was_known) {
        // The timeslots before this one, back to the last lock, were the ones before it.
        const int n = static_cast<int>(pending_.size());
        for (int i = 0; i < n; ++i)
            if (pending_[static_cast<size_t>(i)].number < 0)
                pending_[static_cast<size_t>(i)].number =
                    ((timeslot_ - (n - 1 - i)) % static_cast<int>(p25p2::kSlots) + static_cast<int>(p25p2::kSlots)) %
                    static_cast<int>(p25p2::kSlots);
    }

    if (!status_.keyed) {
        // The timeslots just numbered, or else the new one: the rest have been tried.
        if (!was_known) {
            for (auto it = pending_.rbegin(); it != pending_.rend() && !status_.keyed; ++it)
                try_systems(*it);
        } else {
            try_systems(pending_.back());
        }
    }
    if (!status_.keyed) {
        // Unscrambled signalling can be read now; of the rest, only that there is a call.
        if (const int burst = p25p2::duid_decode(static_cast<uint8_t>(
                (t.dibits[p25p2::kDuidAt[0]] << 6) | (t.dibits[p25p2::kDuidAt[1]] << 4) |
                (t.dibits[p25p2::kDuidAt[2]] << 2) | t.dibits[p25p2::kDuidAt[3]]));
            burst >= 0 && !p25p2::is_scrambled(static_cast<uint8_t>(burst))) {
            read_timeslot(t, voice);
            pending_.pop_back();
        } else {
            note_unreadable(t);
        }
        return;
    }
    for (const Timeslot& p : pending_)
        read_timeslot(p, voice);
    pending_.clear();
}

namespace {

uint8_t duid_code(const std::array<uint8_t, p25p2::kSlotDibits>& d)
{
    return static_cast<uint8_t>((d[p25p2::kDuidAt[0]] << 6) | (d[p25p2::kDuidAt[1]] << 4) | (d[p25p2::kDuidAt[2]] << 2) |
                                d[p25p2::kDuidAt[3]]);
}

int burst_of(const std::array<uint8_t, p25p2::kSlotDibits>& d) { return p25p2::duid_decode(duid_code(d)); }

}  // namespace

// The burst's type.  Its code mends one error; with two it is ambiguous, and
// then, during a call's voice, the type its place in the superframe has (4V
// four times, then 2V) is taken if it is one of the nearest.
int P25Phase2Decoder::burst_type(const Timeslot& t) const
{
    const uint8_t code  = duid_code(t.dibits);
    const int     burst = p25p2::duid_decode(code);
    if (burst >= 0 || t.number < 0 || t.number >= 10)
        return burst;
    const Slot& s = slots_[static_cast<size_t>(p25p2::kChannelOf[t.number])];
    if (!s.call.active || !s.call.voice)
        return -1;
    const uint8_t expected = t.number / 2 == 4 ? p25p2::k2V : p25p2::k4V;
    return std::popcount(static_cast<unsigned>(code ^ p25p2::duid_codeword(expected))) <= 2 ? expected : -1;
}

// Voice we cannot read is still a call.
void P25Phase2Decoder::note_unreadable(const Timeslot& t)
{
    if (!t.readable || t.number < 0)
        return;
    const int burst = burst_type(t);
    if (burst != p25p2::k4V && burst != p25p2::k2V)
        return;
    Slot& s = slots_[static_cast<size_t>(p25p2::kChannelOf[t.number])];
    if (!s.call.active) {
        if (!t.good)
            return;
        start_call(p25p2::kChannelOf[t.number]);
    }
    s.call.voice    = true;
    s.last_activity = front_.written();
}

void P25Phase2Decoder::read_timeslot(const Timeslot& t, std::vector<P25Phase2Voice>& voice)
{
    if (!t.readable || t.number < 0)
        return;
    const int burst = burst_type(t);
    if (burst < 0)
        return;
    const int channel = p25p2::kChannelOf[t.number];
    Slot&     s       = slots_[static_cast<size_t>(channel)];

    // Descramble.
    std::array<uint8_t, p25p2::kSlotDibits> d = t.dibits;
    if (p25p2::is_scrambled(static_cast<uint8_t>(burst))) {
        const size_t base = static_cast<size_t>(t.number) * p25p2::kSlotDibits;
        for (size_t i = 10; i < p25p2::kSlotDibits; ++i)
            d[i] = static_cast<uint8_t>(d[i] ^ mask_[base + i - 10]);
    }

    switch (burst) {
    case p25p2::k4V:
    case p25p2::k2V: {
        if (!s.call.active) {
            if (!t.good)  // a call does not start on a timeslot that may be noise
                return;
            start_call(channel);
        }
        s.call.voice    = true;
        s.last_activity = front_.written();
        if (status_.playing < 0 && (wanted_slot_ < 0 || wanted_slot_ == channel) &&
            !(s.encryption_known && s.call.encrypted))
            status_.playing = channel;
        if (status_.playing != channel || (s.encryption_known && s.call.encrypted))
            break;
        const size_t frames = burst == p25p2::k4V ? 4 : 2;
        for (size_t f = 0; f < frames; ++f) {
            uint8_t bits[72];
            for (size_t k = 0; k < 36; ++k) {
                bits[2 * k]     = static_cast<uint8_t>(d[p25p2::kVoiceAt[f] + k] >> 1);
                bits[2 * k + 1] = static_cast<uint8_t>(d[p25p2::kVoiceAt[f] + k] & 1u);
            }
            P25Phase2Voice v;
            dmr::ambe_deinterleave(bits, v.frame);
            v.slot = channel;
            if (s.encryption_known) {
                output(s, v, voice);
                continue;
            }
            // Joined after the call's start: wait for the signalling to say
            // whether it can be listened to - though not beyond a superframe.
            s.held.push_back(v);
            if (s.held.size() > kMaxHeld) {
                s.encryption_known = true;
                for (const P25Phase2Voice& h : s.held)
                    output(s, h, voice);
                s.held.clear();
            }
        }
        break;
    }
    case p25p2::kFacch:
    case p25p2::kFacchClear:
    case p25p2::kSacch:
    case p25p2::kSacchClear: {
        std::vector<uint8_t> pdu;
        const bool           facch = burst == p25p2::kFacch || burst == p25p2::kFacchClear;
        if (p25p2::decode_mac(d.data(), facch, pdu))
            apply_mac(channel, pdu, voice);
        break;
    }
    default:
        break;
    }
}

void P25Phase2Decoder::key(const p25p2::System& system, const std::vector<uint8_t>& mask, bool recovered)
{
    mask_             = mask;
    status_.keyed     = true;
    status_.system    = system;
    status_.recovered = recovered;
    candidates_.clear();
}

// The first system whose scrambling makes a signalling burst check: one of
// those given, or one worked out from an earlier burst.  Failing that, the
// system this burst says, if it came through clean.
bool P25Phase2Decoder::try_systems(const Timeslot& t)
{
    const int burst = burst_of(t.dibits);
    if (t.number < 0 || (burst != p25p2::kFacch && burst != p25p2::kSacch))
        return false;
    const bool   facch  = burst == p25p2::kFacch;
    const size_t base   = static_cast<size_t>(t.number) * p25p2::kSlotDibits;
    const auto   checks = [&](const std::vector<uint8_t>& mask) {
        std::array<uint8_t, p25p2::kSlotDibits> d = t.dibits;
        for (size_t i = 10; i < p25p2::kSlotDibits; ++i)
            d[i] = static_cast<uint8_t>(d[i] ^ mask[base + i - 10]);
        std::vector<uint8_t> pdu;
        return p25p2::decode_mac(d.data(), facch, pdu);
    };
    for (size_t k = 0; k < systems_.size(); ++k) {
        if (checks(masks_[k])) {
            key(systems_[k], masks_[k], false);
            return true;
        }
    }
    for (const Candidate& c : candidates_) {
        if (c.from != t.serial && checks(c.mask)) {
            key(c.system, c.mask, true);
            return true;
        }
    }
    if (const auto system = p25p2::recover_system(t.dibits.data(), t.number, facch)) {
        const bool known = std::any_of(candidates_.begin(), candidates_.end(),
                                       [&](const Candidate& c) { return c.system == *system; });
        if (!known) {
            if (candidates_.size() >= kMaxCandidates)
                candidates_.erase(candidates_.begin());
            candidates_.push_back({*system, p25p2::scrambling(*system), t.serial});
        }
    }
    return false;
}

void P25Phase2Decoder::output(Slot& s, P25Phase2Voice v, std::vector<P25Phase2Voice>& voice)
{
    v.first          = !s.started_output;
    s.started_output = true;
    voice.push_back(v);
    ++status_.voice_frames;
}

void P25Phase2Decoder::apply_mac(int channel, const std::vector<uint8_t>& pdu, std::vector<P25Phase2Voice>& voice)
{
    ++status_.macs;
    Slot&         s      = slots_[static_cast<size_t>(channel)];
    const uint8_t opcode = p25p2::mac_opcode(pdu);
    // Messages that do not say who is talking give 0 (or, ending a call, all
    // ones) for the source: that is no other radio.
    const auto given = [](uint32_t source) { return source != 0 && source != 0xFFFFFFu; };
    if (opcode == p25p2::kMacEndPtt || opcode == p25p2::kMacIdle) {
        if (const auto c = p25p2::mac_call(pdu); c && c->ids_known) {
            s.call.ids_known   = true;
            s.call.group       = c->group;
            s.call.destination = c->destination;
            if (given(c->source))
                s.call.source = c->source;
        }
        end_call(channel);
        return;
    }
    const auto c = p25p2::mac_call(pdu);
    if (!c)
        return;
    const bool other = s.call.ids_known && (c->destination != s.call.destination ||
                                            (given(c->source) && given(s.call.source) && c->source != s.call.source));
    if (opcode == p25p2::kMacPtt || !s.call.active || other) {
        // A new call (or transmission): its voice starts afresh.
        if (other && status_.playing == channel)
            status_.playing = -1;
        start_call(channel);
    }
    s.call.ids_known   = true;
    s.call.group       = c->group;
    s.call.destination = c->destination;
    if (given(c->source))
        s.call.source = c->source;
    if (c->encryption_known) {
        s.call.encrypted   = c->encrypted;
        s.encryption_known = true;
        if (s.call.encrypted && status_.playing == channel)
            status_.playing = -1;
        if (!s.call.encrypted && status_.playing == channel)
            for (const P25Phase2Voice& h : s.held)
                output(s, h, voice);
        s.held.clear();
    }
    s.last_activity = front_.written();
}

void P25Phase2Decoder::start_call(int channel)
{
    Slot& s          = slots_[static_cast<size_t>(channel)];
    s.call             = {};
    s.call.active      = true;
    s.started_output   = false;
    s.encryption_known = false;
    s.held.clear();
    s.last_activity  = front_.written();
}

void P25Phase2Decoder::end_call(int channel)
{
    Slot& s          = slots_[static_cast<size_t>(channel)];
    s.call.active    = false;
    s.call.voice     = false;
    s.started_output = false;
    s.held.clear();
    if (status_.playing == channel)
        status_.playing = -1;
}

}  // namespace spectro
