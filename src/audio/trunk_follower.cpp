#include "audio/trunk_follower.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace spectro {

namespace {

constexpr double kSameHz       = 1e3;  // the listener is where it was sent
constexpr double kControlNearHz = 5e3; // the control channel as announced, against where it was found

}  // namespace

void TrunkFollower::set_enabled(bool enabled)
{
    if (enabled == this->enabled())
        return;
    state_ = enabled ? State::Waiting : State::Off;
    recent_.clear();
}

bool TrunkFollower::follows(const P25Grant& grant, const P25Status& p25, double now) const
{
    if (grant.frequency <= 0.0 || grant.encrypted || p25.time - grant.time > kFreshSeconds)
        return false;
    return std::none_of(recent_.begin(), recent_.end(), [&](const Recent& r) {
        return r.destination == grant.destination && r.channel == grant.channel && r.until > now;
    });
}

FollowAction TrunkFollower::back_to_control(double now, bool call_heard)
{
    recent_.push_back({call_.destination, call_.channel, now + (call_heard ? kAfterCallSeconds : kNotFoundSeconds)});
    state_    = State::Control;
    tuned_at_ = now;
    return {true, control_hz_, -1};
}

FollowAction TrunkFollower::update(double frequency, const P25Status& p25, const P25Phase2Status& p25p2, double now)
{
    if (state_ == State::Off || std::isnan(frequency))
        return {};
    recent_.erase(std::remove_if(recent_.begin(), recent_.end(), [&](const Recent& r) { return r.until <= now; }),
                  recent_.end());
    // A retune of ours takes a moment to show; any other is the user's.
    const bool settling = now - tuned_at_ < kSettleSeconds;

    switch (state_) {
    case State::Off:
        break;
    case State::Waiting:
        if (p25.present && p25.control) {
            state_      = State::Control;
            control_hz_ = frequency;
        }
        break;
    case State::Control: {
        if (std::fabs(frequency - control_hz_) > kSameHz) {
            if (!settling)
                state_ = State::Waiting;
            break;
        }
        if (!p25.present || !p25.control)
            break;
        // The grants come most recent first.
        const auto grant = std::find_if(p25.grants.begin(), p25.grants.end(),
                                        [&](const P25Grant& g) { return follows(g, p25, now); });
        if (grant == p25.grants.end())
            break;
        // The receiver's tuning error: the control channel was found off where
        // the system says it is by as much as the voice channel will be.
        const double announced = p25.site.control_hz;
        const double error     = announced > 0.0 && std::fabs(frequency - announced) < kControlNearHz
                                     ? frequency - announced
                                     : 0.0;
        call_       = *grant;
        target_hz_  = grant->frequency + error;
        tuned_at_   = now;
        call_heard_ = false;
        state_      = State::Voice;
        return {true, target_hz_, grant->slot >= 0 ? grant->slot % 2 : -1};
    }
    case State::Voice: {
        if (std::fabs(frequency - target_hz_) > kSameHz) {
            if (!settling)
                state_ = State::Waiting;
            break;
        }
        const bool active = call_.slot < 0 ? p25.present && p25.call.active
                                           : p25p2.present && p25p2.slot[call_.slot % 2].active;
        if (active) {
            call_heard_ = true;
            last_heard_ = now;
        }
        if (call_heard_ ? now - last_heard_ > kHangSeconds : now - tuned_at_ > kNoCallSeconds)
            return back_to_control(now, call_heard_);
        break;
    }
    }
    return {};
}

std::string TrunkFollower::describe() const
{
    char text[160];
    switch (state_) {
    case State::Off:
        return {};
    case State::Waiting:
        return "Waiting for a P25 control channel";
    case State::Control:
        std::snprintf(text, sizeof(text), "On the control channel, %.5f MHz: waiting for a call", control_hz_ / 1e6);
        return text;
    case State::Voice: {
        char slot[16] = "";
        if (call_.slot >= 0)
            std::snprintf(slot, sizeof(slot), ", slot %d", call_.slot % 2 + 1);
        std::snprintf(text, sizeof(text), "Following %s on %.5f MHz%s", p25_grant_destination(call_).c_str(),
                      target_hz_ / 1e6, slot);
        return text;
    }
    }
    return {};
}

}  // namespace spectro
