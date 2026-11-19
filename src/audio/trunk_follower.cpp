#include "audio/trunk_follower.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace spectro {

namespace {

constexpr double kSameHz       = 1e3;  // the listener is where it was sent
constexpr double kControlNearHz = 5e3; // the control channel as announced (or listed), against where it was found

}  // namespace

void TrunkFollower::set_enabled(bool enabled)
{
    if (enabled == this->enabled())
        return;
    state_   = enabled ? State::Waiting : State::Off;
    hunting_ = false;
    listed_  = -1;
    recent_.clear();
}

void TrunkFollower::set_control_channels(std::vector<double> hz)
{
    channels_ = std::move(hz);
    listed_   = -1;
    if (channels_.empty())
        stop_hunting();
}

int TrunkFollower::list_index(double hz) const
{
    for (size_t i = 0; i < channels_.size(); ++i) {
        if (std::fabs(channels_[i] - hz) < kControlNearHz)
            return static_cast<int>(i);
    }
    return -1;
}

FollowAction TrunkFollower::hunt(double frequency, double now)
{
    if (channels_.empty() || hunting_)
        return {};
    set_enabled(true);
    hunting_ = true;
    // On a system already - and still there, which after a scan for signals,
    // say, it need not be: the hunt starts once its control channel is lost.
    if ((state_ == State::Control && std::fabs(frequency - control_hz_) <= kSameHz) ||
        (state_ == State::Voice && std::fabs(frequency - target_hz_) <= kSameHz))
        return {};
    listed_ = -1;  // from the top of the list
    return try_next(now);
}

void TrunkFollower::stop_hunting()
{
    hunting_ = false;
    if (state_ == State::Hunting)
        state_ = State::Waiting;  // staying there: a control channel heard there is still followed
}

FollowAction TrunkFollower::skip(double now)
{
    return hunting_ ? try_next(now) : FollowAction{};
}

FollowAction TrunkFollower::try_next(double now)
{
    listed_     = (listed_ + 1) % static_cast<int>(channels_.size());
    trying_hz_  = channels_[static_cast<size_t>(listed_)];
    tuned_at_   = now;
    arrived_at_ = std::numeric_limits<double>::quiet_NaN();
    state_      = State::Hunting;
    return {true, trying_hz_, -1};
}

void TrunkFollower::found_control(double frequency, double now)
{
    state_        = State::Control;
    control_hz_   = frequency;
    control_seen_ = now;
}

// The listener has been tuned elsewhere by hand: off the system, and no more hunting.
void TrunkFollower::tuned_away()
{
    state_   = State::Waiting;
    hunting_ = false;
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
    state_        = State::Control;
    tuned_at_     = now;
    control_seen_ = now;
    return {true, control_hz_, -1};
}

FollowAction TrunkFollower::update(double frequency, const P25Status& p25, const P25Phase2Status& p25p2, double now)
{
    if (state_ == State::Off)
        return {};
    recent_.erase(std::remove_if(recent_.begin(), recent_.end(), [&](const Recent& r) { return r.until <= now; }),
                  recent_.end());
    // A retune of ours takes a moment to show; any other is the user's.
    const bool settling  = now - tuned_at_ < kSettleSeconds;
    const bool listening = !std::isnan(frequency);
    const bool control   = listening && p25.present && p25.control;

    switch (state_) {
    case State::Off:
        break;
    case State::Waiting:
        if (control) {
            found_control(frequency, now);
            listed_ = list_index(frequency);
        }
        break;
    case State::Hunting:
        if (!listening || std::fabs(frequency - trying_hz_) > kSameHz) {
            if (listening && !settling)
                tuned_away();
            else if (now - tuned_at_ > kSettleSeconds + kTrySeconds)
                return try_next(now);  // never got there (outside the band received?)
            break;
        }
        if (std::isnan(arrived_at_))
            arrived_at_ = now;
        if (control)
            found_control(frequency, now);
        else if (now - arrived_at_ > kTrySeconds)
            return try_next(now);
        break;
    case State::Control: {
        if (!listening) {
            control_seen_ = now;  // it can only be lost while it is listened to
            break;
        }
        if (std::fabs(frequency - control_hz_) > kSameHz) {
            if (!settling)
                tuned_away();
            break;
        }
        if (!control) {
            if (hunting_ && !settling && now - control_seen_ > kLostSeconds)
                return try_next(now);  // gone quiet: look for it, or another, elsewhere
            break;
        }
        control_seen_ = now;
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
        if (!listening)
            break;
        if (std::fabs(frequency - target_hz_) > kSameHz) {
            if (!settling)
                tuned_away();
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
    case State::Hunting:
        std::snprintf(text, sizeof(text), "Looking for a control channel: %.5f MHz (%d of %zu)", trying_hz_ / 1e6,
                      listed_ + 1, channels_.size());
        return text;
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
