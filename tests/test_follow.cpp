#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <thread>
#include <vector>

#include "app/pipeline.hpp"
#include "audio/heard_channels.hpp"
#include "audio/listener.hpp"
#include "audio/trunk_follower.hpp"
#include "radio/sim_source.hpp"
#include "test_framework.hpp"

using namespace spectro;

namespace {

constexpr double kControlHz = 851.1e6;
constexpr double kError     = 200.0;  // the receiver's tuning error

// A control channel as the listener reports it, granting `grants` just now.
P25Status control_channel(std::vector<P25Grant> grants, bool system_known = true)
{
    P25Status s;
    s.present         = true;
    s.control         = true;
    s.nac             = 0x293;
    s.site.wacn       = system_known ? 0xA1234 : -1;
    s.site.system     = 0x123;
    s.site.control_hz = kControlHz;
    s.time            = 10.0;
    for (P25Grant& g : grants)
        g.time = 9.9;
    s.grants = std::move(grants);
    return s;
}

P25Grant grant(uint32_t talkgroup, double frequency, int slot = -1, bool encrypted = false)
{
    P25Grant g;
    g.destination = talkgroup;
    g.source      = 1000 + talkgroup;
    g.frequency   = frequency;
    g.slot        = slot;
    g.channel     = static_cast<uint16_t>(slot >= 0 ? 0x2000 | (talkgroup & 0xFFF) : 0x1000 | (talkgroup & 0xFFF));
    g.encrypted   = encrypted;
    return g;
}

P25Status phase1_call(bool active)
{
    P25Status s;
    s.present          = true;
    s.call.active      = active;
    s.call.voice       = active;
    s.call.ids_known   = true;
    s.call.destination = 100;
    return s;
}

}  // namespace

TEST(trunk_follower_goes_to_each_call_and_back)
{
    const double   nan = std::numeric_limits<double>::quiet_NaN();
    TrunkFollower  f;
    const auto     here = kControlHz + kError;
    CHECK(f.state() == TrunkFollower::State::Off);
    CHECK(!f.update(here, control_channel({grant(100, 851.0e6)}), {}, 0.0).tune);  // off: nothing
    f.set_enabled(true);
    CHECK(f.state() == TrunkFollower::State::Waiting);

    // On the control channel: an encrypted call is left out, the other followed
    // - to where the system says, off by as much as the control channel is.
    FollowAction a = f.update(here, control_channel({grant(300, 851.2e6, -1, true), grant(100, 851.0e6)}), {}, 0.0);
    a              = a.tune ? a : f.update(here, control_channel({grant(300, 851.2e6, -1, true), grant(100, 851.0e6)}), {}, 0.1);
    CHECK(a.tune);
    CHECK_NEAR(a.frequency, 851.0e6 + kError, 1e-6);
    CHECK_EQ(a.slot, -1);
    CHECK(f.state() == TrunkFollower::State::Voice);
    CHECK(f.describe().find("Following TG 100 on 851.00020 MHz") == 0);

    // The call, then a second of quiet in case of a reply, then back.
    const double voice = a.frequency;
    CHECK(!f.update(nan, {}, {}, 0.2).tune);  // not settled yet
    CHECK(!f.update(voice, phase1_call(true), {}, 0.5).tune);
    CHECK(!f.update(voice, phase1_call(true), {}, 3.0).tune);  // calls take as long as they take
    CHECK(!f.update(voice, phase1_call(false), {}, 3.5).tune);
    CHECK(!f.update(voice, phase1_call(false), {}, 3.9).tune);
    a = f.update(voice, phase1_call(false), {}, 4.1);
    CHECK(a.tune);
    CHECK_NEAR(a.frequency, here, 1e-6);
    CHECK(f.state() == TrunkFollower::State::Control);

    // The call's grant still being repeated is not followed again at once...
    CHECK(!f.update(here, control_channel({grant(100, 851.0e6)}), {}, 4.5).tune);
    // ... nor is a grant that is no longer fresh.
    P25Status stale = control_channel({grant(200, 851.0125e6)});
    stale.grants[0].time = 8.0;
    CHECK(!f.update(here, stale, {}, 4.6).tune);

    // A Phase 2 call: to its slot.  Not found there: back after a while.
    a = f.update(here, control_channel({grant(400, 851.0375e6, 3)}), {}, 4.7);
    CHECK(a.tune);
    CHECK_EQ(a.slot, 1);
    CHECK(f.describe().find(", slot 2") != std::string::npos);
    P25Phase2Status other_slot;
    other_slot.present        = true;
    other_slot.slot[0].active = true;  // the other slot's call does not count
    CHECK(!f.update(a.frequency, {}, other_slot, 6.0).tune);
    a = f.update(a.frequency, {}, other_slot, 6.8);
    CHECK(a.tune);
    CHECK_NEAR(a.frequency, here, 1e-6);
    // Not tried again for a while.
    CHECK(!f.update(here, control_channel({grant(400, 851.0375e6, 3)}), {}, 7.5).tune);

    // Neither phase needs the system's IDs from here (Phase 2 works them out on its channel).
    const FollowAction p2 = f.update(here, control_channel({grant(500, 851.05e6, 0)}, false), {}, 7.6);
    CHECK(p2.tune);
    CHECK_EQ(p2.slot, 0);

    // Tuning elsewhere by hand: off the system until the next control channel.
    CHECK(!f.update(462.5625e6, {}, {}, 9.0).tune);
    CHECK(f.state() == TrunkFollower::State::Waiting);
    f.set_enabled(false);
    CHECK(f.state() == TrunkFollower::State::Off);
    CHECK(f.describe().empty());
}

TEST(heard_channels_list_p25_phase2_calls)
{
    HeardChannels   heard;
    P25Phase2Status p2;
    p2.present                = true;
    p2.keyed                  = true;
    p2.system                 = {0xA1234, 0x123, 0x293};
    p2.slot[1].active         = true;
    p2.slot[1].voice          = true;
    p2.slot[1].ids_known      = true;
    p2.slot[1].destination    = 200;
    p2.slot[1].source         = 3001;
    heard.update(851.0375e6, DmrStatus{}, P25Status{}, 1.0, p2);
    heard.update(851.0375e6, DmrStatus{}, P25Status{}, 1.5, p2);
    p2.slot[1].active = false;
    heard.update(851.0375e6, DmrStatus{}, P25Status{}, 2.0, p2);
    CHECK_EQ(heard.channels().size(), size_t{1});
    if (heard.channels().size() != 1)
        return;
    const HeardChannel& c = heard.channels()[0];
    CHECK(c.system == "P25 Phase 2");
    CHECK_EQ(c.calls, 1);
    CHECK_EQ(c.slot, 1);
    CHECK_EQ(c.nac, 0x293);
    CHECK(c.talkgroup == "TG 200");
}

TEST(following_the_simulators_trunked_system)
{
    // On the simulator's P25 control channel: it grants the P25 radio's call
    // four seconds in, and the Phase 2 channel's call (slot 2) eight seconds
    // in.  The follower goes to each and comes back; the Phase 2 call can be
    // read because the control channel has named the system.
    Options options;
    options.source.kind        = "sim";
    options.source.sample_rate = 2.4e6;
    options.source.frequency   = 150e6;
    options.source.gain        = SimSource::kReferenceGainDb;
    Pipeline pipeline(options);
    pipeline.start();
    Listener& listener = pipeline.listener();
    listener.set_volume(0.0f);
    listener.set_enabled(true);
    const std::vector<SimSource::DigitalStation> stations = SimSource::digital_stations(2.4e6);
    const double control_hz = 150e6 + stations[3].offset_hz;
    const double phase1_hz  = 150e6 + stations[2].offset_hz;
    const double phase2_hz  = 150e6 + stations[4].offset_hz;
    listener.tune(control_hz, 0.0, 12.5e3);

    TrunkFollower follower;
    follower.set_enabled(true);
    bool   on_phase1 = false, heard_phase1 = false, on_phase2 = false, named_phase2 = false;
    int    returns   = 0;
    uint64_t phase2_frames = 0;
    const auto start = std::chrono::steady_clock::now();
    while (returns < 2 && std::chrono::steady_clock::now() - start < std::chrono::milliseconds(14000)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const ListenStatus s   = listener.status();
        const double       now = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const FollowAction a   = follower.update(s.in_band && !s.searching ? s.frequency : std::numeric_limits<double>::quiet_NaN(),
                                                 s.p25, s.p25p2, now);
        if (follower.state() == TrunkFollower::State::Voice && std::fabs(s.frequency - phase1_hz) < 1.0) {
            on_phase1 = true;
            heard_phase1 |= s.p25.present && s.p25.call.active && s.p25.call.destination == SimSource::kP25Call.talkgroup;
        }
        if (follower.state() == TrunkFollower::State::Voice && std::fabs(s.frequency - phase2_hz) < 1.0) {
            on_phase2 = true;
            const P25Call& c = s.p25p2.slot[SimSource::kP25Phase2Call.slot];
            named_phase2 |= s.p25p2.keyed && c.active && c.ids_known && c.destination == SimSource::kP25Phase2Call.talkgroup &&
                            c.source == SimSource::kP25Phase2Call.source;
            phase2_frames = std::max(phase2_frames, s.p25p2.voice_frames);
        }
        if (!a.tune)
            continue;
        std::printf("    %.1f s: %s\n", now, follower.state() == TrunkFollower::State::Voice ? follower.describe().c_str()
                                                                                           : "back to the control channel");
        if (follower.state() == TrunkFollower::State::Control) {
            ++returns;
            CHECK_NEAR(a.frequency, control_hz, 1.0);
        }
        listener.tune(a.frequency, 0.0, 12.5e3);
        listener.set_p25_phase2_slot(a.slot);
    }
    const ListenStatus s = listener.status();
    std::printf("    Phase 2: %llu voice frames; systems known: %zu\n", static_cast<unsigned long long>(phase2_frames),
                s.p25_systems.size());
    CHECK(on_phase1);
    CHECK(heard_phase1);
    CHECK(on_phase2);
    CHECK(named_phase2);
    CHECK(phase2_frames > 50u);
    CHECK_EQ(returns, 2);
    CHECK_EQ(s.p25_systems.size(), size_t{1});
    if (s.p25_systems.size() == 1)
    CHECK(s.p25_systems[0] == (p25p2::System{SimSource::kP25Wacn, SimSource::kP25System, SimSource::kP25Nac}));

    listener.set_enabled(false);
    pipeline.stop();
}
