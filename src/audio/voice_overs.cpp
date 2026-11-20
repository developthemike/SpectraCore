#include "audio/voice_overs.hpp"

#include <chrono>
#include <cmath>
#include <utility>

namespace spectro {

namespace {

constexpr double kSameChannelHz = 1e3;

const char* system_of(VoiceOvers::Stream stream)
{
    switch (stream) {
    case VoiceOvers::DmrSlot1:
    case VoiceOvers::DmrSlot2:
        return "DMR";
    case VoiceOvers::P25Phase1:
        return "P25";
    default:
        return "P25 Phase 2";
    }
}

int slot_of(VoiceOvers::Stream stream)
{
    switch (stream) {
    case VoiceOvers::DmrSlot1:
    case VoiceOvers::P25Phase2Slot1:
        return 0;
    case VoiceOvers::DmrSlot2:
    case VoiceOvers::P25Phase2Slot2:
        return 1;
    default:
        return -1;
    }
}

double wall_clock_now()
{
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

}  // namespace

double VoiceOver::seconds() const
{
    return static_cast<double>(speech.size()) / VoiceOvers::kSampleRate;
}

void VoiceOvers::set_sink(VoiceOverSink* sink)
{
    if (sink == sink_)
        return;
    for (State& s : streams_) {
        s.open = false;
        s.over = VoiceOver{};
    }
    sink_ = sink;
}

void VoiceOvers::set_channel(double frequency)
{
    if (std::fabs(frequency - frequency_) > kSameChannelHz) {
        flush();
        for (State& s : streams_) {
            s.destination.clear();
            s.source = 0;
        }
    }
    frequency_ = frequency;  // (within a channel, the latest: a search may have moved it)
}

void VoiceOvers::identify(Stream stream, const std::string& destination, uint32_t source)
{
    if (!sink_)
        return;
    State& s = streams_[stream];
    // Another radio on the same call is another over.  (A caller that only
    // becomes known during the over is the same one.)
    if (s.open && source != 0 && s.over.source != 0 && source != s.over.source)
        end(stream);
    if (!destination.empty())
        s.destination = destination;
    if (source != 0)
        s.source = source;
    if (s.open) {
        if (!destination.empty())
            s.over.destination = destination;
        if (source != 0)
            s.over.source = source;
    }
}

void VoiceOvers::push(Stream stream, const float* speech, size_t count, bool first)
{
    if (!sink_)
        return;
    State& s = streams_[stream];
    if (first && s.open)
        end(stream);
    if (!s.open)
        begin(stream, 0, wall_clock_now());
    s.last = now_;

    const auto most = static_cast<size_t>(kMaxSeconds * kSampleRate);
    while (count > 0) {
        const size_t room = most - s.over.speech.size();
        const size_t take = count < room ? count : room;
        s.over.speech.insert(s.over.speech.end(), speech, speech + take);
        speech += take;
        count -= take;
        if (s.over.speech.size() >= most) {
            // A long over: hand over what there is, and go on in another part,
            // by the same radio.
            const int         part        = s.over.part + 1;
            const double      start       = s.over.start + s.over.seconds();
            const std::string destination = s.over.destination;
            const uint32_t    source      = s.over.source;
            end(stream);
            s.destination = destination;
            s.source      = source;
            begin(stream, part, start);
        }
    }
}

void VoiceOvers::advance(double now)
{
    now_ = now;
    for (int k = 0; k < kStreams; ++k) {
        if (streams_[k].open && now - streams_[k].last > kGapSeconds)
            end(static_cast<Stream>(k));
    }
}

void VoiceOvers::flush()
{
    for (int k = 0; k < kStreams; ++k) {
        if (streams_[k].open)
            end(static_cast<Stream>(k));
    }
}

void VoiceOvers::begin(Stream stream, int part, double start)
{
    State& s = streams_[stream];
    s.open = true;
    s.last = now_;
    VoiceOver& o = s.over;
    o = VoiceOver{};
    o.start       = start;
    o.frequency   = frequency_;
    o.system      = system_of(stream);
    o.slot        = slot_of(stream);
    o.destination = s.destination;
    o.source      = s.source;
    o.part        = part;
    o.speech.reserve(static_cast<size_t>(kMaxSeconds * kSampleRate));
}

void VoiceOvers::end(Stream stream)
{
    State& s = streams_[stream];
    s.open = false;
    // The next over may be another radio's: its decoder will say.
    s.destination.clear();
    s.source = 0;
    // A part after the first counts however short: it ends what was said.
    if (sink_ && (s.over.seconds() >= kMinSeconds || (s.over.part > 0 && !s.over.speech.empty())))
        sink_->voice_over(std::move(s.over));
    s.over = VoiceOver{};
}

}  // namespace spectro
