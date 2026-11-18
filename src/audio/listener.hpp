// Listening to one channel of the stream: a thread that demodulates FM -
// narrowband or wideband - and fills a FIFO with audio for the sound card.
//
//   receive thread ──tap──▶ IqRing ──▶ audio thread (FmDemodulator) ──▶ AudioFifo ──▶ sound card
//
// A narrowband channel that carries DMR or P25 is heard through its vocoder
// instead: the demodulator's frequency goes to a DmrDecoder, a P25Decoder and
// a P25Phase2Decoder, their voice frames to the AMBE+2 or IMBE decoder, and
// the speech takes the place of the FM audio (silence while there is none, or
// no vocoder).  The systems P25 control channels announce are remembered: a
// Phase 2 voice channel can only be read knowing its system.
//
// While nobody listens there is no tap, and none of this costs anything.
//
// The audio thread also keeps the audio in step with the sound card's clock
// (see AudioRateControl).
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

#include "audio/audio_fifo.hpp"
#include "audio/audio_scope.hpp"
#include "audio/scanner.hpp"
#include "dsp/dmr_decoder.hpp"
#include "dsp/fm_demod.hpp"
#include "dsp/p25_decoder.hpp"
#include "dsp/p25_phase2_decoder.hpp"
#include "dsp/signal_classifier.hpp"
#include "radio/acquisition.hpp"
#include "radio/iq_source.hpp"
#include "util/iq_ring.hpp"

namespace spectro {

struct ListenStatus {
    bool   enabled      = false;
    bool   in_band      = false;    // the channel lies inside the band being received
    bool   searching    = false;    // still looking for the carrier after tune(..., search)
    bool   squelch_open = false;
    double frequency    = 0.0;      // Hz; a search moves it onto the carrier
    double bandwidth    = 0.0;      // Hz, as used: limited to what the stream can carry
    FmMode mode         = FmMode::Narrow;
    bool   dmr_only     = false;    // set_dmr_only()
    bool   p25_only     = false;    // set_p25_only()
    float  level_dbfs   = -200.0f;  // power in the channel
    float  audio_rms    = 0.0f;     // of the demodulated audio, before the volume control
    float  load         = 0.0f;     // fraction of one core used by the audio thread
    double rate_trim    = 0.0;      // current correction of the audio rate
    uint64_t audio_samples = 0;     // audio samples produced so far
    uint64_t dropped_iq    = 0;     // samples the audio thread did not take over in time
    uint64_t underruns     = 0;     // times the sound card ran out of audio
    uint64_t overruns      = 0;     // times audio was discarded because the sound card fell behind
    SignalClass signal;             // what kind of signal the channel carries
    DmrStatus   dmr;                // DMR, when that is what it carries
    P25Status   p25;                // P25 Phase 1, likewise
    P25Phase2Status p25p2;          // P25 Phase 2, likewise
    int         p25p2_slot = -1;    // set_p25_phase2_slot()
    std::vector<p25p2::System> p25_systems;  // heard on control channels: what Phase 2 can be read for
    bool        vocoder = false;    // digital voice can be heard (built with the vocoders)
};

class Listener final : private IqTap, public ScanReceiver {
public:
    static constexpr int kAudioRate = 48000;

    // The channel widths on offer: narrowband FM up to 25 kHz, wideband FM
    // from 100 kHz.  The width chooses the kind of FM.  (Narrowband, the
    // demodulator goes a little wider, which a search for the carrier uses.)
    static constexpr double kMinNarrowBandwidth = FmDemodulator::kNarrowMinBandwidth;
    static constexpr double kMaxNarrowBandwidth = 25e3;
    static constexpr double kMinWideBandwidth   = 100e3;
    static constexpr double kMaxWideBandwidth   = FmDemodulator::kWideMaxBandwidth;
    static constexpr double kDefaultBandwidth   = FmDemodulator::kNarrowDefaultBandwidth;
    static bool valid_bandwidth(double hz)
    {
        return (hz >= kMinNarrowBandwidth && hz <= kMaxNarrowBandwidth) ||
               (hz >= kMinWideBandwidth && hz <= kMaxWideBandwidth);
    }

    Listener(IqSource& source, Acquisition& acquisition);
    ~Listener() override;

    Listener(const Listener&)            = delete;
    Listener& operator=(const Listener&) = delete;

    void start();
    // Detaches from the receive thread and joins the audio thread.
    void stop();

    // False if the stream is too narrow to hold an FM channel; nothing below
    // has any effect then.
    bool available() const { return available_; }
    // Whether the stream can hold a channel of this kind, and how wide one can be.
    bool   supports(FmMode mode) const { return FmDemodulator::supports(sample_rate_, kAudioRate, mode); }
    double widest(FmMode mode) const { return FmDemodulator::widest(sample_rate_, kAudioRate, mode); }
    // The width a channel actually gets when `hz` is asked for.
    double usable_bandwidth(double hz) const { return FmDemodulator::usable_bandwidth(sample_rate_, kAudioRate, hz); }

    // ---- Control: from any thread -------------------------------------------

    void set_enabled(bool enabled);
    // Selects the channel by its absolute frequency.  With `search_hz` > 0 the
    // strongest signal within that distance is located first, and the channel
    // centred on it: use it when the frequency is only known roughly.  With
    // `bandwidth_hz` > 0 the channel width changes in the same step.
    void tune(double frequency_hz, double search_hz = 0.0, double bandwidth_hz = 0.0);
    void set_bandwidth(double hz);           // also selects narrowband or wideband FM
    void set_deemphasis(double seconds);     // wideband FM: 75e-6 or 50e-6
    void set_squelch(float threshold_dbfs);  // FmDemodulator::kSquelchOff: always open
    void set_volume(float volume);           // 0 ... 1
    void set_muted(bool muted);              // silence without stopping (the audio keeps flowing)
    // Play DMR voice only: whatever else the channel carries - analog audio,
    // other digital modes - stays silent.  With P25 only as well, both play.
    void set_dmr_only(bool dmr_only);
    void set_p25_only(bool p25_only);  // Phase 1 and Phase 2
    // P25 Phase 2: which of the two calls on a channel to hear (0 or 1);
    // -1: the first to speak.
    void set_p25_phase2_slot(int slot);

    ListenStatus status() const;

    // Where the audio goes: mono, kAudioRate samples per second.
    AudioFifo& audio() { return fifo_; }
    // The last seconds of it as played, before the volume control: for a plot.
    const AudioScope& scope() const { return scope_; }

    // ScanReceiver: what a Scanner drives.
    void   scan_tune(double frequency_hz, double bandwidth_hz, double search_hz) override
    {
        tune(frequency_hz, search_hz, bandwidth_hz);
    }
    void   scan_mute(bool muted) override { set_muted(muted); }
    double scan_frequency() const override;
    ScanContent scan_content() const override;

private:
    static double clamp_bandwidth(double hz);
    void          forget_signal();

    struct Params {
        bool     enabled     = false;
        uint64_t session     = 0;  // counts the times listening was switched on
        double   frequency   = 0.0;
        double   search      = 0.0;
        uint64_t tune_serial = 0;  // counts the calls of tune()
        double   bandwidth   = kDefaultBandwidth;
        double   deemphasis  = FmDemodulator::kDefaultDeemphasis;
        float    squelch     = FmDemodulator::kSquelchOff;
        float    volume      = 0.5f;
        bool     muted       = false;
        bool     dmr_only    = false;
        bool     p25_only    = false;
        int      p25p2_slot  = -1;
    };

    void on_samples(const int16_t* iq, size_t samples) override;  // receive thread
    void run();
    template <typename Change> void change(Change&& change);

    IqSource&    source_;
    Acquisition& acquisition_;
    const double sample_rate_;
    const bool   available_;

    IqRing     ring_;  // samples on their way to the audio thread
    AudioFifo  fifo_;
    AudioScope scope_;

    mutable std::mutex    params_mutex_;
    Params                params_;
    std::atomic<uint64_t> generation_{0};
    std::atomic<bool>     enabled_{false};

    std::thread       thread_;
    std::atomic<bool> stop_{false};
    bool              stopped_ = true;

    // Status, written by the audio thread (dropped_iq_: by the receive thread).
    std::atomic<bool>     st_in_band_{false};
    std::atomic<uint64_t> st_settled_serial_{0};  // last tune() applied, its carrier search done
    std::atomic<bool>     st_squelch_open_{false};
    std::atomic<float>    st_level_{-200.0f};
    std::atomic<float>    st_audio_rms_{0.0f};
    std::atomic<float>    st_load_{0.0f};
    std::atomic<double>   st_trim_{0.0};
    std::atomic<uint64_t> st_audio_samples_{0};
    std::atomic<uint64_t> st_overruns_{0};
    std::atomic<uint64_t> dropped_iq_{0};
    mutable std::mutex    signal_mutex_;
    SignalClass           signal_;
    DmrStatus             dmr_;
    P25Status             p25_;
    P25Phase2Status       p25p2_;
    std::vector<p25p2::System> p25_systems_;
};

}  // namespace spectro
