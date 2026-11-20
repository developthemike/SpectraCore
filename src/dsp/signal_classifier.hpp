// Tells what kind of signal an FM channel carries: noise, a bare carrier,
// analog voice or audio, or digital modulation - and then which: AFSK, 2FSK,
// GMSK or 4FSK, its symbol rate and deviation, and where a sync pattern gives
// it away, the system (DMR, P25 Phase 1, POCSAG).
//
// It works on what the FM demodulator sees: the instantaneous frequency of the
// channel and its power, at the audio rate, half a second at a time.
//
//   - Noise: the envelope of an FM signal is constant, that of noise is not;
//     the two moments of the channel power give the signal-to-noise ratio.
//   - Digital or not: data is sent at a fixed symbol rate, which shows as a
//     sharp line in the spectrum of the squared derivative of the frequency
//     (the symbol transitions).  Speech has no such line.  A plain tone does -
//     at twice its frequency - so a line that is matched by a tone at half its
//     frequency does not count.
//   - Which modulation: the frequency at the symbol instants falls into two or
//     four levels.  Two levels with a modulation index around 0.5 are GMSK or
//     MSK, otherwise FSK.  AFSK shows instead as two audio tones, 1200 and
//     2200 Hz (Bell 202, as used for packet radio).
//   - Sync patterns: DMR, P25 Phase 1 (4FSK, 4800 Bd) and POCSAG paging
//     (2FSK) are recognised by the sync words that start their frames.
//   - Analog: voice or other audio between 300 Hz and 3 kHz, standing out from
//     the noise, and a sub-audible (CTCSS) tone if there is one.  A wideband
//     channel is broadcast FM, in stereo if its 19 kHz pilot is there.
//
// Not thread safe: one thread calls everything.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "dsp/fm_demod.hpp"

namespace spectro {

enum class SignalKind {
    Analysing,  // not enough seen yet
    Noise,      // nothing there
    Weak,       // something, but too weak to tell what
    Carrier,    // a carrier without audio (perhaps with a sub-audible tone)
    Analog,     // analog FM: voice or other audio
    Broadcast,  // wideband FM broadcast
    Afsk,       // audio tones keyed by data
    Fsk2,       // two-level FSK
    Gmsk,       // two-level FSK with a modulation index of about 0.5: GMSK, MSK
    Fsk4,       // four-level FSK
    Dqpsk,      // differential QPSK: four phase steps (not told by the statistics; P25 Phase 2 once decoded)
};

struct SignalClass {
    SignalKind  kind         = SignalKind::Analysing;
    float       snr_db       = 0.0f;   // in the channel
    double      symbol_rate  = 0.0;    // digital: symbols per second
    double      deviation_hz = 0.0;    // digital: of the outermost level
    double      ctcss_hz     = 0.0;    // analog: sub-audible tone, 0 = none
    bool        stereo       = false;  // broadcast: stereo pilot present
    std::string protocol;              // digital: named by its sync pattern, e.g. "DMR"

    bool digital() const
    {
        return kind == SignalKind::Afsk || kind == SignalKind::Fsk2 || kind == SignalKind::Gmsk ||
               kind == SignalKind::Fsk4 || kind == SignalKind::Dqpsk;
    }
};

// "Analog voice or audio, CTCSS 100.0 Hz", "Digital: 4FSK, 4800 Bd, ±1.94 kHz, DMR", ...
std::string describe(const SignalClass& signal);
// A few words for a label: "DMR", "4FSK", "voice", "stereo", ...
std::string short_name(const SignalClass& signal);
// Systems known to use this kind of modulation, when no sync pattern named one.
std::string typical_systems(const SignalClass& signal);

class SignalClassifier {
public:
    static constexpr double kWindowSeconds = 0.5;
    static constexpr float  kMinSnrDb      = 6.0f;   // below: noise
    static constexpr float  kMinAnalogSnrDb = 8.0f;  // below: too weak to call analog

    explicit SignalClassifier(double sample_rate);
    ~SignalClassifier();

    SignalClassifier(const SignalClassifier&)            = delete;
    SignalClassifier& operator=(const SignalClassifier&) = delete;

    // What kind of channel it is listening to.  A change starts afresh.
    void set_channel(FmMode mode, double bandwidth_hz);
    void reset();

    // Instantaneous frequency (Hz) and channel power, as FmDemodulator gives them.
    void process(const float* frequency, const float* power, size_t count);

    // The latest verdict - it changes only once two windows in a row agree.
    const SignalClass& result() const { return result_; }
    // Counts the windows analysed.
    uint64_t windows() const { return windows_; }

private:
    struct Line {
        double frequency  = 0.0;
        double power      = 0.0;  // Hz^2
        double prominence = 0.0;  // over the spectrum around it
        bool   peak       = false;  // a true peak, not the flank of a stronger line nearby
    };

    void        analyse();
    SignalClass classify();
    bool        classify_digital(double rate, SignalClass& out);
    void        spectrum(const std::vector<float>& x, std::vector<float>& out);
    void        filter(const std::vector<float>& taps, const std::vector<float>& in, std::vector<float>& out);
    void        transition_spectrum(const std::vector<float>& lowpassed, std::vector<float>& out);
    bool        afsk();
    Line        line(const std::vector<float>& spec, double frequency, double tolerance) const;
    double      band(const std::vector<float>& spec, double low, double high) const;
    bool        tone_at(double frequency) const;
    double      tonal_share() const;

    const double sample_rate_;
    FmMode       mode_      = FmMode::Narrow;
    double       bandwidth_ = 0.0;
    size_t       window_    = 0;  // samples per analysis

    std::vector<float> frequency_, power_;  // the window being filled
    std::vector<float> centred_, lowpassed_, filtered_, padded_, transitions_, smoothed_, symbols_;
    std::vector<float> freq_spec_, afsk_spec_;  // power spectra, Hz^2 per bin

    // Symbol transitions are looked for below a cutoff to suit the symbol
    // rate: the discriminator's noise rises steeply with frequency.
    struct RateBand {
        double             max_rate;  // symbol rates up to this
        std::vector<float> taps;      // the low-pass
        std::vector<float> spectrum;  // of the transitions, this window
        bool               ready = false;
    };
    std::vector<RateBand> bands_;
    std::vector<float>    lowpass_taps_;  // 6 kHz: the frequency lightly filtered
    std::vector<float> contrast_;      // AFSK: mark against space, per sample

    // FFTW
    float* fft_in_  = nullptr;
    void*  fft_out_ = nullptr;
    void*  plan_    = nullptr;
    std::vector<float> fft_window_;
    double             fft_scale_ = 1.0;

    SignalClass result_;
    SignalClass previous_;           // the last window's verdict
    std::string recent_protocol_;    // seen within the last few windows
    double      last_ctcss_ = 0.0;   // in the previous window
    int         protocol_age_ = 1000;
    uint64_t    windows_      = 0;
};

}  // namespace spectro
