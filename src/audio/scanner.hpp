// Scanning: finds signals in the spectrogram, tunes the listener to one, stays
// with it for as long as it is on the air, then moves on to the next.
//
// The scanner reads the same spectrum rows as the display, so it costs next
// to nothing and agrees with what is on the screen:
//
//   - A signal is a stretch of the spectrum standing clear of the noise floor.
//     It is a candidate when its peak reaches the threshold - a level in dBFS
//     on the same scale as the plot - for long enough not to be a blip.
//   - Its width tells wideband (broadcast) FM from narrowband FM: broadcast
//     stations occupy well over 60 kHz, two-way radio channels 25 kHz at most.
//     Stretches far wider than any FM channel (data bursts, Wi-Fi) are ignored.
//   - While the signal stays above the threshold the scanner stays.  When it
//     drops below, the scanner holds on for a while (the hang time) in case it
//     comes back, as two-way radio conversations pause between overs.  Then it
//     goes on to the next signal above it in frequency, wrapping around at the
//     top of the band.
//   - The audio is muted while there is nothing to hear: while searching and
//     during the hang time.
//
// Not thread safe: one thread (the user interface's) calls everything.
#pragma once

#include <cstddef>
#include <vector>

#include "dsp/spectrum_engine.hpp"

namespace spectro {

// How the columns of a spectrum row map to frequency.
struct SpectrumView {
    double centre_hz   = 0.0;
    double sample_rate = 1.0;
    int    columns     = 1;
    int    fft_size    = 1;

    double column_hz() const { return sample_rate / columns; }
    double frequency(double column) const
    {
        return centre_hz + SpectrumEngine::column_frequency(column, columns, fft_size, sample_rate);
    }
};

// A signal found in the spectrum.
struct DetectedSignal {
    double frequency_hz = 0.0;      // centre of its power
    double occupied_hz  = 0.0;      // width between the points 26 dB below its peak
    double low_hz       = 0.0;      // the stretch standing clear of the noise
    double high_hz      = 0.0;
    float  peak_dbfs    = -200.0f;  // strongest column
    bool   wide         = false;    // looks like wideband (broadcast) FM
};

// Signals far wider than an FM channel are not FM.
inline constexpr double kWideFmMinOccupied = 60e3;
inline constexpr double kFmMaxOccupied     = 350e3;

// The noise floor of a spectrum given as linear power per column: its median.
float spectrum_noise_floor(const float* power, int columns);

// Finds the signals in a spectrum given as linear power per column (relative
// to a full-scale sine, like the dBFS of the display), strongest first.
std::vector<DetectedSignal> find_signals(const float* power, const SpectrumView& view, float floor_power);

// The channel width to listen to a signal with.
double listen_bandwidth_for(const DetectedSignal& signal, double column_hz);

// What a Scanner drives (the Listener, or a stand-in in tests).
class ScanReceiver {
public:
    virtual ~ScanReceiver() = default;
    // Listen at `frequency_hz` in a channel `bandwidth_hz` wide, after looking
    // for the carrier within `search_hz` of it.
    virtual void scan_tune(double frequency_hz, double bandwidth_hz, double search_hz) = 0;
    virtual void scan_mute(bool muted) = 0;
    // Where the receiver settled on the carrier; NaN while it is still looking.
    virtual double scan_frequency() const = 0;
};

class Scanner {
public:
    enum class State {
        Off,
        Searching,  // looking for a signal above the threshold
        Receiving,  // listening to one
        Holding,    // it has dropped below the threshold: waiting for it to come back
    };

    static constexpr float  kDefaultThresholdAboveFloor = 15.0f;  // dB, when no threshold was given
    static constexpr double kDefaultHang                = 2.0;    // seconds
    static constexpr float  kHysteresisDb               = 3.0f;   // a signal ends this far below the threshold
    static constexpr double kSmoothingSeconds           = 0.1;    // of the spectrum
    static constexpr double kConfirmSeconds             = 0.06;   // a candidate must last this long

    explicit Scanner(ScanReceiver& receiver);

    // Starts searching; the receiver is muted until something is found.
    void start();
    // Stops; the receiver stays where it is, unmuted.
    void stop();
    bool active() const { return state_ != State::Off; }

    void  set_threshold(float dbfs) { threshold_ = dbfs; }
    float threshold() const { return threshold_; }
    void   set_hang(double seconds);
    double hang() const { return hang_; }

    // Moves on from the signal being received, and leaves it out from now on
    // (until clear_lockouts()).
    void   skip();
    void   clear_lockouts() { lockouts_.clear(); }
    size_t lockouts() const { return lockouts_.size(); }

    // Feeds one spectrogram row (dBFS per column).  A gap row - the radio was
    // retuned - starts the search afresh.
    void on_row(const float* row_dbfs, const RowInfo& info, const SpectrumView& view);

    struct Status {
        State  state        = State::Off;
        double frequency    = 0.0;      // of the signal received (Receiving, Holding)
        double bandwidth    = 0.0;      // channel width chosen for it
        double occupied_hz  = 0.0;      // its measured width
        bool   wide         = false;    // wideband FM
        float  level_dbfs   = -200.0f;  // its current peak level
        double hold_left    = 0.0;      // seconds of hang time left (Holding)
        float  floor_dbfs   = -200.0f;  // noise floor of the spectrum
        bool   warming_up   = false;    // too soon after a (re)start to tell
    };
    Status status() const { return status_; }

    // The signals above the threshold in the latest row (not locked out).
    const std::vector<DetectedSignal>& candidates() const { return candidates_; }

private:
    struct Range {
        double low, high;
    };

    void restart_spectrum();
    void lock(const DetectedSignal& signal, const SpectrumView& view);
    void release();
    float channel_peak(const SpectrumView& view) const;
    bool  locked_out(double frequency) const;

    ScanReceiver& receiver_;
    State         state_     = State::Off;
    float         threshold_ = -60.0f;
    double        hang_      = kDefaultHang;

    // Smoothed spectrum, in dB per column like the display: a signal that stops
    // falls below the threshold just as fast whatever its strength.  And the
    // same in linear power, for find_signals().
    std::vector<float> smoothed_;
    std::vector<float> power_;
    int                fft_size_   = 0;
    double             warmup_left_ = 0.0;
    float              floor_power_ = 0.0f;

    std::vector<DetectedSignal> candidates_;
    std::vector<Range>          lockouts_;

    // Searching: the candidate waiting to be confirmed, and where to go on from.
    bool   pending_       = false;
    double pending_hz_    = 0.0;
    double pending_for_   = 0.0;  // seconds
    double resume_above_  = -1e300;

    // Receiving / Holding
    double locked_hz_    = 0.0;
    double locked_bw_    = 0.0;

    Status status_;
};

}  // namespace spectro
