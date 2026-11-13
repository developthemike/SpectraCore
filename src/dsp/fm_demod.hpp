// FM receiver for one channel of the IQ stream: narrowband (two-way radio
// voice) or wideband (broadcast).
//
//   IQ samples ──▶ [1] shift the channel to 0 Hz and decimate
//              ──▶ [2] channel filter, resample to the discriminator rate
//              ──▶ [3] FM discriminator
//              ──▶ [4] de-emphasis, audio filters, squelch ──▶ audio
//
//                     narrowband: 5 to 36 kHz          wideband: 50 to 300 kHz
//   after stage 1     at least 100 kS/s                at least 1.2 MS/s
//   discriminator     at the audio rate                at 8 x the audio rate
//   full scale        deviation of 0.4 x bandwidth     75 kHz deviation
//   de-emphasis       750 us, 0 dB at 1 kHz            75 or 50 us, 0 dB at low frequencies
//   audio band        300 Hz to 3.4 kHz (voice)        30 Hz to 15 kHz (mono, L + R)
//
// Stage 1 is the only one that runs at the full sample rate.  It is a single
// FIR filter with complex taps - the frequency shift is folded into them - so
// it costs four multiply-adds per tap and output sample and nothing else.  Its
// window is that of a fourth-order CIC decimator, which puts deep nulls
// exactly on the frequencies that would alias into the channel.  Its cost
// does not depend on how far it decimates: the filter grows as the output
// rate falls.
//
// Stage 2 is a polyphase resampler whose prototype is the channel filter.  Its
// output rate can be trimmed by a fraction of a percent, which is how the
// audio is kept in step with the sound card's clock.
//
// Not thread safe: one thread calls everything.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace spectro {

enum class FmMode {
    Narrow,  // two-way radio voice: 12.5 / 25 kHz channels
    Wide,    // broadcast: +/-75 kHz deviation
};

class FmDemodulator {
public:
    // Below this there is no room for an FM channel in the stream.
    static constexpr double kMinSampleRate = 16e3;

    // Channel bandwidth: the width of the channel filter between its -6 dB
    // points.  The mode follows from it: kWideMinBandwidth and up is wideband.
    static constexpr double kNarrowMinBandwidth     = 5e3;
    static constexpr double kNarrowMaxBandwidth     = 36e3;
    static constexpr double kNarrowDefaultBandwidth = 12.5e3;
    static constexpr double kWideMinBandwidth       = 50e3;
    static constexpr double kWideMaxBandwidth       = 300e3;
    static constexpr double kWideDefaultBandwidth   = 200e3;

    // Broadcast FM: the deviation that is full scale, and the de-emphasis time
    // constant (75 us in the Americas and South Korea, 50 us elsewhere).
    static constexpr double kWideDeviation     = 75e3;
    static constexpr double kDefaultDeemphasis = 75e-6;

    // Squelch threshold that never closes the squelch.
    static constexpr float kSquelchOff = -200.0f;

    static FmMode mode_for(double bandwidth_hz)
    {
        return bandwidth_hz >= kWideMinBandwidth ? FmMode::Wide : FmMode::Narrow;
    }
    // The widest channel of a mode that a stream of `sample_rate` can carry,
    // and whether it can carry that mode at all.
    static double widest(double sample_rate, double audio_rate, FmMode mode);
    static bool   supports(double sample_rate, double audio_rate, FmMode mode);
    // The bandwidth actually used when `bandwidth_hz` is asked for: clamped to
    // the range of its mode and to what the stream can carry.
    static double usable_bandwidth(double sample_rate, double audio_rate, double bandwidth_hz);

    // Throws std::invalid_argument for rates it cannot work with.
    FmDemodulator(double sample_rate, double audio_rate);

    // Selects the channel: its centre relative to the centre of the IQ stream,
    // its bandwidth (both in Hz) and the kind of FM.  The bandwidth is limited
    // to the range of the mode and to what the stream can carry.  Changing the
    // mode starts the demodulator afresh.
    void   set_channel(double offset_hz, double bandwidth_hz, FmMode mode);
    void   set_channel(double offset_hz, double bandwidth_hz) { set_channel(offset_hz, bandwidth_hz, mode_for(bandwidth_hz)); }
    double offset() const { return offset_hz_; }
    double bandwidth() const { return bandwidth_hz_; }
    FmMode mode() const { return mode_; }

    // Wideband only: the de-emphasis time constant in seconds.
    void set_deemphasis(double seconds);

    // The audio is muted while the power in the channel is below this level
    // (dBFS, as on the spectrum display).
    void set_squelch(float threshold_dbfs) { squelch_dbfs_ = threshold_dbfs; }

    // Produces audio at audio_rate * (1 + trim) samples per second of signal.
    void set_rate_trim(double trim);

    // Forgets the signal seen so far.  Use after a break in the stream.
    void reset();

    // Demodulates `samples` IQ pairs (SC16 Q11) and appends the audio to `audio`:
    // mono, at the audio rate, +/-1.0 at full-scale deviation (see above).
    void process(const int16_t* iq, size_t samples, std::vector<float>& audio);

    double sample_rate() const { return sample_rate_; }
    double audio_rate() const { return audio_rate_; }
    double discriminator_rate() const { return audio_rate_ * disc_factor_; }

    struct Stats {
        float level_dbfs   = -200.0f;  // power in the channel
        bool  squelch_open = false;
        float audio_rms    = 0.0f;     // of the audio delivered
    };
    Stats stats() const;

    // Where the signal sits relative to the channel centre: the mean of the
    // instantaneous frequency since the previous call, in Hz.  For an FM signal
    // that is its carrier.  `samples` counts discriminator samples.
    struct FrequencyEstimate {
        double mean_hz = 0.0;
        size_t samples = 0;
    };
    FrequencyEstimate take_frequency_estimate();

    // For tests and diagnostics.
    int    decimation() const { return decimation_; }
    int    channel_taps() const { return channel_taps_; }
    double channel_rate() const { return sample_rate_ / decimation_; }

private:
    struct Biquad {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
        double z1 = 0.0, z2 = 0.0;
        float  run(float x);
        void   clear() { z1 = z2 = 0.0; }
    };

    void configure();  // everything that depends on the mode
    void design_shift();
    void design_channel_filter();
    void design_deemphasis();
    void update_rates();
    void resample_and_demodulate(std::vector<float>& audio);
    void discriminate(float re, float im, std::vector<float>& audio);
    void emit(float sample, std::vector<float>& audio);
    void update_squelch(float mean_power);

    const double sample_rate_;
    const double audio_rate_;
    FmMode       mode_         = FmMode::Narrow;
    double       offset_hz_    = 0.0;
    double       bandwidth_hz_ = kNarrowDefaultBandwidth;
    double       rate_trim_    = 0.0;
    double       deemphasis_s_ = kDefaultDeemphasis;
    float        squelch_dbfs_ = kSquelchOff;

    // --- Stage 1: shift and decimate ---
    int                decimation_ = 1;
    size_t             shift_len_  = 1;   // taps
    std::vector<float> window_;           // real window, shift_len_ values summing to 1
    std::vector<float> shift_re_;         // taps arranged for interleaved I/Q input:
    std::vector<float> shift_im_;         //   out.re = in . shift_re_, out.im = in . shift_im_
    std::vector<float> in_;               // input waiting to be filtered, interleaved I/Q
    size_t             in_count_ = 0;     // IQ pairs in in_
    double             phase_      = 0.0; // rotation of the next output sample
    double             phase_step_ = 0.0;

    // --- Stage 2: channel filter and resampler ---
    int                channel_taps_ = 0;  // per phase
    std::vector<float> phase_taps_;        // (kPhases + 1) rows of channel_taps_
    std::vector<float> ch_re_, ch_im_;     // stage 1 output waiting to be resampled
    size_t             ch_count_ = 0;
    double             time_ = 0.0;        // position of the next output sample in ch_re_/ch_im_
    double             step_ = 1.0;        // input samples per output sample

    // --- Stage 3: discriminator ---
    int    disc_factor_ = 1;  // discriminator rate / audio rate
    float  prev_re_ = 0.0f, prev_im_ = 0.0f;
    float  hz_per_radian_ = 0.0f;
    float  audio_per_hz_  = 0.0f;
    double freq_sum_   = 0.0;
    size_t freq_count_ = 0;

    // --- Stage 4: audio ---
    float  deemph_coeff_ = 1.0f, deemph_gain_ = 1.0f, deemph_state_ = 0.0f;
    Biquad highpass_[2];             // narrowband: voice band
    Biquad lowpass_[2];
    std::vector<float> audio_taps_;  // wideband: 15 kHz low-pass, decimating to the audio rate
    std::vector<float> wide_buf_;    // its input
    size_t             wide_count_ = 0;
    int                wide_phase_ = 0;
    Biquad             dc_block_;    // wideband: 30 Hz high-pass

    // --- Squelch and statistics ---
    int    level_block_  = 240;  // discriminator samples per level measurement (5 ms)
    float  block_power_  = 0.0f;
    int    block_count_  = 0;
    float  level_power_  = 0.0f;  // smoothed, for the display
    bool   have_level_   = false;
    bool   squelch_open_ = false;
    int    squelch_hold_ = 0;     // level blocks left before a fading signal is muted
    float  gate_         = 0.0f;  // current squelch gain, 0..1
    float  audio_power_  = 0.0f;
};

}  // namespace spectro
