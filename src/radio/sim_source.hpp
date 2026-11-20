// Synthetic IQ source for running without hardware.
//
// Produces a lively but fully deterministic RF scene at the configured sample
// rate, paced to real time: a noise floor, steady and fading carriers, an FM
// signal, a sweeping chirp, wide-band bursts, a frequency hopper, a
// narrowband FM voice channel that plays a tune, a wideband FM broadcast
// station that plays another, and two digital stations: 4FSK with DMR's sync
// patterns, and POCSAG-style paging.  The signals live at fixed
// "RF" frequencies around the initial centre frequency, so tuning and gain
// behave as they would on a real receiver.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "dsp/p25_coding.hpp"
#include "radio/iq_source.hpp"

namespace spectro {

class AmbeEncoder;
class DmrTransmitter;
class ImbeEncoder;
class P25Transmitter;
class P25Phase2Transmitter;

class SimSource final : public IqSource {
public:
    explicit SimSource(const SourceConfig& config);
    ~SimSource() override;

    void start() override;
    void stop() override;
    int  read(int16_t* iq, int max_samples, ReadStatus& status) override;

    const SourceInfo& info() const override { return info_; }
    double sample_rate() const override { return sample_rate_; }
    double frequency() const override { return frequency_.load(std::memory_order_relaxed); }
    double bandwidth() const override { return sample_rate_; }
    double gain() const override { return gain_.load(std::memory_order_relaxed); }

    bool set_frequency(double hz) override;
    bool set_gain(double db) override;

    // Positions and levels of the steady test carriers, for tests and docs.
    struct Carrier {
        double offset_hz;   // from the initial centre frequency
        double level_dbfs;  // at the reference gain
    };
    static std::vector<Carrier> carriers(double sample_rate);
    static constexpr double kReferenceGainDb = 30.0;

    // The narrowband FM voice channel: something to listen to.  It carries a
    // tune, pre-emphasised like speech on a two-way radio, so that a receiver
    // with the matching de-emphasis plays every note equally loud.
    struct VoiceChannel {
        double offset_hz;     // from the initial centre frequency
        double level_dbfs;    // at the reference gain
        double deviation_hz;  // peak deviation a 1 kHz tone would have
    };
    static VoiceChannel voice_channel(double sample_rate);

    // The wideband FM broadcast station: an arpeggio with the 19 kHz stereo
    // pilot, pre-emphasised with 75 us like the broadcasts of the Americas.
    // It needs a stream of at least kBroadcastMinRate to fit in.
    struct BroadcastStation {
        double offset_hz;        // from the initial centre frequency
        double level_dbfs;       // at the reference gain
        double deviation_hz;     // peak deviation of the arpeggio, before pre-emphasis
        double pilot_hz;         // deviation of the pilot
        double deemphasis_s;     // the matching de-emphasis
    };
    static BroadcastStation broadcast_station(double sample_rate);
    static constexpr double kBroadcastMinRate = 2e6;

    // The digital stations: a DMR repeater, idle but for a voice call now and
    // then on each time slot (see kDmrCalls), a pager sending random data, a
    // P25 radio on the air only for its calls (see kP25Call), the control
    // channel of the trunked system it belongs to, which grants them, and a
    // Phase 2 voice channel of that system (see kP25Phase2Call), granted too.
    struct DigitalStation {
        const char* name;          // "DMR", "POCSAG"
        double      offset_hz;     // from the initial centre frequency
        double      level_dbfs;    // at the reference gain
        double      symbol_rate;
        double      deviation_hz;  // of the outermost level
        int         levels;        // 2 or 4
    };
    static std::vector<DigitalStation> digital_stations(double sample_rate);

    // The DMR repeater's calls: every kDmrCycleSeconds, on each slot, a group
    // call of `seconds` that starts `start` seconds into the cycle.
    static constexpr uint8_t kDmrColorCode   = 1;
    static constexpr double  kDmrCycleSeconds = 12.0;
    struct DmrCall {
        int      slot;         // 0 = time slot 1
        uint32_t talkgroup;
        uint32_t source;
        double   start;
        double   seconds;
    };
    static constexpr DmrCall kDmrCalls[2] = {{0, 1, 1001, 1.0, 3.6}, {1, 2, 1002, 7.0, 2.2}};
    // The P25 radio's call, every kDmrCycleSeconds too.
    static constexpr uint16_t kP25Nac  = 0x293;
    static constexpr DmrCall  kP25Call = {0, 100, 2001, 4.0, 2.7};
    // Its control channel, kP25ControlAboveHz above it.
    static constexpr double   kP25ControlAboveHz = 37.5e3;
    static constexpr uint32_t kP25Wacn   = 0xA1234;
    static constexpr uint16_t kP25System = 0x123;
    // The Phase 2 voice channel, kP25Phase2BelowHz below the P25 radio: a
    // call on its second slot every kDmrCycleSeconds.
    static constexpr double  kP25Phase2BelowHz = 37.5e3;
    static constexpr DmrCall kP25Phase2Call    = {1, 200, 3001, 8.0, 2.2};

private:
    struct Tone {
        double offset;       // fraction of the sample rate, relative to the initial centre
        double level_dbfs;
        double fade_hz;      // amplitude fading rate, 0 = steady
        double fade_depth;   // 0..1
        double phase = 0.0;  // radians
    };

    void build_tables();
    void generate(int n);
    void add_noise(int n);
    void add_tone(Tone& tone, int n, double shift);
    void add_fm(int n, double shift);
    void add_voice(int n, double shift);
    void add_broadcast(int n, double shift);

    // A digital station's transmitter: shaped FSK of random symbols, with a
    // sync pattern at a fixed place in every frame - or symbols from `source`.
    struct Fsk {
        double              center;      // fraction of the sample rate
        double              level_dbfs;
        double              rate;        // symbols per second
        std::vector<double> levels_hz;   // frequency of each symbol value
        std::vector<int>    sync;        // symbol values
        size_t              frame;       // symbols per frame
        size_t              sync_at;     // where in the frame the sync pattern starts
        std::function<int()> source = nullptr;  // the next symbol's index into levels_hz; -1: off the air
        // State
        double   phase   = 0.0;
        double   in_symbol = 0.0;  // 0..1 through the current symbol
        int      prev = 0, curr = 0, next = 0;
        size_t   index = 0;        // in the frame, of `next`
        uint32_t random = 1;
    };
    void add_fsk(Fsk& fsk, int n, double shift);
    static int next_fsk_symbol(Fsk& fsk);
    int  next_dmr_symbol();
    int  next_p25_symbol();
    int  next_p25_control_symbol();
    int  next_p25_phase2_symbol();
    p25::Tsbk next_p25_block();
    void add_chirp(int n, double shift);
    void add_bursts(int n, double shift);
    void add_hopper(int n, double shift);
    static void add_table_signal(const std::vector<float>& table, size_t& table_pos, double& phase,
                                 double freq, float amplitude, const float* envelope, float* mix, int n);
    uint32_t next_random();
    double   random_uniform();

    SourceInfo info_;
    double     sample_rate_;
    double     initial_frequency_;
    unsigned   seed_;

    std::atomic<double> frequency_;
    std::atomic<double> gain_;
    std::atomic<bool>   running_{false};

    // Pacing
    std::chrono::steady_clock::time_point start_time_;
    uint64_t                              produced_ = 0;
    double                                time_s_   = 0.0;  // simulated time

    // Precomputed material
    std::vector<float> noise_;        // white Gaussian noise, unit variance per component
    std::vector<float> wide_table_;   // band-limited noise for the wide-band bursts
    std::vector<float> narrow_table_; // band-limited noise for the hopper

    // Signal state
    std::vector<Tone>  tones_;
    double             fm_phase_ = 0.0;
    double             voice_phase_ = 0.0;        // carrier
    double             voice_audio_phase_ = 0.0;  // the note being played
    double             broadcast_phase_ = 0.0;        // carrier
    double             broadcast_audio_phase_ = 0.0;  // the note being played
    double             broadcast_pilot_phase_ = 0.0;
    std::vector<Fsk>   fsk_;
    // The DMR repeater
    std::shared_ptr<DmrTransmitter> dmr_;
    std::shared_ptr<AmbeEncoder>    dmr_encoder_;
    uint64_t           dmr_symbols_ = 0;   // sent so far
    uint64_t           dmr_speech_  = 0;   // speech samples spoken so far (8 kHz)
    // The P25 radio
    std::shared_ptr<P25Transmitter> p25_;
    std::shared_ptr<ImbeEncoder>    p25_encoder_;
    uint64_t           p25_symbols_ = 0;
    uint64_t           p25_speech_  = 0;
    std::shared_ptr<P25Transmitter> p25_control_;
    uint64_t           p25_blocks_  = 0;
    // The Phase 2 voice channel
    std::shared_ptr<P25Phase2Transmitter> p25p2_;
    std::shared_ptr<AmbeEncoder>          p25p2_encoder_;
    uint64_t           p25p2_symbols_ = 0;
    uint64_t           p25p2_speech_  = 0;
    double             chirp_phase_ = 0.0;
    double             chirp_freq_ = 0.0;   // cycles/sample relative to the initial centre
    size_t             wide_pos_ = 0;
    double             wide_phase_ = 0.0;
    bool               burst_on_ = false;
    int64_t            burst_remaining_ = 0;
    float              burst_level_ = 0.0f;
    size_t             narrow_pos_ = 0;
    double             hop_phase_ = 0.0;
    double             hop_offset_ = 0.0;
    bool               hop_on_ = false;
    int64_t            hop_remaining_ = 0;
    float              hop_level_ = 0.0f;
    uint32_t           rng_state_[4] = {};

    std::vector<float> mix_;       // interleaved I/Q accumulator for one block
    std::vector<float> envelope_;  // per-sample burst envelope
};

}  // namespace spectro
