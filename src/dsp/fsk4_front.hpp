// The front end of a receiver for 4FSK at 4800 Bd, as DMR and P25 Phase 1
// send it, working on what an FM demodulator gives: the instantaneous
// frequency of the channel.  Also for P25 Phase 2's differential QPSK at
// 6000 Bd: through an FM demodulator its phase steps of +-45 and +-135
// degrees look like four frequencies, each held for a symbol.
//
//   - A matched filter: root raised cosine (rolloff 0.2) for 4FSK, or the
//     mean over one symbol for differential QPSK, which sums the symbol's
//     phase step.
//   - A history of its output, for slicing symbols at any (fractional) time,
//     also a little in the past.
//   - A search for sync patterns: 24 symbols (20 for Phase 2), all outer (+3
//     or -3).  At every sample, each pattern is fitted to the symbols that end there by least
//     squares, value = level x pattern + dc; how well it fits, relative to the
//     level, says whether a pattern is there, the best fit over the samples
//     around it exactly where - and the fit gives the deviation and the
//     tuning offset too.
//
// Not thread safe: one thread calls everything.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace spectro {

class Fsk4Front {
public:
    static constexpr double kSymbolRate = 4800.0;

    struct Options {
        double symbol_rate     = kSymbolRate;
        size_t pattern_symbols = 24;    // 20 for P25 Phase 2
        bool   boxcar          = false; // the mean over a symbol, not a root raised cosine
    };

    struct Sync {
        size_t pattern = 0;    // which, as an index into the patterns given
        double time    = 0.0;  // of its first symbol, in samples (see sample())
        double level   = 0.0;  // of the outer symbols, Hz
        double dc      = 0.0;  // offset of the centre, Hz
    };

    // `patterns`: 2 x pattern_symbols bits each, the first symbol's dibit
    // highest.  `history_symbols`: how far back sample() must reach.
    Fsk4Front(double sample_rate, std::vector<uint64_t> patterns, double history_symbols);
    Fsk4Front(double sample_rate, std::vector<uint64_t> patterns, double history_symbols, Options options);

    void reset();
    // The sample rate may change a little (the audio rate is trimmed).
    void set_sample_rate(double sample_rate) { sps_ = sample_rate / symbol_rate_; }

    // Takes one frequency sample (Hz).  Returns true when a sync pattern has
    // just been placed, a little more than half a symbol after its end.
    bool push(float frequency, Sync& sync);

    double   sps() const { return sps_; }
    uint64_t written() const { return written_; }
    // The filtered frequency at time t (samples since the start, fractional).
    double sample(double t) const;
    // The earliest time sample() can still reach.
    double oldest() const { return static_cast<double>(written_) - static_cast<double>(history_.size()) + 2.0; }

    // The symbol (+3, +1, -1, -3) nearest to `value` (dc removed).
    static int decide(double value, double level)
    {
        const double third = level * (2.0 / 3.0);
        return value > third ? 3 : value > 0.0 ? 1 : value > -third ? -1 : -3;
    }

private:
    struct Candidate {
        bool     active  = false;
        uint64_t index   = 0;    // sample of the pattern's last symbol
        double   misfit  = 0.0;  // the lower, the better
        size_t   pattern = 0;
        double   level = 0.0, dc = 0.0;
        double   before = 0.0, at = 0.0, after = 0.0;  // correlation either side, for the fine timing
        bool     have_after = false;
    };

    bool search(Sync& sync);

    const double symbol_rate_;
    const size_t length_;  // symbols in a pattern
    const double nominal_sps_;
    double       sps_;

    std::vector<float> taps_;
    std::vector<float> fir_;  // the last taps_.size() inputs, twice over
    size_t             fir_pos_ = 0;
    std::vector<float> history_;  // filtered, a ring
    uint64_t           written_ = 0;
    size_t             mask_    = 0;

    std::vector<std::array<int8_t, 24>> patterns_;
    std::vector<double>                 pattern_mean_;
    std::vector<double>                 correlation_, last_correlation_;
    Candidate                           candidate_;
};

}  // namespace spectro
