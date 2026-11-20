// The vocoders of digital voice, 20 ms frames of 8 kHz speech, from
// mbelib-neo: AMBE+2 for DMR, IMBE for P25 Phase 1.
//
// They are optional: the codecs are patented and mbelib-neo is GPL, so they
// are built only with SPECTRO_WITH_MBELIB=ON (see the README).  Without it,
// available() is false, decode() gives silence and encode() sends frames of
// silence (DMR) or of zeros (P25).
#pragma once

#include <memory>

#include "dsp/dmr_coding.hpp"
#include "dsp/p25_coding.hpp"

namespace spectro {

class AmbeDecoder {
public:
    static constexpr int kFrameSamples = 160;   // 20 ms
    static constexpr int kSampleRate   = 8000;

    static bool available();

    AmbeDecoder();
    ~AmbeDecoder();
    AmbeDecoder(const AmbeDecoder&)            = delete;
    AmbeDecoder& operator=(const AmbeDecoder&) = delete;

    // Before a new call.
    void reset();
    // One frame into kFrameSamples samples in -1 ... 1.  Returns the bit errors
    // the frame's codes corrected (a damaged frame is repeated, many in a row
    // are muted), or -1 without a vocoder.
    int decode(const dmr::AmbeFrame& frame, float* samples);

private:
    struct State;
    std::unique_ptr<State> state_;
};

class AmbeEncoder {
public:
    static constexpr int kFrameSamples = AmbeDecoder::kFrameSamples;

    static bool available() { return AmbeDecoder::available(); }

    AmbeEncoder();
    ~AmbeEncoder();
    AmbeEncoder(const AmbeEncoder&)            = delete;
    AmbeEncoder& operator=(const AmbeEncoder&) = delete;

    void reset();
    // kFrameSamples samples at 8 kHz, in -1 ... 1.
    dmr::AmbeFrame encode(const float* samples);

private:
    struct State;
    std::unique_ptr<State> state_;
};

// IMBE, as P25 Phase 1 uses it (7200 bit/s with its error-correcting codes).
class ImbeDecoder {
public:
    static constexpr int kFrameSamples = AmbeDecoder::kFrameSamples;

    static bool available() { return AmbeDecoder::available(); }

    ImbeDecoder();
    ~ImbeDecoder();
    ImbeDecoder(const ImbeDecoder&)            = delete;
    ImbeDecoder& operator=(const ImbeDecoder&) = delete;

    void reset();
    // As AmbeDecoder::decode().
    int decode(const p25::ImbeFrame& frame, float* samples);

private:
    struct State;
    std::unique_ptr<State> state_;
};

class ImbeEncoder {
public:
    static constexpr int kFrameSamples = AmbeDecoder::kFrameSamples;

    static bool available() { return AmbeDecoder::available(); }

    ImbeEncoder();
    ~ImbeEncoder();
    ImbeEncoder(const ImbeEncoder&)            = delete;
    ImbeEncoder& operator=(const ImbeEncoder&) = delete;

    void           reset();
    p25::ImbeFrame encode(const float* samples);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace spectro
