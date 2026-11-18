#include "dsp/ambe_vocoder.hpp"

#include <algorithm>
#include <cstring>

#if SPECTRO_HAVE_MBELIB
#include <mbelib-neo/mbelib.h>
#endif

namespace spectro {

namespace {

#if SPECTRO_HAVE_MBELIB
// mbelib's float samples are on a scale of about 32768 / 7.
constexpr float kFromMbelib = 7.0f / 32768.0f;

void to_mbelib(const dmr::AmbeFrame& frame, char out[4][24])
{
    for (size_t r = 0; r < 4; ++r)
        std::memcpy(out[r], frame.c[r].data(), 24);
}

void to_mbelib(const p25::ImbeFrame& frame, char out[8][23])
{
    for (size_t r = 0; r < 8; ++r)
        std::memcpy(out[r], frame.c[r].data(), 23);
}
#endif

}  // namespace

// ---- Decoder ----------------------------------------------------------------------

#if SPECTRO_HAVE_MBELIB
struct AmbeDecoder::State {
    mbe_parms current{}, previous{}, enhanced{};
};

bool AmbeDecoder::available() { return true; }

AmbeDecoder::AmbeDecoder() : state_(std::make_unique<State>()) { reset(); }

void AmbeDecoder::reset() { mbe_initMbeParms(&state_->current, &state_->previous, &state_->enhanced); }

int AmbeDecoder::decode(const dmr::AmbeFrame& frame, float* samples)
{
    char fr[4][24];
    to_mbelib(frame, fr);
    char               parameters[49];
    mbe_process_result result;
    mbe_initProcessResult(&result);
    float     pcm[kFrameSamples];
    const int errors = mbe_processAmbe3600x2450Framef(pcm, &result, fr, parameters, &state_->current,
                                                      &state_->previous, &state_->enhanced);
    if (errors < 0) {
        std::fill(samples, samples + kFrameSamples, 0.0f);
        return 0;
    }
    for (int i = 0; i < kFrameSamples; ++i)
        samples[i] = std::clamp(pcm[i] * kFromMbelib, -1.0f, 1.0f);
    return errors;
}
#else
struct AmbeDecoder::State {};

bool AmbeDecoder::available() { return false; }

AmbeDecoder::AmbeDecoder() : state_(std::make_unique<State>()) {}

void AmbeDecoder::reset() {}

int AmbeDecoder::decode(const dmr::AmbeFrame&, float* samples)
{
    std::fill(samples, samples + kFrameSamples, 0.0f);
    return -1;
}
#endif

AmbeDecoder::~AmbeDecoder() = default;

// ---- Encoder ----------------------------------------------------------------------

#if SPECTRO_HAVE_MBELIB
struct AmbeEncoder::State {
    mbe_ambe2450_encoder* encoder = nullptr;
    mbe_parms             current{}, previous{}, enhanced{};
};

AmbeEncoder::AmbeEncoder() : state_(std::make_unique<State>())
{
    state_->encoder = mbe_ambe2450EncoderAlloc();
    reset();
}

AmbeEncoder::~AmbeEncoder()
{
    if (state_)
        mbe_ambe2450EncoderFree(state_->encoder);
}

void AmbeEncoder::reset()
{
    if (state_->encoder)
        mbe_ambe2450EncoderReset(state_->encoder);
    mbe_initMbeParms(&state_->current, &state_->previous, &state_->enhanced);
}

dmr::AmbeFrame AmbeEncoder::encode(const float* samples)
{
    char parameters[49];
    if (!state_->encoder ||
        mbe_encodeAmbe2450Parms(state_->encoder, samples, parameters, &state_->current, &state_->previous) < 0)
        return dmr::ambe_silence();
    mbe_moveMbeParms(&state_->current, &state_->previous);
    char fr[4][24];
    if (mbe_encodeAmbe3600x2450Frame(parameters, fr) < 0)
        return dmr::ambe_silence();
    dmr::AmbeFrame frame;
    for (size_t r = 0; r < 4; ++r)
        std::memcpy(frame.c[r].data(), fr[r], 24);
    return frame;
}
#else
struct AmbeEncoder::State {};

AmbeEncoder::AmbeEncoder() : state_(std::make_unique<State>()) {}

AmbeEncoder::~AmbeEncoder() = default;

void AmbeEncoder::reset() {}

dmr::AmbeFrame AmbeEncoder::encode(const float*) { return dmr::ambe_silence(); }
#endif

// ---- IMBE ----------------------------------------------------------------------------

#if SPECTRO_HAVE_MBELIB
struct ImbeDecoder::State {
    mbe_parms current{}, previous{}, enhanced{};
};

ImbeDecoder::ImbeDecoder() : state_(std::make_unique<State>()) { reset(); }

void ImbeDecoder::reset() { mbe_initMbeParms(&state_->current, &state_->previous, &state_->enhanced); }

int ImbeDecoder::decode(const p25::ImbeFrame& frame, float* samples)
{
    char fr[8][23];
    to_mbelib(frame, fr);
    char               parameters[88];
    mbe_process_result result;
    mbe_initProcessResult(&result);
    float     pcm[kFrameSamples];
    const int errors = mbe_processImbe7200x4400Framef(pcm, &result, fr, parameters, &state_->current,
                                                      &state_->previous, &state_->enhanced);
    if (errors < 0) {
        std::fill(samples, samples + kFrameSamples, 0.0f);
        return 0;
    }
    for (int i = 0; i < kFrameSamples; ++i)
        samples[i] = std::clamp(pcm[i] * kFromMbelib, -1.0f, 1.0f);
    return errors;
}

struct ImbeEncoder::State {
    mbe_imbe4400_encoder* encoder = nullptr;
    mbe_parms             current{}, previous{}, enhanced{};
};

ImbeEncoder::ImbeEncoder() : state_(std::make_unique<State>())
{
    state_->encoder = mbe_imbe4400EncoderAlloc();
    reset();
}

ImbeEncoder::~ImbeEncoder()
{
    if (state_)
        mbe_imbe4400EncoderFree(state_->encoder);
}

void ImbeEncoder::reset()
{
    if (state_->encoder)
        mbe_imbe4400EncoderReset(state_->encoder);
    mbe_initMbeParms(&state_->current, &state_->previous, &state_->enhanced);
}

p25::ImbeFrame ImbeEncoder::encode(const float* samples)
{
    char parameters[88];
    if (!state_->encoder ||
        mbe_encodeImbe4400Parms(state_->encoder, samples, parameters, &state_->current, &state_->previous) < 0)
        return {};
    mbe_moveMbeParms(&state_->current, &state_->previous);
    char fr[8][23];
    if (mbe_encodeImbe7200x4400Frame(parameters, fr) < 0)
        return {};
    p25::ImbeFrame frame;
    for (size_t r = 0; r < 8; ++r)
        std::memcpy(frame.c[r].data(), fr[r], 23);
    return frame;
}
#else
struct ImbeDecoder::State {};

ImbeDecoder::ImbeDecoder() : state_(std::make_unique<State>()) {}

void ImbeDecoder::reset() {}

int ImbeDecoder::decode(const p25::ImbeFrame&, float* samples)
{
    std::fill(samples, samples + kFrameSamples, 0.0f);
    return -1;
}

struct ImbeEncoder::State {};

ImbeEncoder::ImbeEncoder() : state_(std::make_unique<State>()) {}

ImbeEncoder::~ImbeEncoder() = default;

void ImbeEncoder::reset() {}

p25::ImbeFrame ImbeEncoder::encode(const float*) { return {}; }
#endif

ImbeDecoder::~ImbeDecoder() = default;

}  // namespace spectro
