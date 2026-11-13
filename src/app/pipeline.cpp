#include "app/pipeline.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace spectro {

Pipeline::Pipeline(const Options& options)
    : initial_params_(options.spectrum), columns_(options.columns), source_(make_source(options.source))
{
}

Pipeline::~Pipeline()
{
    stop();
}

void Pipeline::start()
{
    if (started_)
        return;

    // Left over from a previous run; each refers to the ones before it.
    listener_.reset();
    acquisition_.reset();
    engine_.reset();
    ring_.reset();

    source_->start();  // throws if the device or file cannot be opened

    // Room for ~0.4 s of signal between the receive and DSP threads: plenty to
    // ride out scheduling hiccups without ever touching the sample stream.
    const double rate     = source_->sample_rate();
    const size_t capacity = std::clamp<size_t>(std::bit_ceil(static_cast<size_t>(std::ceil(rate * 0.4))),
                                               static_cast<size_t>(SpectrumEngine::kMaxFftSize) * 4,
                                               size_t{1} << 24);
    ring_   = std::make_unique<IqRing>(capacity);
    engine_ = std::make_unique<SpectrumEngine>(*ring_, rate, columns_);
    engine_->set_params(initial_params_);
    acquisition_ = std::make_unique<Acquisition>(*source_, *ring_);
    listener_    = std::make_unique<Listener>(*source_, *acquisition_);

    engine_->start();
    listener_->start();
    acquisition_->start();
    started_ = true;
}

void Pipeline::stop()
{
    if (!started_)
        return;
    // Order matters: stop reading, let the DSP thread finish, then close the device.
    acquisition_->stop();
    listener_->stop();
    engine_->stop();
    source_->stop();
    started_ = false;
}

}  // namespace spectro
