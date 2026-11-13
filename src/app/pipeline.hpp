// The signal path shared by the GUI and the headless mode:
//
//   IqSource --(receive thread)--> IqRing --(DSP thread)--> spectrogram rows
//                       |
//                       +--(only while listening)--> audio thread --> audio FIFO
#pragma once

#include <memory>

#include "app/options.hpp"
#include "audio/listener.hpp"
#include "dsp/spectrum_engine.hpp"
#include "radio/acquisition.hpp"
#include "radio/iq_source.hpp"
#include "util/iq_ring.hpp"

namespace spectro {

class Pipeline {
public:
    explicit Pipeline(const Options& options);
    ~Pipeline();

    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Opens the source and starts the threads.  Throws std::runtime_error with
    // a user-presentable message if the source cannot be started.
    void start();
    void stop();
    bool started() const { return started_; }

    IqSource&       source() { return *source_; }
    const IqSource& source() const { return *source_; }
    SpectrumEngine& engine() { return *engine_; }
    Acquisition&    acquisition() { return *acquisition_; }
    Listener&       listener() { return *listener_; }

    int columns() const { return columns_; }

private:
    SpectrumParams initial_params_;
    int            columns_;
    bool           started_ = false;

    // Declaration order matters: destruction happens in reverse.
    std::unique_ptr<IqSource>       source_;
    std::unique_ptr<IqRing>         ring_;
    std::unique_ptr<SpectrumEngine> engine_;
    std::unique_ptr<Acquisition>    acquisition_;
    std::unique_ptr<Listener>       listener_;
};

}  // namespace spectro
