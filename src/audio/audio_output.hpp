// Plays the audio waiting in an AudioFifo on the default sound device.
//
// Built on miniaudio, which talks to PulseAudio / PipeWire, ALSA or JACK -
// whichever it finds at run time.  Where there is no sound device at all, a
// silent stand-in fetches the audio at the right pace, so that everything
// upstream behaves the same with and without one.
#pragma once

#include <memory>
#include <string>

#include "audio/audio_fifo.hpp"

namespace spectro {

class AudioOutput {
public:
    // `fifo` holds mono audio at `sample_rate` and must outlive this object.
    AudioOutput(AudioFifo& fifo, int sample_rate);
    ~AudioOutput();

    AudioOutput(const AudioOutput&)            = delete;
    AudioOutput& operator=(const AudioOutput&) = delete;

    // Opens the device the first time, then starts playing.  Returns false,
    // with a message for the user, if nothing can be played at all.
    bool start(std::string& message);
    // Stops playing; the device stays open for the next start().
    void stop();

    bool running() const;
    // True if no sound device could be opened and the audio is being discarded.
    bool silent() const;
    // The sound system and device in use, e.g. "PulseAudio: Built-in Audio".
    const std::string& description() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace spectro
