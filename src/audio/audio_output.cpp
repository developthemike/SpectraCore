#include "audio/audio_output.hpp"

#include <miniaudio.h>

#include "util/log.hpp"

namespace spectro {

struct AudioOutput::Impl {
    AudioFifo& fifo;
    const int  sample_rate;

    ma_context  context{};
    ma_device   device{};
    bool        open    = false;
    bool        running = false;
    bool        silent  = false;
    std::string description = "not opened";

    Impl(AudioFifo& audio, int rate) : fifo(audio), sample_rate(rate) {}

    // Runs on the sound system's thread: one float per frame, as configured.
    static void on_audio(ma_device* dev, void* output, const void* /*input*/, ma_uint32 frames)
    {
        static_cast<Impl*>(dev->pUserData)->fifo.read(static_cast<float*>(output), frames);
    }

    // Opens the default device of the first working backend among `backends`
    // (null: every backend this platform has, best first).
    bool open_device(const ma_backend* backends, ma_uint32 backend_count, std::string& error)
    {
        ma_context_config context_config   = ma_context_config_init();
        context_config.pulse.pApplicationName = "Spectrogram";
        ma_result result = ma_context_init(backends, backend_count, &context_config, &context);
        if (result != MA_SUCCESS) {
            error = ma_result_description(result);
            return false;
        }

        ma_device_config config         = ma_device_config_init(ma_device_type_playback);
        config.playback.format          = ma_format_f32;
        config.playback.channels        = 1;  // miniaudio spreads it over the device's channels
        config.sampleRate               = static_cast<ma_uint32>(sample_rate);  // and resamples if need be
        config.periodSizeInMilliseconds = 10;
        config.dataCallback             = &Impl::on_audio;
        config.pUserData                = this;
        config.pulse.pStreamNamePlayback = "FM audio";
        result = ma_device_init(&context, &config, &device);
        if (result != MA_SUCCESS) {
            error = ma_result_description(result);
            ma_context_uninit(&context);
            return false;
        }

        silent      = context.backend == ma_backend_null;
        description = silent ? "no sound device"
                             : std::string(ma_get_backend_name(context.backend)) + ": " + device.playback.name;
        open        = true;
        return true;
    }
};

AudioOutput::AudioOutput(AudioFifo& fifo, int sample_rate) : impl_(std::make_unique<Impl>(fifo, sample_rate)) {}

AudioOutput::~AudioOutput()
{
    if (impl_->open) {
        ma_device_uninit(&impl_->device);  // stops it first
        ma_context_uninit(&impl_->context);
    }
}

bool AudioOutput::start(std::string& message)
{
    Impl& impl = *impl_;
    message.clear();

    if (!impl.open) {
        std::string error;
        if (!impl.open_device(nullptr, 0, error)) {
            // A sound system is there but its device cannot be used.  Fall
            // back to the stand-in that only keeps time.
            LOG_WARN("Could not open the sound device: %s", error.c_str());
            const ma_backend fallback = ma_backend_null;
            std::string      ignored;
            if (!impl.open_device(&fallback, 1, ignored)) {
                message = "Could not open a sound device: " + error;
                return false;
            }
        }
        if (impl.silent)
            message = "No sound device was found: the audio cannot be heard";
        else
            LOG_INFO("Audio output: %s", impl.description.c_str());
    }

    if (!impl.running) {
        const ma_result result = ma_device_start(&impl.device);
        if (result != MA_SUCCESS) {
            message = std::string("Could not start the sound device: ") + ma_result_description(result);
            return false;
        }
        impl.running = true;
    }
    return true;
}

void AudioOutput::stop()
{
    Impl& impl = *impl_;
    if (impl.open && impl.running) {
        ma_device_stop(&impl.device);
        impl.running = false;
    }
}

bool AudioOutput::running() const { return impl_->running; }

bool AudioOutput::silent() const { return impl_->silent; }

const std::string& AudioOutput::description() const { return impl_->description; }

}  // namespace spectro
