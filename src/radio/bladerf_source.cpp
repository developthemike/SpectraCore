#include "radio/bladerf_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <libbladeRF.h>

#include "util/log.hpp"
#include "util/units.hpp"

namespace spectro {

namespace {

ValueRange to_range(const struct bladerf_range* r)
{
    if (!r)
        return {};
    const double scale = r->scale != 0.0f ? static_cast<double>(r->scale) : 1.0;
    return {static_cast<double>(r->min) * scale, static_cast<double>(r->max) * scale,
            static_cast<double>(r->step) * scale};
}

std::string open_error_message(int status, const std::string& device)
{
    std::string msg = "Could not open the bladeRF";
    if (!device.empty())
        msg += " '" + device + "'";
    msg += ": ";
    msg += bladerf_strerror(status);
    msg += ".";
    if (status == BLADERF_ERR_NODEV) {
        msg += " Check that it is plugged in, that no other program (bladeRF-cli, GNU Radio, ...) "
               "is using it, and that this user may access it (udev rules / 'plugdev' group).";
    }
    return msg;
}

}  // namespace

BladeRfSource::BladeRfSource(const SourceConfig& config) : config_(config)
{
    info_.kind        = "bladeRF";
    info_.description = "bladeRF (not opened)";
    frequency_.store(config.frequency);
    gain_.store(config.gain);
    sample_rate_ = config.sample_rate;
    bandwidth_   = config.bandwidth > 0.0 ? config.bandwidth : config.sample_rate;
}

BladeRfSource::~BladeRfSource()
{
    stop();
}

void BladeRfSource::fail(const std::string& what, int status)
{
    throw std::runtime_error("bladeRF: could not " + what + ": " + bladerf_strerror(status));
}

void BladeRfSource::set_error(const std::string& message)
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = message;
}

std::string BladeRfSource::last_error() const
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

void BladeRfSource::start()
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (dev_)
        return;

    bladerf_log_set_verbosity(config_.verbose ? BLADERF_LOG_LEVEL_DEBUG : BLADERF_LOG_LEVEL_WARNING);

    int status = bladerf_open(&dev_, config_.device.empty() ? nullptr : config_.device.c_str());
    if (status != 0) {
        dev_ = nullptr;
        throw std::runtime_error(open_error_message(status, config_.device));
    }

    try {
        // --- Device identification --------------------------------------------
        status = bladerf_is_fpga_configured(dev_);
        if (status < 0)
            fail("query the FPGA state", status);
        if (status == 0) {
            throw std::runtime_error(
                "The bladeRF's FPGA is not configured. Load an FPGA image first, e.g. with "
                "'bladeRF-cli -l <image.rbf>', or install the image where libbladeRF autoloads it.");
        }

        struct bladerf_devinfo devinfo;
        std::memset(&devinfo, 0, sizeof(devinfo));
        bladerf_get_devinfo(dev_, &devinfo);

        struct bladerf_version fpga = {}, firmware = {};
        bladerf_fpga_version(dev_, &fpga);
        bladerf_fw_version(dev_, &firmware);

        const bladerf_dev_speed speed      = bladerf_device_speed(dev_);
        const char*             speed_text = speed == BLADERF_DEVICE_SPEED_SUPER ? "USB 3 SuperSpeed"
                                           : speed == BLADERF_DEVICE_SPEED_HIGH  ? "USB 2 Hi-Speed"
                                                                                 : "USB speed unknown";
        const char* board = bladerf_get_board_name(dev_);

        char description[256];
        std::snprintf(description, sizeof(description), "%s  serial %.8s  %s  FPGA %u.%u.%u  FW %u.%u.%u",
                      board ? board : "bladeRF", devinfo.serial, speed_text,
                      fpga.major, fpga.minor, fpga.patch,
                      firmware.major, firmware.minor, firmware.patch);
        info_.description = description;

        // --- Channel ------------------------------------------------------------
        const size_t rx_channels = bladerf_get_channel_count(dev_, BLADERF_RX);
        if (config_.channel < 0 || static_cast<size_t>(config_.channel) >= rx_channels) {
            throw std::runtime_error("bladeRF: RX channel " + std::to_string(config_.channel) +
                                     " does not exist (this board has " + std::to_string(rx_channels) + ")");
        }
        channel_ = BLADERF_CHANNEL_RX(config_.channel);

        // --- Sample rate --------------------------------------------------------
        const struct bladerf_range* range = nullptr;
        status = bladerf_get_sample_rate_range(dev_, channel_, &range);
        if (status != 0)
            fail("query the sample rate range", status);
        const ValueRange rate_range = to_range(range);
        if (config_.sample_rate < rate_range.min || config_.sample_rate > rate_range.max) {
            throw std::runtime_error("bladeRF: sample rate " + format_si(config_.sample_rate, "S/s") +
                                     " is outside the supported range " + format_si(rate_range.min, "S/s") +
                                     " ... " + format_si(rate_range.max, "S/s"));
        }
        bladerf_sample_rate actual_rate = 0;
        status = bladerf_set_sample_rate(dev_, channel_,
                                         static_cast<bladerf_sample_rate>(std::llround(config_.sample_rate)),
                                         &actual_rate);
        if (status != 0)
            fail("set the sample rate", status);
        sample_rate_ = static_cast<double>(actual_rate);

        if (speed != BLADERF_DEVICE_SPEED_SUPER && sample_rate_ > 8e6) {
            LOG_WARN("The bladeRF is on a %s link, which cannot carry %s. Expect dropped samples; "
                     "use a USB 3 port and cable.", speed_text, format_si(sample_rate_, "S/s").c_str());
        }

        // --- Analog bandwidth ---------------------------------------------------
        status = bladerf_get_bandwidth_range(dev_, channel_, &range);
        if (status != 0)
            fail("query the bandwidth range", status);
        const ValueRange bw_range  = to_range(range);
        const double     bw_wanted = std::clamp(config_.bandwidth > 0.0 ? config_.bandwidth : sample_rate_,
                                                bw_range.min, bw_range.max);
        bladerf_bandwidth actual_bw = 0;
        status = bladerf_set_bandwidth(dev_, channel_, static_cast<bladerf_bandwidth>(std::llround(bw_wanted)),
                                       &actual_bw);
        if (status != 0)
            fail("set the bandwidth", status);
        bandwidth_ = static_cast<double>(actual_bw);

        // --- Frequency ----------------------------------------------------------
        status = bladerf_get_frequency_range(dev_, channel_, &range);
        if (status != 0)
            fail("query the frequency range", status);
        info_.frequency_range = to_range(range);
        info_.tunable         = true;
        if (config_.frequency < info_.frequency_range.min || config_.frequency > info_.frequency_range.max) {
            throw std::runtime_error("bladeRF: frequency " + format_frequency(config_.frequency) +
                                     " is outside the tuning range " +
                                     format_frequency(info_.frequency_range.min) + " ... " +
                                     format_frequency(info_.frequency_range.max));
        }
        status = bladerf_set_frequency(dev_, channel_,
                                       static_cast<bladerf_frequency>(std::llround(config_.frequency)));
        if (status != 0)
            fail("set the frequency", status);
        bladerf_frequency actual_freq = 0;
        if (bladerf_get_frequency(dev_, channel_, &actual_freq) == 0)
            frequency_.store(static_cast<double>(actual_freq));
        else
            frequency_.store(config_.frequency);

        // --- Gain ---------------------------------------------------------------
        status = bladerf_get_gain_range(dev_, channel_, &range);
        if (status != 0)
            fail("query the gain range", status);
        info_.gain_range = to_range(range);
        info_.has_gain   = true;

        bool agc_active = false;
        if (config_.agc) {
            status = bladerf_set_gain_mode(dev_, channel_, BLADERF_GAIN_DEFAULT);
            if (status == 0) {
                agc_active = true;
            } else {
                LOG_WARN("bladeRF: automatic gain control is unavailable (%s); using manual gain",
                         bladerf_strerror(status));
            }
        }
        if (!agc_active) {
            status = bladerf_set_gain_mode(dev_, channel_, BLADERF_GAIN_MGC);
            if (status != 0 && status != BLADERF_ERR_UNSUPPORTED)
                fail("select manual gain control", status);
            const double wanted = std::clamp(config_.gain, info_.gain_range.min, info_.gain_range.max);
            status = bladerf_set_gain(dev_, channel_, static_cast<bladerf_gain>(std::lround(wanted)));
            if (status != 0)
                fail("set the gain", status);
        }
        bladerf_gain actual_gain = 0;
        if (bladerf_get_gain(dev_, channel_, &actual_gain) == 0)
            gain_.store(static_cast<double>(actual_gain));
        info_.has_gain = !agc_active;

        // --- Stream -------------------------------------------------------------
        // About 2 ms of signal per USB transfer: low latency at every sample rate
        // while keeping the transfer rate (and thus CPU overhead) modest.
        const unsigned buffer_size = std::clamp(
            static_cast<unsigned>(std::ceil(sample_rate_ * 0.002 / 1024.0)) * 1024u, 1024u, 65536u);
        const unsigned num_buffers   = 64;   // ~130 ms of slack before samples are dropped
        const unsigned num_transfers = 16;
        const unsigned timeout_ms    = 2000;
        status = bladerf_sync_config(dev_, BLADERF_RX_X1, BLADERF_FORMAT_SC16_Q11_META, num_buffers,
                                     buffer_size, num_transfers, timeout_ms);
        if (status != 0)
            fail("configure the sample stream", status);
        chunk_samples_ = static_cast<int>(buffer_size);

        status = bladerf_enable_module(dev_, channel_, true);
        if (status != 0)
            fail("enable the receiver", status);

        // The FPGA produces samples from the moment the receiver is enabled, but
        // the USB stream only starts with the first read; whatever was produced
        // in between overflows the device's FIFO.  Read and discard a short
        // stretch so that this start-up transient neither reaches the display
        // nor counts as lost samples.  It also proves that samples are flowing.
        {
            std::vector<int16_t> scratch(2 * static_cast<size_t>(buffer_size));
            long long            remaining = std::llround(sample_rate_ * 0.05);
            int                  attempts  = 0;
            while (remaining > 0) {
                struct bladerf_metadata meta;
                std::memset(&meta, 0, sizeof(meta));
                meta.flags = BLADERF_META_FLAG_RX_NOW;
                const unsigned count = static_cast<unsigned>(std::min<long long>(remaining, buffer_size));
                status = bladerf_sync_rx(dev_, scratch.data(), count, &meta, 1000);
                if (status != 0)
                    fail("start the sample stream", status);
                remaining -= meta.actual_count;
                if (++attempts > 10000)
                    throw std::runtime_error("bladeRF: the receiver is enabled but delivers no samples");
            }
        }

        streaming_            = true;
        have_timestamp_       = false;
        consecutive_timeouts_ = 0;

        LOG_INFO("bladeRF: %s", info_.description.c_str());
        LOG_INFO("bladeRF: %s at %s, bandwidth %s, gain %s", format_frequency(frequency_.load()).c_str(),
                 format_si(sample_rate_, "S/s").c_str(), format_si(bandwidth_, "Hz").c_str(),
                 agc_active ? "automatic" : (std::to_string(static_cast<int>(gain_.load())) + " dB").c_str());
    } catch (...) {
        bladerf_close(dev_);
        dev_       = nullptr;
        streaming_ = false;
        throw;
    }
}

void BladeRfSource::stop()
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!dev_)
        return;
    if (streaming_) {
        const int status = bladerf_enable_module(dev_, channel_, false);
        if (status != 0)
            LOG_WARN("bladeRF: could not disable the receiver: %s", bladerf_strerror(status));
        streaming_ = false;
    }
    bladerf_close(dev_);
    dev_ = nullptr;
}

int BladeRfSource::read(int16_t* iq, int max_samples, ReadStatus& status)
{
    status = {};
    if (!dev_ || !streaming_ || max_samples <= 0)
        return -1;

    struct bladerf_metadata meta;
    std::memset(&meta, 0, sizeof(meta));
    meta.flags = BLADERF_META_FLAG_RX_NOW;

    const int result = bladerf_sync_rx(dev_, iq, static_cast<unsigned>(max_samples), &meta, 500);
    if (result == BLADERF_ERR_TIMEOUT) {
        if (++consecutive_timeouts_ >= 8) {
            set_error("The bladeRF stopped delivering samples (stream timed out)");
            return -1;
        }
        return 0;
    }
    if (result != 0) {
        set_error(std::string("Receiving from the bladeRF failed: ") + bladerf_strerror(result));
        return -1;
    }
    consecutive_timeouts_ = 0;

    // The FPGA timestamps every block of samples; a jump means samples were lost
    // somewhere between the ADC and this call.
    if (meta.status & BLADERF_META_STATUS_OVERRUN)
        status.discontinuity = true;
    if (have_timestamp_ && meta.timestamp != next_timestamp_) {
        status.discontinuity = true;
        if (meta.timestamp > next_timestamp_)
            status.lost_samples = meta.timestamp - next_timestamp_;
    }
    next_timestamp_ = meta.timestamp + meta.actual_count;
    have_timestamp_ = true;

    return static_cast<int>(meta.actual_count);
}

bool BladeRfSource::set_frequency(double hz)
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!dev_)
        return false;

    hz = std::clamp(hz, info_.frequency_range.min, info_.frequency_range.max);
    const int status = bladerf_set_frequency(dev_, channel_, static_cast<bladerf_frequency>(std::llround(hz)));
    if (status != 0) {
        LOG_WARN("bladeRF: could not tune to %s: %s", format_frequency(hz).c_str(), bladerf_strerror(status));
        return false;
    }
    bladerf_frequency actual = 0;
    if (bladerf_get_frequency(dev_, channel_, &actual) == 0)
        frequency_.store(static_cast<double>(actual));
    else
        frequency_.store(hz);
    return true;
}

bool BladeRfSource::set_gain(double db)
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!dev_ || !info_.has_gain)
        return false;

    db = std::clamp(db, info_.gain_range.min, info_.gain_range.max);
    const int status = bladerf_set_gain(dev_, channel_, static_cast<bladerf_gain>(std::lround(db)));
    if (status != 0) {
        LOG_WARN("bladeRF: could not set the gain to %.0f dB: %s", db, bladerf_strerror(status));
        return false;
    }
    bladerf_gain actual = 0;
    if (bladerf_get_gain(dev_, channel_, &actual) == 0)
        gain_.store(static_cast<double>(actual));
    else
        gain_.store(db);
    return true;
}

}  // namespace spectro
