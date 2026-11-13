#include "radio/rtlsdr_source.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <rtl-sdr.h>

#include "radio/sample_convert.hpp"
#include "util/log.hpp"
#include "util/thread_util.hpp"
#include "util/units.hpp"

namespace spectro {

namespace {

struct TunerInfo {
    const char* name;
    double      min_hz;
    double      max_hz;
};

// Nominal tuning ranges.  librtlsdr does not report them, and most tuners do
// not fail when asked for a frequency they cannot reach: they just deliver noise.
TunerInfo tuner_info(enum rtlsdr_tuner tuner)
{
    switch (tuner) {
    case RTLSDR_TUNER_E4000:  return {"E4000", 52e6, 2200e6};
    case RTLSDR_TUNER_FC0012: return {"FC0012", 22e6, 948.6e6};
    case RTLSDR_TUNER_FC0013: return {"FC0013", 22e6, 1100e6};
    case RTLSDR_TUNER_FC2580: return {"FC2580", 146e6, 924e6};
    case RTLSDR_TUNER_R820T:  return {"R820T", 24e6, 1766e6};
    case RTLSDR_TUNER_R828D:  return {"R828D", 24e6, 1766e6};
    case RTLSDR_TUNER_UNKNOWN: break;
    }
    return {nullptr, 0.0, 0.0};
}

// librtlsdr reports failures as negative libusb error codes.
std::string open_error_message(int status, const std::string& device)
{
    std::string msg = "Could not open the RTL-SDR";
    if (!device.empty())
        msg += " '" + device + "'";
    msg += " (error " + std::to_string(status) + ").";
    if (status == -3) {  // LIBUSB_ERROR_ACCESS
        msg += " This user may not access it: install the udev rules that come with librtlsdr "
               "(package 'rtl-sdr') and plug the dongle in again.";
    } else if (status == -6) {  // LIBUSB_ERROR_BUSY
        msg += " It is in use by another program (rtl_tcp, GQRX, ...) or by the kernel's DVB-T driver "
               "(unload it with 'sudo rmmod dvb_usb_rtl28xxu', or blacklist it).";
    } else {
        msg += " Check that no other program (rtl_tcp, GQRX, ...) or the kernel's DVB-T driver "
               "(dvb_usb_rtl28xxu) is using it, and that this user may access it (udev rules).";
    }
    return msg;
}

bool all_digits(const std::string& text)
{
    return !text.empty() && std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

}  // namespace

RtlSdrSource::RtlSdrSource(const SourceConfig& config) : config_(config)
{
    info_.kind        = "RTL-SDR";
    info_.description = "RTL-SDR (not opened)";
    frequency_.store(config.frequency);
    gain_.store(config.gain);
    sample_rate_ = config.sample_rate;
    bandwidth_   = config.bandwidth > 0.0 ? config.bandwidth : config.sample_rate;
}

RtlSdrSource::~RtlSdrSource()
{
    stop();
}

void RtlSdrSource::set_error(const std::string& message)
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = message;
}

std::string RtlSdrSource::last_error() const
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

void RtlSdrSource::start()
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (dev_)
        return;

    if (!rtlsdr_rate_supported(config_.sample_rate)) {
        throw std::runtime_error("RTL-SDR: sample rate " + format_si(config_.sample_rate, "S/s") +
                                 " is not supported; use 225.001 k ... 300 k or 900.001 k ... 3.2 M "
                                 "(2.4 M is the highest that is reliably lossless)");
    }

    // --- Device selection ---------------------------------------------------
    const uint32_t count = rtlsdr_get_device_count();
    if (count == 0) {
        throw std::runtime_error(
            "No RTL-SDR was found. Check that it is plugged in and that this user may access it "
            "(the udev rules that come with librtlsdr, package 'rtl-sdr').");
    }
    uint32_t index = 0;
    if (!config_.device.empty()) {
        // A serial number, or else an index.  (Serials first: the usual factory
        // serial, "00000001", is all digits too.)
        const int by_serial = rtlsdr_get_index_by_serial(config_.device.c_str());
        if (by_serial >= 0) {
            index = static_cast<uint32_t>(by_serial);
        } else if (all_digits(config_.device) && config_.device.size() <= 3 &&
                   static_cast<uint32_t>(std::stoi(config_.device)) < count) {
            index = static_cast<uint32_t>(std::stoi(config_.device));
        } else {
            throw std::runtime_error("No RTL-SDR with the serial number or index '" + config_.device +
                                     "' was found (" + std::to_string(count) + " connected)");
        }
    }

    char product[256] = "", serial[256] = "";
    rtlsdr_get_device_usb_strings(index, nullptr, product, serial);

    int status = rtlsdr_open(&dev_, index);
    if (status != 0) {
        dev_ = nullptr;
        throw std::runtime_error(open_error_message(status, config_.device));
    }

    try {
        // --- Device identification --------------------------------------------
        const TunerInfo tuner = tuner_info(rtlsdr_get_tuner_type(dev_));
        if (!tuner.name)
            throw std::runtime_error("RTL-SDR: this dongle has no supported tuner");

        char description[640];
        std::snprintf(description, sizeof(description), "%s  serial %s  %s tuner",
                      product[0] ? product : rtlsdr_get_device_name(index), serial[0] ? serial : "?",
                      tuner.name);
        info_.description = description;

        // --- Sample rate --------------------------------------------------------
        status = rtlsdr_set_sample_rate(dev_, static_cast<uint32_t>(std::llround(config_.sample_rate)));
        if (status != 0)
            throw std::runtime_error("RTL-SDR: could not set the sample rate to " +
                                     format_si(config_.sample_rate, "S/s"));
        const uint32_t actual_rate = rtlsdr_get_sample_rate(dev_);
        sample_rate_ = actual_rate ? static_cast<double>(actual_rate) : config_.sample_rate;
        if (sample_rate_ > 2.4e6) {
            LOG_WARN("RTL-SDR: above 2.4 MS/s most dongles lose samples, and they cannot report it");
        }

        // --- Crystal error correction -------------------------------------------
        // (librtlsdr rejects a value equal to the current one, which starts at 0.)
        if (config_.ppm != 0 && rtlsdr_set_freq_correction(dev_, config_.ppm) != 0)
            LOG_WARN("RTL-SDR: could not set the frequency correction to %d ppm", config_.ppm);

        // --- Tuner bandwidth ----------------------------------------------------
        // 0 lets the tuner driver pick the filter that suits the sample rate.
        const uint32_t bw_wanted =
            config_.bandwidth > 0.0 ? static_cast<uint32_t>(std::llround(config_.bandwidth)) : 0u;
        status = rtlsdr_set_tuner_bandwidth(dev_, bw_wanted);
        if (status != 0)
            LOG_WARN("RTL-SDR: could not set the tuner bandwidth");
        bandwidth_ = bw_wanted != 0 && status == 0 ? static_cast<double>(bw_wanted) : sample_rate_;

        // --- Frequency ----------------------------------------------------------
        info_.frequency_range = {tuner.min_hz, tuner.max_hz, 1.0};
        info_.tunable         = true;
        if (config_.frequency < tuner.min_hz || config_.frequency > tuner.max_hz) {
            throw std::runtime_error("RTL-SDR: frequency " + format_frequency(config_.frequency) +
                                     " is outside the " + tuner.name + " tuner's range " +
                                     format_frequency(tuner.min_hz) + " ... " + format_frequency(tuner.max_hz));
        }
        status = rtlsdr_set_center_freq(dev_, static_cast<uint32_t>(std::llround(config_.frequency)));
        if (status != 0)
            throw std::runtime_error("RTL-SDR: could not tune to " + format_frequency(config_.frequency));
        const uint32_t actual_freq = rtlsdr_get_center_freq(dev_);
        frequency_.store(actual_freq ? static_cast<double>(actual_freq) : config_.frequency);

        // --- Gain ---------------------------------------------------------------
        gains_db_.clear();
        const int gain_count = rtlsdr_get_tuner_gains(dev_, nullptr);
        if (gain_count > 0) {
            std::vector<int> tenths(static_cast<size_t>(gain_count));
            rtlsdr_get_tuner_gains(dev_, tenths.data());
            for (const int g : tenths)
                gains_db_.push_back(g / 10.0);
            std::sort(gains_db_.begin(), gains_db_.end());
            info_.gain_range = {gains_db_.front(), gains_db_.back(), 0.0};
        }

        const bool manual_gain = !config_.agc && !gains_db_.empty();
        status = rtlsdr_set_tuner_gain_mode(dev_, manual_gain ? 1 : 0);
        if (status != 0)
            throw std::runtime_error("RTL-SDR: could not select the gain mode");
        if (manual_gain) {
            const double wanted = gains_db_[rtlsdr_pick_gain(gains_db_, config_.gain, config_.gain)];
            status = rtlsdr_set_tuner_gain(dev_, static_cast<int>(std::lround(wanted * 10.0)));
            if (status != 0)
                throw std::runtime_error("RTL-SDR: could not set the gain");
            gain_.store(wanted);
        } else {
            gain_.store(0.0);
        }
        info_.has_gain = manual_gain;

        // --- Stream -------------------------------------------------------------
        // About 10 ms of signal per USB buffer, in whole 16 KiB USB transfers.
        const double   bytes_per_sec = 2.0 * sample_rate_;
        const uint32_t buffer_bytes  = std::clamp(
            static_cast<uint32_t>(std::ceil(bytes_per_sec * 0.010 / 16384.0)) * 16384u, 16384u, 262144u);
        const uint32_t buffer_count = 15;
        chunk_samples_ = static_cast<int>(buffer_bytes / 2);

        status = rtlsdr_reset_buffer(dev_);
        if (status != 0)
            throw std::runtime_error("RTL-SDR: could not reset the sample buffer");

        {
            std::lock_guard<std::mutex> fifo_lock(fifo_mutex_);
            // Half a second of slack between librtlsdr's thread and read().
            const size_t capacity =
                std::max<size_t>(static_cast<size_t>(bytes_per_sec * 0.5), size_t{8} * buffer_bytes);
            fifo_.assign((capacity + 1) & ~size_t{1}, 0);
            fifo_head_ = 0;
            fifo_size_ = 0;
            // The first samples after tuning are a settling transient: keep
            // them off the display.
            skip_bytes_      = static_cast<size_t>(sample_rate_ * 0.05) * 2;
            dropped_samples_ = 0;
            stream_ended_    = false;
            stream_result_   = 0;
        }
        consecutive_timeouts_ = 0;
        thread_ = std::thread([this, buffer_count, buffer_bytes] { stream_thread(buffer_count, buffer_bytes); });

        // Wait for the first samples, which proves that they are flowing.
        {
            std::unique_lock<std::mutex> fifo_lock(fifo_mutex_);
            const bool ready = fifo_cv_.wait_for(fifo_lock, std::chrono::seconds(3),
                                                 [this] { return fifo_size_ > 0 || stream_ended_; });
            if (!ready || fifo_size_ == 0)
                throw std::runtime_error("RTL-SDR: the receiver was started but delivers no samples");
        }

        LOG_INFO("RTL-SDR: %s", info_.description.c_str());
        char gain_text[32] = "automatic";
        if (manual_gain)
            std::snprintf(gain_text, sizeof(gain_text), "%.1f dB", gain_.load());
        LOG_INFO("RTL-SDR: %s at %s, gain %s", format_frequency(frequency_.load()).c_str(),
                 format_si(sample_rate_, "S/s").c_str(), gain_text);
    } catch (...) {
        stop_stream();
        rtlsdr_close(dev_);
        dev_ = nullptr;
        throw;
    }
}

void RtlSdrSource::stop()
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!dev_)
        return;
    stop_stream();
    rtlsdr_close(dev_);
    dev_ = nullptr;
}

void RtlSdrSource::stop_stream()
{
    if (!thread_.joinable())
        return;
    // Cancelling only works once librtlsdr's loop is running, so repeat it
    // until the thread reports that the loop has ended.
    {
        std::unique_lock<std::mutex> lock(fifo_mutex_);
        while (!stream_ended_) {
            lock.unlock();
            rtlsdr_cancel_async(dev_);
            lock.lock();
            fifo_cv_.wait_for(lock, std::chrono::milliseconds(20), [this] { return stream_ended_; });
        }
    }
    thread_.join();
}

void RtlSdrSource::stream_thread(uint32_t buffer_count, uint32_t buffer_bytes)
{
    set_current_thread_name("spectro-rtlusb");
    // Blocks until cancelled or until the device is lost.
    const int result = rtlsdr_read_async(dev_, &RtlSdrSource::usb_callback, this, buffer_count, buffer_bytes);
    {
        std::lock_guard<std::mutex> lock(fifo_mutex_);
        stream_ended_  = true;
        stream_result_ = result;
    }
    fifo_cv_.notify_all();
}

void RtlSdrSource::usb_callback(unsigned char* data, uint32_t length, void* context)
{
    static_cast<RtlSdrSource*>(context)->on_usb_data(data, length);
}

// Runs on librtlsdr's thread.  Never blocks for longer than a copy, so the
// USB transfers are always resubmitted in time.
void RtlSdrSource::on_usb_data(const unsigned char* data, size_t length)
{
    length &= ~size_t{1};  // whole I/Q pairs only
    {
        std::lock_guard<std::mutex> lock(fifo_mutex_);
        if (skip_bytes_ > 0) {
            const size_t skipped = std::min(skip_bytes_, length);
            skip_bytes_ -= skipped;
            data        += skipped;
            length      -= skipped;
        }
        if (length == 0)
            return;
        if (fifo_.size() - fifo_size_ < length) {
            dropped_samples_ += length / 2;  // read() is not keeping up
            return;
        }
        const size_t tail  = (fifo_head_ + fifo_size_) % fifo_.size();
        const size_t first = std::min(length, fifo_.size() - tail);
        std::copy_n(data, first, fifo_.begin() + static_cast<std::ptrdiff_t>(tail));
        std::copy_n(data + first, length - first, fifo_.begin());
        fifo_size_ += length;
    }
    fifo_cv_.notify_one();
}

int RtlSdrSource::read(int16_t* iq, int max_samples, ReadStatus& status)
{
    status = {};
    if (!dev_ || max_samples <= 0)
        return -1;

    std::unique_lock<std::mutex> lock(fifo_mutex_);
    fifo_cv_.wait_for(lock, std::chrono::milliseconds(500), [this] { return fifo_size_ >= 2 || stream_ended_; });

    if (fifo_size_ < 2) {
        if (stream_ended_) {
            set_error("The RTL-SDR stopped delivering samples (was it unplugged?)");
            return -1;
        }
        if (++consecutive_timeouts_ >= 8) {
            set_error("The RTL-SDR stopped delivering samples (stream timed out)");
            return -1;
        }
        return 0;
    }
    consecutive_timeouts_ = 0;

    if (dropped_samples_ > 0) {
        status.discontinuity = true;
        status.lost_samples  = dropped_samples_;
        dropped_samples_     = 0;
    }

    // The FIFO's size and read position are always even, so a wrap never
    // splits an I/Q pair.
    const size_t values = 2 * std::min(static_cast<size_t>(max_samples), fifo_size_ / 2);
    const size_t first  = std::min(values, fifo_.size() - fifo_head_);
    cu8_to_q11(fifo_.data() + fifo_head_, iq, first);
    cu8_to_q11(fifo_.data(), iq + first, values - first);
    fifo_head_ = (fifo_head_ + values) % fifo_.size();
    fifo_size_ -= values;
    return static_cast<int>(values / 2);
}

bool RtlSdrSource::set_frequency(double hz)
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!dev_)
        return false;

    hz = std::clamp(hz, info_.frequency_range.min, info_.frequency_range.max);
    if (rtlsdr_set_center_freq(dev_, static_cast<uint32_t>(std::llround(hz))) != 0) {
        LOG_WARN("RTL-SDR: could not tune to %s", format_frequency(hz).c_str());
        return false;
    }
    const uint32_t actual = rtlsdr_get_center_freq(dev_);
    frequency_.store(actual ? static_cast<double>(actual) : hz);
    return true;
}

bool RtlSdrSource::set_gain(double db)
{
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (!dev_ || !info_.has_gain)
        return false;

    const double wanted = gains_db_[rtlsdr_pick_gain(gains_db_, gain_.load(), db)];
    if (rtlsdr_set_tuner_gain(dev_, static_cast<int>(std::lround(wanted * 10.0))) != 0) {
        LOG_WARN("RTL-SDR: could not set the gain to %.1f dB", wanted);
        return false;
    }
    gain_.store(wanted);
    return true;
}

}  // namespace spectro
