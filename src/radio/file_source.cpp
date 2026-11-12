#include "radio/file_source.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace spectro {

namespace {

constexpr int kMaxBlock = 16384;

std::string lower_extension(const std::string& path)
{
    const size_t dot   = path.find_last_of('.');
    const size_t slash = path.find_last_of("/\\");
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return {};
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

int16_t to_q11(float v)
{
    return static_cast<int16_t>(std::lrintf(std::clamp(v, -2047.0f, 2047.0f)));
}

}  // namespace

bool FileSource::resolve_format(const std::string& name, const std::string& path, Format& format)
{
    std::string key = name;
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (key.empty() || key == "auto") {
        key = lower_extension(path);
        if (key == "bin" || key == "iq" || key == "raw" || key.empty())
            key = "sc16";
        else if (key == "cfile" || key == "fc32")
            key = "cf32";
    }

    if (key == "sc16" || key == "sc16q11")
        format = Format::SC16Q11;
    else if (key == "cs16" || key == "s16")
        format = Format::CS16;
    else if (key == "cf32" || key == "f32")
        format = Format::CF32;
    else if (key == "cs8" || key == "s8")
        format = Format::CS8;
    else if (key == "cu8" || key == "u8")
        format = Format::CU8;
    else
        return false;
    return true;
}

size_t FileSource::bytes_per_sample(Format format)
{
    switch (format) {
    case Format::SC16Q11:
    case Format::CS16: return 4;
    case Format::CF32: return 8;
    case Format::CS8:
    case Format::CU8:  return 2;
    }
    return 4;
}

FileSource::FileSource(const SourceConfig& config) : config_(config)
{
    info_.kind        = "file";
    info_.description = "File playback: " + config.file_path;
    if (config.file_path.empty())
        throw std::runtime_error("file source: no file given (use --file <path>)");
    if (!resolve_format(config.file_format, config.file_path, format_)) {
        throw std::runtime_error("file source: unknown sample format '" + config.file_format +
                                 "' (expected sc16, cs16, cf32, cs8 or cu8)");
    }
    if (!(config.sample_rate > 0.0))
        throw std::runtime_error("file source: the sample rate must be positive");
}

FileSource::~FileSource()
{
    stop();
}

void FileSource::set_error(const std::string& message)
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = message;
}

std::string FileSource::last_error() const
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

void FileSource::start()
{
    if (file_)
        return;
    file_ = std::fopen(config_.file_path.c_str(), "rb");
    if (!file_) {
        throw std::runtime_error("Could not open '" + config_.file_path + "': " + std::strerror(errno));
    }

    // Reject files too short to hold a single sample, which would otherwise
    // make a looping source spin forever.
    const size_t bps = bytes_per_sample(format_);
    raw_.resize(static_cast<size_t>(kMaxBlock) * bps);
    const size_t got = std::fread(raw_.data(), 1, bps, file_);
    if (got < bps) {
        std::fclose(file_);
        file_ = nullptr;
        throw std::runtime_error("'" + config_.file_path + "' contains no samples");
    }
    std::rewind(file_);

    produced_   = 0;
    start_time_ = std::chrono::steady_clock::now();
    running_.store(true);
}

void FileSource::stop()
{
    running_.store(false);
    if (file_) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

int FileSource::read(int16_t* iq, int max_samples, ReadStatus& status)
{
    status = {};
    if (!file_ || !running_.load(std::memory_order_relaxed) || max_samples <= 0)
        return -1;

    const size_t bps    = bytes_per_sample(format_);
    const size_t wanted = static_cast<size_t>(std::min(max_samples, kMaxBlock));

    size_t samples = std::fread(raw_.data(), bps, wanted, file_);
    if (samples == 0) {
        if (std::ferror(file_)) {
            set_error("Reading '" + config_.file_path + "' failed: " + std::strerror(errno));
            return -1;
        }
        if (!config_.file_loop) {
            set_error("End of file");
            return -1;
        }
        std::rewind(file_);
        status.discontinuity = true;
        samples = std::fread(raw_.data(), bps, wanted, file_);
        if (samples == 0) {
            set_error("Reading '" + config_.file_path + "' failed after rewinding");
            return -1;
        }
    }

    const size_t values = 2 * samples;
    switch (format_) {
    case Format::SC16Q11:
        std::memcpy(iq, raw_.data(), values * sizeof(int16_t));
        break;
    case Format::CS16: {
        const int16_t* src = reinterpret_cast<const int16_t*>(raw_.data());
        for (size_t i = 0; i < values; ++i)
            iq[i] = static_cast<int16_t>(src[i] >> 4);
        break;
    }
    case Format::CF32: {
        const float* src = reinterpret_cast<const float*>(raw_.data());
        for (size_t i = 0; i < values; ++i)
            iq[i] = to_q11(src[i] * 2048.0f);
        break;
    }
    case Format::CS8: {
        const int8_t* src = reinterpret_cast<const int8_t*>(raw_.data());
        for (size_t i = 0; i < values; ++i)
            iq[i] = static_cast<int16_t>(src[i] * 16);
        break;
    }
    case Format::CU8: {
        const uint8_t* src = raw_.data();
        for (size_t i = 0; i < values; ++i)
            iq[i] = to_q11((static_cast<float>(src[i]) - 127.5f) * 16.0f);
        break;
    }
    }

    // Pace playback to the recording's sample rate.
    produced_ += samples;
    const auto due = start_time_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                       std::chrono::duration<double>(static_cast<double>(produced_) / config_.sample_rate));
    const auto now = std::chrono::steady_clock::now();
    if (due > now)
        std::this_thread::sleep_until(due);
    else if (now - due > std::chrono::milliseconds(250))
        start_time_ += now - due;

    return static_cast<int>(samples);
}

}  // namespace spectro
