#include "util/log.hpp"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace spectro {

namespace {
std::atomic<int> g_level{static_cast<int>(LogLevel::Info)};
std::mutex       g_mutex;
const auto       g_start = std::chrono::steady_clock::now();

const char* level_tag(LogLevel level)
{
    switch (level) {
    case LogLevel::Debug: return "debug";
    case LogLevel::Info:  return "info ";
    case LogLevel::Warn:  return "WARN ";
    case LogLevel::Error: return "ERROR";
    }
    return "?";
}
}  // namespace

void log_set_level(LogLevel level) { g_level.store(static_cast<int>(level), std::memory_order_relaxed); }

LogLevel log_level() { return static_cast<LogLevel>(g_level.load(std::memory_order_relaxed)); }

void log_write(LogLevel level, const char* fmt, ...)
{
    if (static_cast<int>(level) < g_level.load(std::memory_order_relaxed))
        return;

    char    message[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);

    const double t =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();

    std::lock_guard<std::mutex> lock(g_mutex);
    std::fprintf(stderr, "[%9.3f] %s  %s\n", t, level_tag(level), message);
    std::fflush(stderr);
}

}  // namespace spectro
