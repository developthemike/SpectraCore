// Minimal thread-safe logging to stderr.
#pragma once

namespace spectro {

enum class LogLevel : int { Debug = 0, Info, Warn, Error };

void     log_set_level(LogLevel level);
LogLevel log_level();

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
void log_write(LogLevel level, const char* fmt, ...);

}  // namespace spectro

#define LOG_DEBUG(...) ::spectro::log_write(::spectro::LogLevel::Debug, __VA_ARGS__)
#define LOG_INFO(...)  ::spectro::log_write(::spectro::LogLevel::Info, __VA_ARGS__)
#define LOG_WARN(...)  ::spectro::log_write(::spectro::LogLevel::Warn, __VA_ARGS__)
#define LOG_ERROR(...) ::spectro::log_write(::spectro::LogLevel::Error, __VA_ARGS__)
