// Small helpers for platform-specific conventions (keyboard modifiers,
// windowing options, and user-facing install/troubleshooting hints).
#pragma once

#include <string>

namespace spectro {

enum class HostPlatform { Linux, MacOS, Other };

constexpr HostPlatform current_platform()
{
#if defined(__APPLE__)
    return HostPlatform::MacOS;
#elif defined(__linux__)
    return HostPlatform::Linux;
#else
    return HostPlatform::Other;
#endif
}

// Primary shortcut modifier: Cmd on macOS (also accepting Ctrl for familiarity
// and cross-platform docs), Ctrl elsewhere.
constexpr bool primary_modifier_held(bool ctrl, bool super, HostPlatform platform = current_platform())
{
    return platform == HostPlatform::MacOS ? (super || ctrl) : ctrl;
}

constexpr const char* primary_modifier_name(HostPlatform platform = current_platform())
{
    return platform == HostPlatform::MacOS ? "Cmd" : "Ctrl";
}

// Validates a --x11 / --wayland windowing override for `platform`.
// Returns an empty string when valid, or an error description otherwise.
inline std::string validate_window_platform(const std::string& requested, HostPlatform platform = current_platform())
{
    if (requested.empty())
        return {};
    if (platform == HostPlatform::MacOS && (requested == "x11" || requested == "wayland"))
        return "--" + requested + " is only available on Linux (macOS uses Cocoa)";
    return {};
}

}  // namespace spectro
