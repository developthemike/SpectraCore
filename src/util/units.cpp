#include "util/units.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace spectro {

bool parse_quantity(std::string_view text, double& value)
{
    // Trim surrounding whitespace.
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
        text.remove_prefix(1);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
        text.remove_suffix(1);
    if (text.empty())
        return false;

    const std::string buffer(text);
    char*             end    = nullptr;
    const double      number = std::strtod(buffer.c_str(), &end);
    if (end == buffer.c_str() || !std::isfinite(number))
        return false;

    std::string_view rest(end);
    while (!rest.empty() && std::isspace(static_cast<unsigned char>(rest.front())))
        rest.remove_prefix(1);

    double scale = 1.0;
    if (!rest.empty()) {
        switch (rest.front()) {
        case 'k': case 'K': scale = 1e3; rest.remove_prefix(1); break;
        case 'M':           scale = 1e6; rest.remove_prefix(1); break;
        case 'G': case 'g': scale = 1e9; rest.remove_prefix(1); break;
        case 'm':
            // Accept a lower-case "m" as mega: nobody tunes a radio in millihertz.
            scale = 1e6; rest.remove_prefix(1); break;
        default: break;
        }
    }

    // Optional unit, case-insensitive.
    std::string unit(rest);
    std::transform(unit.begin(), unit.end(), unit.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!unit.empty() && unit != "hz" && unit != "sps" && unit != "s/s" && unit != "db")
        return false;

    value = number * scale;
    return true;
}

std::string format_frequency(double hz)
{
    char         buf[64];
    const double a = std::fabs(hz);
    if (a >= 1e9)
        std::snprintf(buf, sizeof(buf), "%.6f GHz", hz / 1e9);
    else if (a >= 1e6)
        std::snprintf(buf, sizeof(buf), "%.3f MHz", hz / 1e6);
    else if (a >= 1e3)
        std::snprintf(buf, sizeof(buf), "%.1f kHz", hz / 1e3);
    else
        std::snprintf(buf, sizeof(buf), "%.0f Hz", hz);
    return buf;
}

std::string format_si(double value, const char* unit, int significant_digits)
{
    static const struct { double scale; const char* prefix; } kPrefixes[] = {
        {1e9, "G"}, {1e6, "M"}, {1e3, "k"}, {1.0, ""}, {1e-3, "m"}, {1e-6, "u"}, {1e-9, "n"}};

    const double a      = std::fabs(value);
    double       scale  = 1.0;
    const char*  prefix = "";
    if (a > 0.0) {
        scale  = kPrefixes[std::size(kPrefixes) - 1].scale;
        prefix = kPrefixes[std::size(kPrefixes) - 1].prefix;
        for (const auto& p : kPrefixes) {
            if (a >= p.scale * 0.9999995) {
                scale  = p.scale;
                prefix = p.prefix;
                break;
            }
        }
    }

    const double scaled   = value / scale;
    const double mag      = std::fabs(scaled);
    const int    int_digs = mag >= 100.0 ? 3 : mag >= 10.0 ? 2 : 1;
    const int    decimals = std::max(0, significant_digits - int_digs);

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f %s%s", decimals, scaled, prefix, unit);
    return buf;
}

Ticks nice_ticks(double lo, double hi, int target_count)
{
    Ticks ticks;
    if (!(hi > lo) || target_count < 1 || !std::isfinite(lo) || !std::isfinite(hi))
        return ticks;

    const double raw_step = (hi - lo) / target_count;
    const double exponent = std::floor(std::log10(raw_step));
    const double base     = std::pow(10.0, exponent);
    const double fraction = raw_step / base;  // in [1, 10)

    // The comparisons are slightly tolerant: fraction is the result of a
    // floating-point division and may land a hair above a round value.
    constexpr double kSlack = 1.0 + 1e-9;
    double nice;
    int    extra_decimals = 0;
    if (fraction <= 1.0 * kSlack)       nice = 1.0;
    else if (fraction <= 2.0 * kSlack)  nice = 2.0;
    else if (fraction <= 2.5 * kSlack) { nice = 2.5; extra_decimals = 1; }
    else if (fraction <= 5.0 * kSlack)  nice = 5.0;
    else                                nice = 10.0;

    ticks.step = nice * base;
    // Guard against values that sit a rounding error away from a tick.
    const double eps = ticks.step * 1e-9;
    ticks.first = std::ceil((lo - eps) / ticks.step) * ticks.step;
    ticks.count = static_cast<int>(std::floor((hi + eps - ticks.first) / ticks.step)) + 1;
    if (ticks.count < 0)
        ticks.count = 0;

    const int exp10 = static_cast<int>(exponent) + (nice >= 10.0 ? 1 : 0);
    ticks.decimals  = std::max(0, -exp10 + (nice >= 10.0 ? 0 : extra_decimals));
    return ticks;
}

}  // namespace spectro
