#include "audio/control_channels.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>

#include "util/json.hpp"
#include "util/units.hpp"

namespace spectro {

namespace {

constexpr double kMhzBelow = 1e6;   // a frequency without a unit below this is in MHz
constexpr double kLowestHz  = 1e6;  // what a radio here could receive, at the very most
constexpr double kHighestHz = 10e9;
constexpr double kSameHz    = 1.0;  // a channel given twice
constexpr size_t kMaxFileBytes = size_t{4} << 20;

std::string at(const JsonValue& value)
{
    return "line " + std::to_string(value.line) + ": ";
}

bool equals_lowercase(std::string_view text, std::string_view lowercase)
{
    return text.size() == lowercase.size() &&
           std::equal(text.begin(), text.end(), lowercase.begin(),
                      [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) == b; });
}

// One entry of a list, in Hz.
bool frequency_of(const JsonValue& entry, double& hz, std::string& error)
{
    double value     = 0.0;
    bool   unit      = false;  // given with it
    char   given[64] = "";
    if (entry.type == JsonValue::Type::Number) {
        value = entry.number;
        std::snprintf(given, sizeof(given), "%.10g", value);
    } else if (entry.type == JsonValue::Type::String) {
        if (!parse_quantity(entry.string, value)) {
            error = at(entry) + "\"" + entry.string + "\" is not a frequency";
            return false;
        }
        // Anything but the digits of a plain number: a suffix, a unit, an exponent.
        unit = entry.string.find_first_not_of("0123456789.+- \t") != std::string::npos;
        std::snprintf(given, sizeof(given), "\"%.40s\"", entry.string.c_str());
    } else {
        error = at(entry) + "expected a frequency, not " + json_type_name(entry.type);
        return false;
    }
    hz = !unit && std::fabs(value) < kMhzBelow ? value * 1e6 : value;
    if (!(hz >= kLowestHz && hz <= kHighestHz)) {
        error = at(entry) + given + " would be " + format_frequency(hz) +
                ": give MHz (153.8125), Hz (153812500) or a unit (\"153.8125 MHz\")";
        return false;
    }
    return true;
}

bool append_list(const JsonValue& list, std::vector<double>& hz, std::string& error)
{
    if (list.type != JsonValue::Type::Array) {
        error = at(list) + "\"control_channels\" must be an array of frequencies, not " + json_type_name(list.type);
        return false;
    }
    for (const JsonValue& entry : list.items) {
        double f = 0.0;
        if (!frequency_of(entry, f, error))
            return false;
        hz.push_back(f);
    }
    return true;
}

}  // namespace

bool parse_control_channels(std::string_view json, std::vector<double>& hz, std::string& error)
{
    JsonValue root;
    if (!parse_json(json, root, error))
        return false;

    std::vector<double> found;
    int                 other_systems = 0;  // left out for their type
    if (root.type == JsonValue::Type::Array) {
        if (!append_list(root, found, error))
            return false;
    } else if (root.type == JsonValue::Type::Object) {
        const JsonValue* list    = root.find("control_channels");
        const JsonValue* systems = root.find("systems");
        if (!list && !systems) {
            error = "expected \"control_channels\": an array of frequencies";
            return false;
        }
        if (list && !append_list(*list, found, error))
            return false;
        if (systems) {
            if (systems->type != JsonValue::Type::Array) {
                error = at(*systems) + "\"systems\" must be an array, not " + json_type_name(systems->type);
                return false;
            }
            for (const JsonValue& system : systems->items) {
                if (system.type != JsonValue::Type::Object) {
                    error = at(system) + "a system must be an object, not " + json_type_name(system.type);
                    return false;
                }
                // trunk-recorder: "p25", "smartnet", "conventional", ...
                const JsonValue* type = system.find("type");
                if (type && !(type->type == JsonValue::Type::String && equals_lowercase(type->string, "p25"))) {
                    ++other_systems;
                    continue;
                }
                if (const JsonValue* channels = system.find("control_channels"))
                    if (!append_list(*channels, found, error))
                        return false;
            }
        }
    } else {
        error = std::string("expected an object or an array, not ") + json_type_name(root.type);
        return false;
    }

    // Each once, in the order first given.
    hz.clear();
    for (const double f : found) {
        if (std::none_of(hz.begin(), hz.end(), [f](double g) { return std::fabs(f - g) < kSameHz; }))
            hz.push_back(f);
    }
    if (hz.empty()) {
        error = other_systems > 0 ? "no P25 control channels in it (" + std::to_string(other_systems) +
                                        " system(s) of another type left out)"
                                  : "no control channels in it";
        return false;
    }
    return true;
}

bool read_control_channels(const std::string& path, std::vector<double>& hz, std::string& error)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    std::string text;
    text.reserve(4096);
    char buffer[4096];
    while (in.read(buffer, sizeof(buffer)) || in.gcount() > 0) {
        text.append(buffer, static_cast<size_t>(in.gcount()));
        if (text.size() > kMaxFileBytes) {
            error = path + " is too large for a list of control channels";
            return false;
        }
    }
    if (in.bad()) {
        error = "cannot read " + path;
        return false;
    }
    if (!parse_control_channels(text, hz, error)) {
        error = path + ": " + error;
        return false;
    }
    return true;
}

}  // namespace spectro
