// A list of P25 control channels to look for a trunked system on, read from a
// JSON file (--control-channels):
//
//   {"control_channels": [153.3375, "153.4375 MHz", 154237500]}
//
// - A frequency is a number or a string.  Below 1 000 000 it is taken as MHz,
//   otherwise as Hz; a string may also give its unit ("153.8125M",
//   "153.8125 MHz", "153812.5 kHz").
// - The list may also be the whole file ([153.3375, 153.4375]), or be spread
//   over "systems" as in trunk-recorder's config.json, whose systems of
//   another type than "p25" are left out:
//     {"systems": [{"type": "p25", "control_channels": [154237500, 154662500]}]}
// - The channels are tried in the order given; one given twice counts once.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace spectro {

// Reads the control channels from a JSON text, in Hz.  On failure returns
// false and says why in `error`.
bool parse_control_channels(std::string_view json, std::vector<double>& hz, std::string& error);

// The same from a file; the error names it.
bool read_control_channels(const std::string& path, std::vector<double>& hz, std::string& error);

}  // namespace spectro
