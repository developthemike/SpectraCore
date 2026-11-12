// Parsing and formatting of engineering quantities, and axis tick selection.
#pragma once

#include <string>
#include <string_view>

namespace spectro {

// Parses a number with an optional SI suffix and unit: "2.437G", "20M", "915e6",
// "100 kHz", "20Msps".  Returns false if the text is not a valid quantity.
bool parse_quantity(std::string_view text, double& value);

// "2.437000 GHz", "915.000 MHz", "100.0 kHz", "12 Hz"
std::string format_frequency(double hz);

// Value with SI prefix and unit: format_si(2.0e7, "S/s") -> "20.00 MS/s"
std::string format_si(double value, const char* unit, int significant_digits = 4);

// Evenly spaced "nice" axis ticks (multiples of 1, 2 or 5 x 10^k) covering [lo, hi].
struct Ticks {
    double first    = 0.0;  // value of the first tick (>= lo)
    double step     = 1.0;  // distance between ticks (> 0)
    int    count    = 0;    // number of ticks
    int    decimals = 0;    // decimal places needed to print the tick values exactly

    double value(int i) const { return first + step * i; }
};
Ticks nice_ticks(double lo, double hi, int target_count);

}  // namespace spectro
