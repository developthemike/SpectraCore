// Command-line options.
#pragma once

#include <limits>
#include <string>

#include "dsp/spectrum_engine.hpp"
#include "radio/iq_source.hpp"

namespace spectro {

struct Options {
    SourceConfig   source;
    SpectrumParams spectrum;

    int columns = 2048;  // spectrogram width in bins (power of two)
    int history = 512;   // spectrogram depth in rows

    // Display
    int         width      = 1600;
    int         height     = 900;
    bool        fullscreen = false;
    int         msaa       = 4;
    bool        vsync      = true;
    std::string gpu;       // index or part of the device name; empty = automatic
    std::string platform;  // "", "x11" or "wayland"
    std::string colormap = "turbo";
    std::string view     = "3d";  // "3d", "top" or "front"
    double      db_min   = std::numeric_limits<double>::quiet_NaN();  // NaN = automatic
    double      db_max   = std::numeric_limits<double>::quiet_NaN();
    bool        hide_ui  = false;

    // Diagnostics and automation
    bool        list_gpus        = false;
    bool        validation       = false;  // enable the Vulkan validation layers
    bool        headless         = false;  // no window: stream, process and print statistics
    double      duration         = 0.0;    // seconds to run, 0 = until closed / interrupted
    std::string screenshot_path;           // save a PNG of the window ...
    double      screenshot_after = 3.0;    // ... this many seconds after start
    std::string spectrum_csv;              // headless: write the time-averaged spectrum here
    bool        verbose          = false;
    bool        self_test        = false;  // run a scripted tour of the interface and exit

    bool show_help    = false;
    bool show_version = false;
};

// Parses argv into `options`.  On failure returns false and describes the
// problem in `error`.
bool parse_options(int argc, char** argv, Options& options, std::string& error);

std::string usage_text(const char* program);

}  // namespace spectro
