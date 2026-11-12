#include "app/options.hpp"

#include <bit>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string_view>
#include <vector>

#include "util/units.hpp"

namespace spectro {

namespace {

struct Flag {
    const char* name;      // "--freq"
    const char* argument;  // placeholder shown in the help text; nullptr for switches
    const char* help;
    std::function<bool(const std::string& value, std::string& error)> apply;
};

bool parse_number(const std::string& text, double& value, std::string& error, const char* what)
{
    if (!parse_quantity(text, value)) {
        error = std::string("'") + text + "' is not a valid " + what;
        return false;
    }
    return true;
}

bool parse_int(const std::string& text, int& value, std::string& error, const char* what)
{
    double v = 0.0;
    if (!parse_number(text, v, error, what))
        return false;
    if (v != std::floor(v) || std::fabs(v) > 1e9) {
        error = std::string("'") + text + "' is not a valid " + what;
        return false;
    }
    value = static_cast<int>(v);
    return true;
}

bool is_power_of_two(int v) { return v > 0 && std::has_single_bit(static_cast<unsigned>(v)); }

std::vector<Flag> make_flags(Options& o)
{
    std::vector<Flag> flags;
    auto add = [&flags](const char* name, const char* argument, const char* help,
                        std::function<bool(const std::string&, std::string&)> apply) {
        flags.push_back({name, argument, help, std::move(apply)});
    };

    // ---- Source ----
    add("--source", "KIND", "Sample source: bladerf (default), sim, or file",
        [&o](const std::string& v, std::string& e) {
            if (v != "bladerf" && v != "sim" && v != "file") {
                e = "unknown source '" + v + "' (expected bladerf, sim or file)";
                return false;
            }
            o.source.kind = v;
            return true;
        });
    add("--freq", "HZ", "Centre frequency, e.g. 2.437G or 915M (default 2.437G)",
        [&o](const std::string& v, std::string& e) { return parse_number(v, o.source.frequency, e, "frequency"); });
    add("--rate", "SPS", "Sample rate, e.g. 20M (default 20M)",
        [&o](const std::string& v, std::string& e) {
            if (!parse_number(v, o.source.sample_rate, e, "sample rate"))
                return false;
            if (!(o.source.sample_rate > 0.0)) {
                e = "the sample rate must be positive";
                return false;
            }
            return true;
        });
    add("--bw", "HZ", "Analog filter bandwidth (default: same as the sample rate)",
        [&o](const std::string& v, std::string& e) { return parse_number(v, o.source.bandwidth, e, "bandwidth"); });
    add("--gain", "DB", "Receiver gain in dB (default 30)",
        [&o](const std::string& v, std::string& e) { return parse_number(v, o.source.gain, e, "gain"); });
    add("--agc", nullptr, "Use the radio's automatic gain control instead of a fixed gain",
        [&o](const std::string&, std::string&) { o.source.agc = true; return true; });
    add("--device", "ID", "bladeRF to open, e.g. \"*:serial=f12ce1\" (default: first one found)",
        [&o](const std::string& v, std::string&) { o.source.device = v; return true; });
    add("--channel", "N", "RX channel index on multi-channel boards (default 0)",
        [&o](const std::string& v, std::string& e) { return parse_int(v, o.source.channel, e, "channel"); });
    add("--file", "PATH", "IQ recording to play back (implies --source file)",
        [&o](const std::string& v, std::string&) {
            o.source.file_path = v;
            o.source.kind      = "file";
            return true;
        });
    add("--format", "FMT", "File sample format: auto (by extension, default), sc16, cs16, cf32, cs8, cu8",
        [&o](const std::string& v, std::string&) { o.source.file_format = v; return true; });
    add("--no-loop", nullptr, "Stop at the end of the file instead of repeating it",
        [&o](const std::string&, std::string&) { o.source.file_loop = false; return true; });

    // ---- Spectrum ----
    add("--fft", "N", "FFT size, a power of two from 256 to 262144 (default 4096)",
        [&o](const std::string& v, std::string& e) {
            if (!parse_int(v, o.spectrum.fft_size, e, "FFT size"))
                return false;
            if (!is_power_of_two(o.spectrum.fft_size) || o.spectrum.fft_size < SpectrumEngine::kMinFftSize ||
                o.spectrum.fft_size > SpectrumEngine::kMaxFftSize) {
                e = "the FFT size must be a power of two between 256 and 262144";
                return false;
            }
            return true;
        });
    add("--window", "NAME", "FFT window: hann (default), hamming, bh, flattop, rect",
        [&o](const std::string& v, std::string& e) {
            if (!window_from_name(v, o.spectrum.window)) {
                e = "unknown window '" + v + "'";
                return false;
            }
            return true;
        });
    add("--overlap", "PCT", "FFT overlap: 0, 50 (default) or 75 percent",
        [&o](const std::string& v, std::string& e) {
            if (!parse_int(v, o.spectrum.overlap_pct, e, "overlap"))
                return false;
            if (o.spectrum.overlap_pct != 0 && o.spectrum.overlap_pct != 50 && o.spectrum.overlap_pct != 75) {
                e = "the overlap must be 0, 50 or 75";
                return false;
            }
            return true;
        });
    add("--detector", "NAME", "How FFTs combine into a row: average (default) or peak",
        [&o](const std::string& v, std::string& e) {
            if (v == "average" || v == "avg" || v == "rms")
                o.spectrum.detector = Detector::Average;
            else if (v == "peak" || v == "max")
                o.spectrum.detector = Detector::Peak;
            else {
                e = "unknown detector '" + v + "' (expected average or peak)";
                return false;
            }
            return true;
        });
    add("--rows-per-sec", "N", "Waterfall speed in rows per second (default 50)",
        [&o](const std::string& v, std::string& e) {
            if (!parse_number(v, o.spectrum.rows_per_sec, e, "row rate"))
                return false;
            if (!(o.spectrum.rows_per_sec >= 0.5 && o.spectrum.rows_per_sec <= 2000.0)) {
                e = "the row rate must be between 0.5 and 2000";
                return false;
            }
            return true;
        });
    add("--no-dc-block", nullptr, "Do not remove the receiver's DC offset spike",
        [&o](const std::string&, std::string&) { o.spectrum.dc_block = false; return true; });
    add("--bins", "N", "Spectrogram width in bins, a power of two from 256 to 4096 (default 2048)",
        [&o](const std::string& v, std::string& e) {
            if (!parse_int(v, o.columns, e, "bin count"))
                return false;
            if (!is_power_of_two(o.columns) || o.columns < 256 || o.columns > 4096) {
                e = "--bins must be a power of two between 256 and 4096";
                return false;
            }
            return true;
        });
    add("--history", "ROWS", "Spectrogram depth in rows, 64 to 4096 (default 512)",
        [&o](const std::string& v, std::string& e) {
            if (!parse_int(v, o.history, e, "history depth"))
                return false;
            if (o.history < 64 || o.history > 4096) {
                e = "--history must be between 64 and 4096";
                return false;
            }
            return true;
        });

    // ---- Display ----
    add("--db-min", "DB", "Bottom of the level scale in dBFS (default: automatic)",
        [&o](const std::string& v, std::string& e) { return parse_number(v, o.db_min, e, "level"); });
    add("--db-max", "DB", "Top of the level scale in dBFS (default: automatic)",
        [&o](const std::string& v, std::string& e) { return parse_number(v, o.db_max, e, "level"); });
    add("--colormap", "NAME", "Colour map: turbo (default), inferno, magma, plasma, viridis, classic, gray",
        [&o](const std::string& v, std::string&) { o.colormap = v; return true; });
    add("--view", "NAME", "Initial view: 3d (default), top (2D waterfall) or front (spectrum)",
        [&o](const std::string& v, std::string& e) {
            if (v != "3d" && v != "top" && v != "front") {
                e = "unknown view '" + v + "' (expected 3d, top or front)";
                return false;
            }
            o.view = v;
            return true;
        });
    add("--size", "WxH", "Window size in pixels (default 1600x900)",
        [&o](const std::string& v, std::string& e) {
            const size_t x = v.find_first_of("xX");
            if (x == std::string::npos || !parse_int(v.substr(0, x), o.width, e, "width") ||
                !parse_int(v.substr(x + 1), o.height, e, "height") || o.width < 320 || o.height < 240) {
                e = "--size expects WIDTHxHEIGHT, at least 320x240";
                return false;
            }
            return true;
        });
    add("--fullscreen", nullptr, "Start in full-screen mode",
        [&o](const std::string&, std::string&) { o.fullscreen = true; return true; });
    add("--msaa", "N", "Anti-aliasing samples: 1, 2, 4 (default) or 8",
        [&o](const std::string& v, std::string& e) {
            if (!parse_int(v, o.msaa, e, "sample count"))
                return false;
            if (o.msaa != 1 && o.msaa != 2 && o.msaa != 4 && o.msaa != 8) {
                e = "--msaa must be 1, 2, 4 or 8";
                return false;
            }
            return true;
        });
    add("--no-vsync", nullptr, "Do not synchronise rendering to the display refresh",
        [&o](const std::string&, std::string&) { o.vsync = false; return true; });
    add("--gpu", "SEL", "GPU to render on: an index from --list-gpus or part of its name",
        [&o](const std::string& v, std::string&) { o.gpu = v; return true; });
    add("--list-gpus", nullptr, "List the available Vulkan devices and exit",
        [&o](const std::string&, std::string&) { o.list_gpus = true; return true; });
    add("--x11", nullptr, "Force the X11 windowing backend",
        [&o](const std::string&, std::string&) { o.platform = "x11"; return true; });
    add("--wayland", nullptr, "Force the Wayland windowing backend",
        [&o](const std::string&, std::string&) { o.platform = "wayland"; return true; });
    add("--hide-ui", nullptr, "Start with the control panel hidden (toggle with H)",
        [&o](const std::string&, std::string&) { o.hide_ui = true; return true; });

    // ---- Diagnostics ----
    add("--headless", nullptr, "No window: receive, process and print stream statistics",
        [&o](const std::string&, std::string&) { o.headless = true; return true; });
    add("--duration", "SEC", "Exit after this many seconds (default: run until closed)",
        [&o](const std::string& v, std::string& e) { return parse_number(v, o.duration, e, "duration"); });
    add("--spectrum-csv", "PATH", "With --headless: write the time-averaged spectrum as CSV",
        [&o](const std::string& v, std::string&) { o.spectrum_csv = v; return true; });
    add("--screenshot", "PATH", "Save a PNG screenshot of the window to PATH",
        [&o](const std::string& v, std::string&) { o.screenshot_path = v; return true; });
    add("--screenshot-after", "SEC", "When to take the --screenshot (default 3 s after start)",
        [&o](const std::string& v, std::string& e) { return parse_number(v, o.screenshot_after, e, "time"); });
    add("--self-test", nullptr, "Run a short scripted tour of views, sizes and settings, then exit",
        [&o](const std::string&, std::string&) { o.self_test = true; return true; });
    add("--validate", nullptr, "Enable the Vulkan validation layers (if installed)",
        [&o](const std::string&, std::string&) { o.validation = true; return true; });
    add("--verbose", nullptr, "Print debug messages, including libbladeRF's",
        [&o](const std::string&, std::string&) { o.verbose = true; return true; });
    add("--version", nullptr, "Print the version and exit",
        [&o](const std::string&, std::string&) { o.show_version = true; return true; });
    add("--help", nullptr, "Show this help and exit",
        [&o](const std::string&, std::string&) { o.show_help = true; return true; });

    return flags;
}

}  // namespace

bool parse_options(int argc, char** argv, Options& options, std::string& error)
{
    // A recording's format is detected from its extension unless stated.
    options.source.file_format = "auto";

    const std::vector<Flag> flags = make_flags(options);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h")
            arg = "--help";

        std::string name = arg, value;
        bool        has_inline_value = false;
        if (const size_t eq = arg.find('='); eq != std::string::npos && arg.rfind("--", 0) == 0) {
            name             = arg.substr(0, eq);
            value            = arg.substr(eq + 1);
            has_inline_value = true;
        }

        const Flag* flag = nullptr;
        for (const Flag& f : flags) {
            if (name == f.name) {
                flag = &f;
                break;
            }
        }
        if (!flag) {
            error = "unknown option '" + arg + "' (see --help)";
            return false;
        }

        if (flag->argument) {
            if (!has_inline_value) {
                if (i + 1 >= argc) {
                    error = std::string(flag->name) + " needs a value (" + flag->argument + ")";
                    return false;
                }
                value = argv[++i];
            }
        } else if (has_inline_value) {
            error = std::string(flag->name) + " does not take a value";
            return false;
        }

        std::string detail;
        if (!flag->apply(value, detail)) {
            error = std::string(flag->name) + ": " + detail;
            return false;
        }
    }

    if (options.source.kind == "file" && options.source.file_path.empty()) {
        error = "--source file needs --file <path>";
        return false;
    }
    if (!std::isnan(options.db_min) && !std::isnan(options.db_max) && options.db_max - options.db_min < 5.0) {
        error = "--db-max must be at least 5 dB above --db-min";
        return false;
    }
    options.source.verbose = options.verbose;
    return true;
}

std::string usage_text(const char* program)
{
    Options           scratch;
    const auto        flags = make_flags(scratch);
    const std::string prog  = program ? program : "spectrogram";

    std::string text;
    text += "Real-time 3D RF spectrogram waterfall for the bladeRF.\n\n";
    text += "Usage: " + prog + " [options]\n\n";
    text += "Examples:\n";
    static const struct { const char* arguments; const char* what; } kExamples[] = {
        {"", "bladeRF at 2.437 GHz, 20 MS/s"},
        {"--freq 915M --gain 40", "another band"},
        {"--source sim", "built-in signal simulator, no hardware needed"},
        {"--file capture.sc16 --rate 20M --freq 2.437G", "play back a recording"},
        {"--headless --duration 10", "stream health check without a window"},
    };
    for (const auto& example : kExamples) {
        std::string line = "  " + prog + " " + example.arguments;
        if (line.size() < prog.size() + 30)
            line.append(prog.size() + 30 - line.size(), ' ');
        text += line + "  " + example.what + "\n";
    }
    text += "\n";
    text += "Options:\n";
    for (const Flag& f : flags) {
        std::string left = std::string("  ") + f.name;
        if (f.argument)
            left += std::string(" ") + f.argument;
        if (left.size() < 26)
            left.append(26 - left.size(), ' ');
        else
            left += "\n" + std::string(26, ' ');
        text += left + f.help + "\n";
    }
    text += "\nFrequencies and rates accept k, M and G suffixes. Keyboard and mouse controls are\n"
            "listed in the application under \"Help\".\n";
    return text;
}

}  // namespace spectro
