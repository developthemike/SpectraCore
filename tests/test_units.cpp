#include <string>
#include <vector>

#include "app/options.hpp"
#include "test_framework.hpp"
#include "util/units.hpp"

using namespace spectro;

TEST(parse_quantity_accepts_suffixes)
{
    double v = 0;
    CHECK(parse_quantity("2.437G", v));   CHECK_NEAR(v, 2.437e9, 1e-3);
    CHECK(parse_quantity("20M", v));      CHECK_NEAR(v, 20e6, 1e-6);
    CHECK(parse_quantity("915e6", v));    CHECK_NEAR(v, 915e6, 1e-6);
    CHECK(parse_quantity("100 kHz", v));  CHECK_NEAR(v, 100e3, 1e-9);
    CHECK(parse_quantity(" 20Msps ", v)); CHECK_NEAR(v, 20e6, 1e-6);
    CHECK(parse_quantity("-12.5", v));    CHECK_NEAR(v, -12.5, 1e-12);
    CHECK(parse_quantity("2437MHz", v));  CHECK_NEAR(v, 2.437e9, 1e-3);
}

TEST(parse_quantity_rejects_garbage)
{
    double v = 0;
    CHECK(!parse_quantity("", v));
    CHECK(!parse_quantity("abc", v));
    CHECK(!parse_quantity("20Q", v));
    CHECK(!parse_quantity("20M junk", v));
    CHECK(!parse_quantity("nan", v));
}

TEST(format_frequency_picks_a_unit)
{
    CHECK(format_frequency(2.437e9) == "2.437000 GHz");
    CHECK(format_frequency(915e6) == "915.000 MHz");
    CHECK(format_frequency(100e3) == "100.0 kHz");
    CHECK(format_si(20e6, "S/s") == "20.00 MS/s");
    CHECK(format_si(4882.8, "Hz") == "4.883 kHz");
    CHECK(format_si(0.0, "Hz") == "0.000 Hz");
}

TEST(nice_ticks_cover_the_range)
{
    // A 20 MHz span around 2.437 GHz, in MHz.
    const Ticks t = nice_ticks(2427.0, 2447.0, 10);
    CHECK_NEAR(t.step, 2.0, 1e-9);
    CHECK_NEAR(t.first, 2428.0, 1e-9);
    CHECK_EQ(t.count, 10);
    CHECK_EQ(t.decimals, 0);
    CHECK_NEAR(t.value(t.count - 1), 2446.0, 1e-9);

    const Ticks d = nice_ticks(-100.0, 0.0, 5);
    CHECK_NEAR(d.step, 20.0, 1e-9);
    CHECK_NEAR(d.first, -100.0, 1e-9);
    CHECK_EQ(d.count, 6);

    const Ticks f = nice_ticks(0.0, 1.0, 4);
    CHECK_NEAR(f.step, 0.25, 1e-12);
    CHECK_EQ(f.count, 5);
    CHECK_EQ(f.decimals, 2);

    const Ticks none = nice_ticks(5.0, 5.0, 4);
    CHECK_EQ(none.count, 0);
}

namespace {
bool parse(std::vector<std::string> args, Options& options, std::string& error)
{
    std::vector<char*> argv;
    static std::string program = "spectrogram";
    argv.push_back(program.data());
    for (auto& a : args)
        argv.push_back(a.data());
    return parse_options(static_cast<int>(argv.size()), argv.data(), options, error);
}
}  // namespace

TEST(options_defaults_match_the_brief)
{
    Options     o;
    std::string error;
    CHECK(parse({}, o, error));
    CHECK(o.source.kind == "bladerf");
    CHECK_NEAR(o.source.sample_rate, 20e6, 1e-6);
    CHECK_EQ(o.spectrum.fft_size, 4096);
    CHECK(!o.headless);
}

TEST(options_parse_values)
{
    Options     o;
    std::string error;
    CHECK(parse({"--freq", "915M", "--rate=10M", "--gain", "42", "--fft", "8192", "--window", "bh",
                 "--detector", "peak", "--size", "1280x720", "--headless", "--duration", "5"},
                o, error));
    CHECK_NEAR(o.source.frequency, 915e6, 1e-6);
    CHECK_NEAR(o.source.sample_rate, 10e6, 1e-6);
    CHECK_NEAR(o.source.gain, 42.0, 1e-9);
    CHECK_EQ(o.spectrum.fft_size, 8192);
    CHECK(o.spectrum.window == WindowType::BlackmanHarris);
    CHECK(o.spectrum.detector == Detector::Peak);
    CHECK_EQ(o.width, 1280);
    CHECK_EQ(o.height, 720);
    CHECK(o.headless);
    CHECK_NEAR(o.duration, 5.0, 1e-12);
}

TEST(options_rtlsdr_gets_its_own_defaults)
{
    Options     o;
    std::string error;
    CHECK(parse({"--source", "rtlsdr"}, o, error));
    CHECK(o.source.kind == "rtlsdr");
    CHECK_NEAR(o.source.frequency, kRtlSdrDefaultFrequency, 1e-6);
    CHECK_NEAR(o.source.sample_rate, kRtlSdrDefaultSampleRate, 1e-6);

    // Explicit values win, whatever the order of the arguments.
    Options explicit_values;
    CHECK(parse({"--freq", "433.92M", "--rate=1M", "--ppm", "-12", "--source=rtlsdr"}, explicit_values, error));
    CHECK_NEAR(explicit_values.source.frequency, 433.92e6, 1e-6);
    CHECK_NEAR(explicit_values.source.sample_rate, 1e6, 1e-6);
    CHECK_EQ(explicit_values.source.ppm, -12);

    Options bad;
    CHECK(!parse({"--ppm", "5000"}, bad, error));
}

TEST(options_reject_bad_input)
{
    Options     o;
    std::string error;
    CHECK(!parse({"--fft", "1000"}, o, error));
    CHECK(!parse({"--bogus"}, o, error));
    CHECK(!parse({"--freq"}, o, error));
    CHECK(!parse({"--source", "hackrf"}, o, error));
    CHECK(!parse({"--source", "file"}, o, error));
    CHECK(!parse({"--headless=1"}, o, error));
    CHECK(!parse({"--bins", "1000"}, o, error));

    Options with_file;
    CHECK(parse({"--file", "capture.cf32"}, with_file, error));
    CHECK(with_file.source.kind == "file");
}
