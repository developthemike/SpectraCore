#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "app/options.hpp"
#include "audio/control_channels.hpp"
#include "test_framework.hpp"
#include "util/json.hpp"

using namespace spectro;

namespace {

// What is wrong with a JSON text; empty if nothing.
std::string json_error(std::string_view text)
{
    JsonValue   value;
    std::string error;
    return parse_json(text, value, error) ? std::string() : error;
}

// What is wrong with a list of control channels; empty if nothing.
std::string list_error(std::string_view json)
{
    std::vector<double> hz;
    std::string         error;
    return parse_control_channels(json, hz, error) ? std::string() : error;
}

// A file with this text in it, removed on destruction.
struct TempFile {
    std::filesystem::path path;

    explicit TempFile(std::string_view text)
    {
        static int counter = 0;
        path = std::filesystem::temp_directory_path() /
               ("spectro_test_" + std::to_string(static_cast<long long>(
                    std::chrono::steady_clock::now().time_since_epoch().count())) +
                "_" + std::to_string(counter++) + ".json");
        std::ofstream out(path, std::ios::binary);
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
    }
    ~TempFile()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

bool parse_args(std::vector<std::string> args, Options& options, std::string& error)
{
    args.insert(args.begin(), "spectrogram");
    std::vector<char*> argv;
    for (auto& a : args)
        argv.push_back(a.data());
    return parse_options(static_cast<int>(argv.size()), argv.data(), options, error);
}

}  // namespace

TEST(json_reads_every_kind_of_value)
{
    JsonValue   v;
    std::string error;
    CHECK(parse_json(R"({"a": [1, -2.5e3, 0], "b": "q\"b\\s\/n\n\u00e9\ud83d\udce1", "c": true, "d": false,
                         "e": null, "f": {}, "g": []})",
                     v, error));
    CHECK(v.type == JsonValue::Type::Object);
    CHECK_EQ(v.members.size(), size_t{7});
    CHECK(v.members.size() == 7 && v.members[0].first == "a" && v.members[6].first == "g");  // in the order written
    const JsonValue* a = v.find("a");
    CHECK(a && a->type == JsonValue::Type::Array && a->items.size() == 3);
    if (a && a->items.size() == 3) {
        CHECK_NEAR(a->items[0].number, 1.0, 0.0);
        CHECK_NEAR(a->items[1].number, -2500.0, 0.0);
        CHECK(a->items[2].type == JsonValue::Type::Number);
    }
    const JsonValue* b = v.find("b");
    CHECK(b && b->string == "q\"b\\s/n\n\xC3\xA9\xF0\x9F\x93\xA1");  // UTF-8, surrogate pair joined
    CHECK(v.find("c") && v.find("c")->type == JsonValue::Type::Boolean && v.find("c")->boolean);
    CHECK(v.find("d") && v.find("d")->type == JsonValue::Type::Boolean && !v.find("d")->boolean);
    CHECK(v.find("e") && v.find("e")->type == JsonValue::Type::Null);
    CHECK(v.find("f") && v.find("f")->type == JsonValue::Type::Object && v.find("f")->members.empty());
    CHECK(v.find("g") && v.find("g")->type == JsonValue::Type::Array && v.find("g")->items.empty());
    CHECK(!v.find("h"));

    // A byte order mark and white space of every kind; each value knows its line.
    CHECK(parse_json("\xEF\xBB\xBF\n[\r\n\t1,\n  \"two\"\n]\n", v, error));
    CHECK(v.type == JsonValue::Type::Array);
    CHECK_EQ(v.line, 2);
    CHECK(v.items.size() == 2 && v.items[0].line == 3 && v.items[1].line == 4 && v.items[1].string == "two");

    // As deep as 64 arrays.
    CHECK(json_error(std::string(64, '[') + std::string(64, ']')).empty());
}

TEST(json_says_what_is_wrong_and_where)
{
    CHECK(json_error("[1, 2,]") == "line 1: a comma before ']': JSON allows no trailing comma");
    CHECK(json_error("{\"a\": 1,\n}") == "line 2: a comma before '}': JSON allows no trailing comma");
    CHECK(json_error("[1\n 2]") == "line 2: expected ',' or ']'");
    CHECK(json_error("{\"a\": 1 \"b\": 2}") == "line 1: expected ',' or '}'");
    CHECK(json_error("{\"a\" 1}") == "line 1: expected ':' after the key");
    CHECK(json_error("{a: 1}") == "line 1: expected a key (a string in double quotes)");
    CHECK(json_error("{\"a\": 1, \"a\": 2}") == "line 1: the key \"a\" is given twice");
    CHECK(json_error("[1,\n// two\n2]") == "line 2: comments are not allowed in JSON");
    CHECK(json_error("[01]") == "line 1: invalid number");
    CHECK(json_error("[1.]") == "line 1: invalid number");
    CHECK(json_error("[1.2.3]") == "line 1: invalid number");
    CHECK(json_error("[-]") == "line 1: invalid number");
    CHECK(json_error("[1e]") == "line 1: invalid number");
    CHECK(json_error("[1MHz]") == "line 1: invalid number");
    CHECK(json_error("[.5]") == "line 1: expected a value");
    CHECK(json_error("[1e999]") == "line 1: number out of range");
    CHECK(json_error("[True]") == "line 1: expected a value");
    CHECK(json_error("[truex]") == "line 1: expected a value");
    CHECK(json_error("['a']") == "line 1: expected a value");
    CHECK(json_error("\"abc") == "line 1: unterminated string");
    CHECK(json_error("[\"a\nb\"]") == "line 1: unterminated string");
    CHECK(json_error("[\"a\tb\"]") == "line 1: control character in a string (write it as an escape, such as \\t)");
    CHECK(json_error("[\"\\x\"]") == "line 1: invalid escape in a string");
    CHECK(json_error("[\"\\u12\"]") == "line 1: invalid \\u escape in a string");
    CHECK(json_error("[\"\\udc00\"]") == "line 1: invalid \\u escape in a string");
    CHECK(json_error("[\"\\ud83d\"]") == "line 1: invalid \\u escape in a string");
    CHECK(json_error("") == "line 1: the text ends where a value should be");
    CHECK(json_error("[1,") == "line 1: the text ends where a value should be");
    CHECK(json_error("{\"a\": 1} x") == "line 1: expected nothing more after the value");
    CHECK(json_error(std::string(65, '[') + std::string(65, ']')) == "line 1: nested too deeply");
    CHECK(json_error(std::string(100000, '[')) == "line 1: nested too deeply");  // and no stack overflow
}

TEST(control_channels_are_read_in_mhz_hz_or_with_a_unit)
{
    std::vector<double> hz;
    std::string         error;
    CHECK(parse_control_channels(R"({"control_channels": [153.3375, "153.4375 MHz", 154237500, "154.6625M",
                                                          "153.8125", "153337.5 kHz", 153.3375, "1.541625e8"]})",
                                 hz, error));
    const std::vector<double> expected = {153.3375e6, 153.4375e6, 154.2375e6, 154.6625e6, 153.8125e6, 154.1625e6};
    CHECK_EQ(hz.size(), expected.size());  // in the order given, those given twice once
    for (size_t i = 0; i < std::min(hz.size(), expected.size()); ++i)
        CHECK_NEAR(hz[i], expected[i], 1e-3);

    // The file can be the list itself.
    CHECK(parse_control_channels("[155.475, 155.7]", hz, error));
    CHECK(hz.size() == 2 && std::fabs(hz[1] - 155.7e6) < 1e-3);
}

TEST(control_channels_are_read_from_a_trunk_recorder_config)
{
    // Its P25 systems' control channels, and only theirs.
    const char* const config = R"({
        "ver": 2,
        "sources": [{"center": 154000000, "rate": 2400000, "driver": "osmosdr"}],
        "systems": [
            {"shortName": "county", "type": "p25", "control_channels": [154237500, 153337500], "modulation": "qpsk"},
            {"shortName": "city", "type": "smartnet", "control_channels": [154162500]},
            {"shortName": "fire", "type": "conventional", "channels": [460025000]},
            {"shortName": "state", "type": "P25", "control_channels": [152312500, 154237500]}
        ]
    })";
    std::vector<double> hz;
    std::string         error;
    CHECK(parse_control_channels(config, hz, error));
    CHECK_EQ(hz.size(), size_t{3});
    if (hz.size() == 3) {
        CHECK_NEAR(hz[0], 154.2375e6, 1e-3);
        CHECK_NEAR(hz[1], 153.3375e6, 1e-3);
        CHECK_NEAR(hz[2], 152.3125e6, 1e-3);
    }
    CHECK(list_error(R"({"systems": [{"type": "smartnet", "control_channels": [154162500]}]})") ==
          "no P25 control channels in it (1 system(s) of another type left out)");
}

TEST(control_channels_say_what_is_wrong)
{
    CHECK(list_error("{\"control_channels\": [153.3375,]}") == "line 1: a comma before ']': JSON allows no trailing comma");
    CHECK(list_error("{}") == "expected \"control_channels\": an array of frequencies");
    CHECK(list_error("153.3375") == "expected an object or an array, not a number");
    CHECK(list_error("{\"control_channels\": 153.3375}") ==
          "line 1: \"control_channels\" must be an array of frequencies, not a number");
    CHECK(list_error("{\"control_channels\": []}") == "no control channels in it");
    CHECK(list_error("{\"control_channels\": [\n  153.3375,\n  true\n]}") == "line 3: expected a frequency, not true or false");
    CHECK(list_error("[{\"frequency\": 153.3375}]") == "line 1: expected a frequency, not an object");
    CHECK(list_error("[\"153.3375 MHzz\"]") == "line 1: \"153.3375 MHzz\" is not a frequency");
    CHECK(list_error("[153812.5]") ==
          "line 1: 153812.5 would be 153.812500 GHz: give MHz (153.8125), Hz (153812500) or a unit (\"153.8125 MHz\")");
    CHECK(list_error("[\"153.8125 Hz\"]") ==
          "line 1: \"153.8125 Hz\" would be 154 Hz: give MHz (153.8125), Hz (153812500) or a unit (\"153.8125 MHz\")");
    CHECK(!list_error("[-153.3375]").empty());
    CHECK(list_error("{\"systems\": {}}") == "line 1: \"systems\" must be an array, not an object");
    CHECK(list_error("{\"systems\": [153.3375]}") == "line 1: a system must be an object, not a number");
}

TEST(control_channels_from_a_file_and_the_command_line)
{
    const TempFile      good("{\"control_channels\": [153.3375, 153.4375]}\n");
    std::vector<double> hz;
    std::string         error;
    CHECK(read_control_channels(good.path.string(), hz, error));
    CHECK_EQ(hz.size(), size_t{2});

    const TempFile bad("{\"control_channels\": [153.3375\n  153.4375]}\n");
    CHECK(!read_control_channels(bad.path.string(), hz, error));
    CHECK(error == bad.path.string() + ": line 2: expected ',' or ']'");
    const std::string missing = good.path.string() + ".missing";
    CHECK(!read_control_channels(missing, hz, error));
    CHECK(error == "cannot open " + missing);

    Options o;
    CHECK(o.control_channels.empty());
    CHECK(parse_args({"--p25-only", "--scan", "--control-channels", good.path.string()}, o, error));
    CHECK_EQ(o.control_channels.size(), size_t{2});
    if (o.control_channels.size() == 2)
        CHECK_NEAR(o.control_channels[1], 153.4375e6, 1e-3);
    Options other;
    CHECK(!parse_args({"--control-channels", missing}, other, error));
    CHECK(error == "--control-channels: cannot open " + missing);
}
