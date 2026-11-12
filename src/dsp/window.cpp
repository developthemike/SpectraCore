#include "dsp/window.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numbers>
#include <string>

namespace spectro {

namespace {

struct WindowDef {
    const char* name;
    const char* key;      // name accepted on the command line
    double      a[5];     // w[n] = sum_k (-1)^k a[k] cos(2 pi k n / N)
};

constexpr WindowDef kWindows[static_cast<int>(WindowType::Count)] = {
    {"Rectangular",     "rect",     {1.0, 0.0, 0.0, 0.0, 0.0}},
    {"Hann",            "hann",     {0.5, 0.5, 0.0, 0.0, 0.0}},
    {"Hamming",         "hamming",  {0.54, 0.46, 0.0, 0.0, 0.0}},
    {"Blackman-Harris", "bh",       {0.35875, 0.48829, 0.14128, 0.01168, 0.0}},
    {"Flat top",        "flattop",  {0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368}},
};

const WindowDef& def(WindowType type)
{
    const int i = std::clamp(static_cast<int>(type), 0, static_cast<int>(WindowType::Count) - 1);
    return kWindows[i];
}

}  // namespace

const char* window_name(WindowType type) { return def(type).name; }

bool window_from_name(std::string_view name, WindowType& type)
{
    std::string lower(name);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (int i = 0; i < static_cast<int>(WindowType::Count); ++i) {
        std::string full(kWindows[i].name);
        std::transform(full.begin(), full.end(), full.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lower == kWindows[i].key || lower == full) {
            type = static_cast<WindowType>(i);
            return true;
        }
    }
    if (lower == "rectangular" || lower == "none" || lower == "boxcar") {
        type = WindowType::Rectangular;
        return true;
    }
    if (lower == "blackman-harris" || lower == "blackmanharris" || lower == "blackman") {
        type = WindowType::BlackmanHarris;
        return true;
    }
    return false;
}

void make_window(WindowType type, int n, float* w)
{
    const double* a    = def(type).a;
    const double  step = 2.0 * std::numbers::pi / static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
        const double x = step * i;
        const double v = a[0] - a[1] * std::cos(x) + a[2] * std::cos(2.0 * x) -
                         a[3] * std::cos(3.0 * x) + a[4] * std::cos(4.0 * x);
        w[i] = static_cast<float>(v);
    }
}

double window_enbw_bins(WindowType type)
{
    // For a cosine-sum window: sum(w^2)/N = a0^2 + sum(a_k^2)/2 and sum(w)/N = a0.
    const double* a   = def(type).a;
    double        sq  = a[0] * a[0];
    for (int k = 1; k < 5; ++k)
        sq += 0.5 * a[k] * a[k];
    return sq / (a[0] * a[0]);
}

}  // namespace spectro
