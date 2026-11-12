#include "gfx/colormap.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace spectro {

namespace {
#include "gfx/colormap_data.inc"
}  // namespace

int colormap_count() { return kColormapCount; }

int colormap_entries() { return kColormapEntries; }

const char* colormap_name(int index)
{
    return kColormapNames[std::clamp(index, 0, kColormapCount - 1)];
}

int colormap_from_key(std::string_view key)
{
    std::string lower(key);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (int i = 0; i < kColormapCount; ++i) {
        if (lower == kColormapKeys[i])
            return i;
    }
    return -1;
}

void colormap_sample(int index, float t, uint8_t rgb[3])
{
    index = std::clamp(index, 0, kColormapCount - 1);
    t     = t >= 0.0f ? std::min(t, 1.0f) : 0.0f;  // also maps NaN to 0
    const int entry = std::clamp(static_cast<int>(std::lround(t * static_cast<float>(kColormapEntries - 1))), 0,
                                 kColormapEntries - 1);
    rgb[0] = kColormapData[index][entry][0];
    rgb[1] = kColormapData[index][entry][1];
    rgb[2] = kColormapData[index][entry][2];
}

void colormap_fill_rgba(uint8_t* pixels)
{
    for (int m = 0; m < kColormapCount; ++m) {
        for (int e = 0; e < kColormapEntries; ++e) {
            uint8_t* p = pixels + (static_cast<size_t>(m) * kColormapEntries + static_cast<size_t>(e)) * 4;
            p[0] = kColormapData[m][e][0];
            p[1] = kColormapData[m][e][1];
            p[2] = kColormapData[m][e][2];
            p[3] = 255;
        }
    }
}

}  // namespace spectro
