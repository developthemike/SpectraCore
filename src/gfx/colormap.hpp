// Colour maps for the level scale.
#pragma once

#include <cstdint>
#include <string_view>

namespace spectro {

int         colormap_count();
int         colormap_entries();          // colours per map
const char* colormap_name(int index);    // for the user interface
// Index of the map with the given command-line key ("turbo", "inferno", ...), or -1.
int         colormap_from_key(std::string_view key);

// One colour of a map; t is clamped to [0, 1].
void colormap_sample(int index, float t, uint8_t rgb[3]);

// Writes every map as rows of an RGBA8 image, colormap_entries() wide and
// colormap_count() high.
void colormap_fill_rgba(uint8_t* pixels);

}  // namespace spectro
