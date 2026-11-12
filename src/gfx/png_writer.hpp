#pragma once

#include <cstdint>
#include <string>

namespace spectro {

// Writes an 8-bit RGB image (rows top to bottom, 3 bytes per pixel) as a PNG.
bool write_png(const std::string& path, const uint8_t* rgb, int width, int height);

}  // namespace spectro
