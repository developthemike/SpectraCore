#include "gfx/png_writer.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

#include <zlib.h>

namespace spectro {

namespace {

void put_u32(std::vector<uint8_t>& out, uint32_t v)
{
    out.push_back(static_cast<uint8_t>(v >> 24));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v));
}

void put_chunk(std::vector<uint8_t>& out, const char type[4], const uint8_t* data, size_t size)
{
    put_u32(out, static_cast<uint32_t>(size));
    const size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    if (size > 0)
        out.insert(out.end(), data, data + size);
    const uLong crc = crc32(0L, out.data() + start, static_cast<uInt>(out.size() - start));
    put_u32(out, static_cast<uint32_t>(crc));
}

}  // namespace

bool write_png(const std::string& path, const uint8_t* rgb, int width, int height)
{
    if (!rgb || width <= 0 || height <= 0)
        return false;

    // Scanlines, each prefixed with filter type 0 (none).
    const size_t         stride = static_cast<size_t>(width) * 3;
    std::vector<uint8_t> raw((stride + 1) * static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        uint8_t* line = raw.data() + static_cast<size_t>(y) * (stride + 1);
        line[0]       = 0;
        std::memcpy(line + 1, rgb + static_cast<size_t>(y) * stride, stride);
    }

    uLongf               compressed_size = compressBound(static_cast<uLong>(raw.size()));
    std::vector<uint8_t> compressed(compressed_size);
    if (compress2(compressed.data(), &compressed_size, raw.data(), static_cast<uLong>(raw.size()), 6) != Z_OK)
        return false;

    std::vector<uint8_t> png;
    png.reserve(compressed_size + 64);
    static const uint8_t kSignature[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    png.insert(png.end(), kSignature, kSignature + 8);

    std::vector<uint8_t> header;
    put_u32(header, static_cast<uint32_t>(width));
    put_u32(header, static_cast<uint32_t>(height));
    header.push_back(8);  // bit depth
    header.push_back(2);  // colour type: RGB
    header.push_back(0);  // compression
    header.push_back(0);  // filter
    header.push_back(0);  // interlace
    put_chunk(png, "IHDR", header.data(), header.size());
    put_chunk(png, "IDAT", compressed.data(), compressed_size);
    put_chunk(png, "IEND", nullptr, 0);

    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (!file)
        return false;
    const bool ok = std::fwrite(png.data(), 1, png.size(), file) == png.size();
    return std::fclose(file) == 0 && ok;
}

}  // namespace spectro
