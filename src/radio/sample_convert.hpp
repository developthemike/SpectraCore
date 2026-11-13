// Conversions from other sample formats to the pipeline's SC16 Q11.
#pragma once

#include <cstddef>
#include <cstdint>

namespace spectro {

// Unsigned 8-bit samples with a 127.5 offset, as produced by the RTL2832
// (rtl_sdr recordings, RTL-SDR dongles).  Maps 0...255 to -2040...+2040, so
// that the converter's full scale lands at the Q11 full scale.
inline void cu8_to_q11(const uint8_t* src, int16_t* dst, size_t values)
{
    for (size_t i = 0; i < values; ++i)
        dst[i] = static_cast<int16_t>((2 * static_cast<int>(src[i]) - 255) * 8);
}

}  // namespace spectro
