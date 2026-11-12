// FFT window functions.
#pragma once

#include <string_view>

namespace spectro {

enum class WindowType : int {
    Rectangular = 0,
    Hann,
    Hamming,
    BlackmanHarris,  // 4-term, -92 dB side lobes
    FlatTop,         // 5-term, accurate amplitude readout
    Count
};

const char* window_name(WindowType type);
bool        window_from_name(std::string_view name, WindowType& type);

// Fills `w` with n coefficients of the periodic ("DFT-even") form of the window.
void make_window(WindowType type, int n, float* w);

// Equivalent noise bandwidth of the window, in FFT bins.
double window_enbw_bins(WindowType type);

// Every window above is a sum of at most five cosines, so its spectrum is
// non-zero in at most nine bins: DC and four on either side.
inline constexpr int kWindowSpectrumHalfWidth = 4;

}  // namespace spectro
