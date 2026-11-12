#pragma once

#include "app/options.hpp"

namespace spectro {

// Runs the receive + DSP pipeline without a window and prints stream statistics
// once per second.  Returns the process exit code.
int run_headless(const Options& options);

}  // namespace spectro
