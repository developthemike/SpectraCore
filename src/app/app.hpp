#pragma once

#include "app/options.hpp"

namespace spectro {

// Runs the interactive application.  Returns the process exit code; throws
// std::runtime_error if the radio or the display cannot be set up.
int run_gui(const Options& options);

// Prints the Vulkan devices on this machine, numbered as --gpu counts them.
int run_list_gpus(const Options& options);

}  // namespace spectro
