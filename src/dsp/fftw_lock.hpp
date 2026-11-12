// FFTW's planner (creating and destroying plans) is not thread-safe, only plan
// execution is.  Everything in this program that plans takes this mutex.
#pragma once

#include <mutex>

namespace spectro {

std::mutex& fftw_planner_mutex();

}  // namespace spectro
