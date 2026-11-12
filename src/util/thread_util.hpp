// Small per-thread helpers that have no portable standard-library equivalent.
#pragma once

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#include <time.h>
#endif

namespace spectro {

// Names the calling thread for debuggers and `top -H` (at most 15 characters).
inline void set_current_thread_name(const char* name)
{
#if defined(__linux__)
    pthread_setname_np(pthread_self(), name);
#elif defined(__APPLE__)
    pthread_setname_np(name);
#else
    (void)name;
#endif
}

// CPU time consumed so far by the calling thread, in seconds (0 if unsupported).
inline double current_thread_cpu_seconds()
{
#if defined(__linux__) || defined(__APPLE__)
    timespec ts{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0)
        return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
#endif
    return 0.0;
}

}  // namespace spectro
