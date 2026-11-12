#include "app/signals.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>

namespace spectro {

namespace {
std::atomic<bool> g_quit{false};

void on_signal(int)
{
    // First signal: ask for an orderly shutdown.  A second one means that did
    // not get anywhere (e.g. the display is blocking): leave immediately.
    if (g_quit.exchange(true, std::memory_order_relaxed))
        std::_Exit(130);
}
}  // namespace

void install_signal_handlers()
{
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
}

bool quit_requested()
{
    return g_quit.load(std::memory_order_relaxed);
}

void request_quit()
{
    g_quit.store(true, std::memory_order_relaxed);
}

}  // namespace spectro
