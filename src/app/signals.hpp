// Ctrl-C / SIGTERM handling: request an orderly shutdown so the radio is always
// stopped and closed properly.  A second signal exits immediately.
#pragma once

namespace spectro {

void install_signal_handlers();
bool quit_requested();
void request_quit();

}  // namespace spectro
