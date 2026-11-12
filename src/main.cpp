#include <cstdio>
#include <exception>
#include <string>

#include "app/headless.hpp"
#include "app/options.hpp"
#include "app/signals.hpp"
#include "util/log.hpp"

#if SPECTRO_HAVE_GUI
#include "app/app.hpp"
#endif

int main(int argc, char** argv)
{
    using namespace spectro;

    Options     options;
    std::string error;
    if (!parse_options(argc, argv, options, error)) {
        std::fprintf(stderr, "spectrogram: %s\n", error.c_str());
        return 2;
    }
    if (options.show_help) {
        std::fputs(usage_text(argv[0]).c_str(), stdout);
        return 0;
    }
    if (options.show_version) {
        std::printf("spectrogram %s\n", SPECTRO_VERSION);
        return 0;
    }
    if (options.verbose)
        log_set_level(LogLevel::Debug);

    install_signal_handlers();

    try {
#if SPECTRO_HAVE_GUI
        if (options.list_gpus)
            return run_list_gpus(options);
        if (!options.headless)
            return run_gui(options);
#else
        if (options.list_gpus) {
            std::fprintf(stderr, "spectrogram: this build has no graphics support\n");
            return 2;
        }
        if (!options.headless)
            LOG_WARN("This build has no graphics support; running in --headless mode");
#endif
        return run_headless(options);
    } catch (const std::exception& e) {
        LOG_ERROR("%s", e.what());
        return 1;
    }
}
