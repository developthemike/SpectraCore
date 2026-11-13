#include "radio/iq_source.hpp"

#include <stdexcept>

#include "radio/file_source.hpp"
#include "radio/sim_source.hpp"

#if SPECTRO_HAVE_BLADERF
#include "radio/bladerf_source.hpp"
#endif
#if SPECTRO_HAVE_RTLSDR
#include "radio/rtlsdr_source.hpp"
#endif

namespace spectro {

bool bladerf_supported()
{
#if SPECTRO_HAVE_BLADERF
    return true;
#else
    return false;
#endif
}

bool rtlsdr_supported()
{
#if SPECTRO_HAVE_RTLSDR
    return true;
#else
    return false;
#endif
}

std::unique_ptr<IqSource> make_source(const SourceConfig& config)
{
    if (config.kind == "bladerf") {
#if SPECTRO_HAVE_BLADERF
        return std::make_unique<BladeRfSource>(config);
#else
        throw std::runtime_error(
            "This build has no bladeRF support (libbladeRF was not found when it was compiled). "
            "Use --source sim or --source file, or install libbladerf-dev and rebuild.");
#endif
    }
    if (config.kind == "rtlsdr") {
#if SPECTRO_HAVE_RTLSDR
        return std::make_unique<RtlSdrSource>(config);
#else
        throw std::runtime_error(
            "This build has no RTL-SDR support (librtlsdr was not found when it was compiled). "
            "Use --source sim or --source file, or install librtlsdr-dev and rebuild.");
#endif
    }
    if (config.kind == "sim")
        return std::make_unique<SimSource>(config);
    if (config.kind == "file")
        return std::make_unique<FileSource>(config);

    throw std::runtime_error("Unknown source '" + config.kind + "' (expected bladerf, rtlsdr, sim or file)");
}

}  // namespace spectro
