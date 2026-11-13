cmake_minimum_required(VERSION 3.28)

project(spectrogram
    VERSION 0.1.0
    DESCRIPTION "Real-time 3D RF spectrogram waterfall (Vulkan + bladeRF / RTL-SDR)"
    LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)  # no C++20 modules here; skips a scan of every source file
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SPECTRO_BUILD_GUI     "Build the Vulkan user interface (OFF: headless diagnostics only)" ON)
option(SPECTRO_WITH_BLADERF  "Support bladeRF hardware (OFF: simulator and file playback only)" ON)
option(SPECTRO_WITH_RTLSDR   "Support RTL-SDR dongles, if librtlsdr is installed" ON)
option(SPECTRO_BUILD_TESTS   "Build the unit tests" ON)
option(SPECTRO_NATIVE_ARCH   "Optimise for the CPU of the build machine (-march=native)" ON)
option(SPECTRO_FETCH_MISSING "Download build dependencies that are not installed system-wide" ON)
option(SPECTRO_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

# ---------------------------------------------------------------------------
# Compiler flags shared by all first-party targets
# ---------------------------------------------------------------------------
add_library(spectro_flags INTERFACE)
target_compile_features(spectro_flags INTERFACE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(spectro_flags INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
        # DSP inner loops never inspect errno; this lets the compiler vectorise them.
        -fno-math-errno)
    if(SPECTRO_WARNINGS_AS_ERRORS)
        target_compile_options(spectro_flags INTERFACE -Werror)
    endif()
    if(SPECTRO_NATIVE_ARCH)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-march=native SPECTRO_HAVE_MARCH_NATIVE)
        if(SPECTRO_HAVE_MARCH_NATIVE)
            target_compile_options(spectro_flags INTERFACE -march=native)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Core library: radio sources, DSP and the pipeline that connects them.
# No graphics dependencies.
# ---------------------------------------------------------------------------
add_library(spectro_core STATIC
    src/util/log.cpp
    src/util/units.cpp
    src/dsp/window.cpp
    src/dsp/spectrum_engine.cpp
    src/radio/iq_source.cpp
    src/radio/sim_source.cpp
    src/radio/file_source.cpp
    src/radio/acquisition.cpp
    src/app/options.cpp
    src/app/pipeline.cpp)
target_include_directories(spectro_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(spectro_core
    PUBLIC  spectro_flags Threads::Threads
    PRIVATE PkgConfig::FFTW3F)
if(SPECTRO_HAVE_BLADERF)
    target_sources(spectro_core PRIVATE src/radio/bladerf_source.cpp)
    target_link_libraries(spectro_core PRIVATE spectro_bladerf)
    target_compile_definitions(spectro_core PUBLIC SPECTRO_HAVE_BLADERF=1)
else()
    target_compile_definitions(spectro_core PUBLIC SPECTRO_HAVE_BLADERF=0)
endif()

if(SPECTRO_HAVE_RTLSDR)
    target_sources(spectro_core PRIVATE src/radio/rtlsdr_source.cpp)
    target_link_libraries(spectro_core PRIVATE spectro_rtlsdr)
    target_compile_definitions(spectro_core PUBLIC SPECTRO_HAVE_RTLSDR=1)
else()
    target_compile_definitions(spectro_core PUBLIC SPECTRO_HAVE_RTLSDR=0)
endif()

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
add_executable(spectrogram
    src/main.cpp
    src/app/signals.cpp
    src/app/headless.cpp)
target_link_libraries(spectrogram PRIVATE spectro_core spectro_flags)
target_compile_definitions(spectrogram PRIVATE SPECTRO_VERSION="${PROJECT_VERSION}")

if(SPECTRO_BUILD_GUI)
    include(Shaders)
    target_sources(spectrogram PRIVATE
        src/app/app.cpp
        src/gfx/vk_context.cpp
        src/gfx/renderer.cpp
        src/gfx/colormap.cpp
        src/gfx/camera.cpp
        src/gfx/png_writer.cpp
        src/ui/ui.cpp
        src/ui/axes.cpp)
    target_link_libraries(spectrogram PRIVATE
        spectro_vulkan spectro_glfw spectro_imgui ZLIB::ZLIB)
    target_compile_definitions(spectrogram PRIVATE SPECTRO_HAVE_GUI=1 GLFW_INCLUDE_NONE)
    spectro_add_shaders(spectrogram
        INCLUDES shaders/common.glsl
        SOURCES
            shaders/surface.vert
            shaders/surface.frag
            shaders/line.vert
            shaders/line.frag)
else()
    target_compile_definitions(spectrogram PRIVATE SPECTRO_HAVE_GUI=0)
endif()

install(TARGETS spectrogram RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(SPECTRO_BUILD_TESTS)
    enable_testing()
    add_executable(spectro_tests
        tests/test_main.cpp
        tests/test_ring.cpp
        tests/test_units.cpp
        tests/test_spectrum.cpp
        tests/test_sources.cpp)
    target_link_libraries(spectro_tests PRIVATE spectro_core spectro_flags)
    add_test(NAME spectro_tests COMMAND spectro_tests)
endif()
