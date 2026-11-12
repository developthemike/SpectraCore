# Locates every third-party dependency.
#
# Policy: use what is installed on the system when it is there, and otherwise
# download a pinned, checksummed release into the build tree (never into the
# system).  The result is always the same set of targets:
#
#   PkgConfig::FFTW3F   single-precision FFTW                  (required, system)
#   spectro_bladerf     libbladeRF                             (optional, system)
#
# and, when SPECTRO_BUILD_GUI is ON:
#
#   spectro_vulkan      Vulkan headers + loader
#   spectro_glfw        GLFW 3.3+
#   spectro_imgui       Dear ImGui + its GLFW/Vulkan backends  (always fetched)
#   ZLIB::ZLIB          for PNG screenshots                    (required, system)
#
# plus SPECTRO_SHADER_COMPILER / SPECTRO_SHADER_COMPILER_KIND for Shaders.cmake.

include(FetchContent)

# Pinned releases --------------------------------------------------------------
set(SPECTRO_VULKAN_HEADERS_URL
    "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.341.0.tar.gz")
set(SPECTRO_VULKAN_HEADERS_SHA256 d73bc5036b6556b741f6985ff600ca720308c5f2850e4a43ceb498bd3de069e7)

set(SPECTRO_GLFW_URL "https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz")
set(SPECTRO_GLFW_SHA256 c038d34200234d071fae9345bc455e4a8f2f544ab60150765d7704e08f3dac01)

set(SPECTRO_IMGUI_URL "https://github.com/ocornut/imgui/archive/refs/tags/v1.93.0.tar.gz")
set(SPECTRO_IMGUI_SHA256 f7bf6bc76e5d79d40d751096663e6c5d18daf08129dd52e7f7b09e4251c8eeee)

set(SPECTRO_GLSLANG_URL
    "https://github.com/KhronosGroup/glslang/archive/refs/tags/vulkan-sdk-1.4.341.0.tar.gz")
set(SPECTRO_GLSLANG_SHA256 18cdf7d40a39001dde842c6a1338b2c9321ac5e487139b9d52b4b9c666da3c86)

# Downloads and unpacks <name> without running its own CMakeLists.txt.
# Sets <name>_SOURCE_DIR in the caller's scope.
macro(spectro_fetch_sources name url sha256)
    if(NOT SPECTRO_FETCH_MISSING)
        message(FATAL_ERROR
            "${name} is not installed and SPECTRO_FETCH_MISSING is OFF.\n"
            "Install it system-wide or configure with -DSPECTRO_FETCH_MISSING=ON.")
    endif()
    FetchContent_Declare(${name}
        URL "${url}"
        URL_HASH SHA256=${sha256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        # A sub-directory that does not exist: FetchContent then only downloads.
        SOURCE_SUBDIR spectro-sources-only)
    FetchContent_MakeAvailable(${name})
endmacro()

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# --- FFTW ----------------------------------------------------------------------
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)

# --- libbladeRF (optional) -------------------------------------------------------
set(SPECTRO_HAVE_BLADERF OFF)
if(SPECTRO_WITH_BLADERF)
    pkg_check_modules(BLADERF IMPORTED_TARGET libbladeRF)
    if(BLADERF_FOUND)
        add_library(spectro_bladerf INTERFACE)
        target_link_libraries(spectro_bladerf INTERFACE PkgConfig::BLADERF)
        set(SPECTRO_HAVE_BLADERF ON)
    else()
        find_path(BLADERF_INCLUDE_DIR libbladeRF.h)
        find_library(BLADERF_LIBRARY NAMES bladeRF)
        if(BLADERF_INCLUDE_DIR AND BLADERF_LIBRARY)
            add_library(spectro_bladerf INTERFACE)
            target_include_directories(spectro_bladerf INTERFACE "${BLADERF_INCLUDE_DIR}")
            target_link_libraries(spectro_bladerf INTERFACE "${BLADERF_LIBRARY}")
            set(SPECTRO_HAVE_BLADERF ON)
        endif()
    endif()
    if(SPECTRO_HAVE_BLADERF)
        message(STATUS "libbladeRF: found")
    else()
        message(WARNING
            "libbladeRF was not found: building WITHOUT radio support "
            "(only the simulator and file playback will work). "
            "Install it with e.g. 'sudo apt install libbladerf-dev'.")
    endif()
else()
    message(STATUS "libbladeRF: disabled (SPECTRO_WITH_BLADERF=OFF)")
endif()

# Everything below is only needed for the user interface.
if(NOT SPECTRO_BUILD_GUI)
    return()
endif()

find_package(ZLIB REQUIRED)

# --- Vulkan --------------------------------------------------------------------
add_library(spectro_vulkan INTERFACE)
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    target_link_libraries(spectro_vulkan INTERFACE Vulkan::Vulkan)
    message(STATUS "Vulkan: system SDK/dev package (${Vulkan_VERSION})")
else()
    # No development package.  The loader itself ships with every GPU driver, so
    # link against the runtime library and bring our own headers.
    find_library(SPECTRO_VULKAN_LOADER NAMES vulkan libvulkan.so.1 vulkan-1)
    if(NOT SPECTRO_VULKAN_LOADER)
        message(FATAL_ERROR
            "The Vulkan loader (libvulkan) was not found. Install your GPU's Vulkan "
            "driver, e.g. 'sudo apt install libvulkan1 mesa-vulkan-drivers'.")
    endif()
    find_path(SPECTRO_VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h)
    if(NOT SPECTRO_VULKAN_INCLUDE_DIR)
        spectro_fetch_sources(vulkan_headers
            "${SPECTRO_VULKAN_HEADERS_URL}" ${SPECTRO_VULKAN_HEADERS_SHA256})
        set(SPECTRO_VULKAN_INCLUDE_DIR "${vulkan_headers_SOURCE_DIR}/include")
        message(STATUS "Vulkan: fetched headers + system loader ${SPECTRO_VULKAN_LOADER}")
    else()
        message(STATUS "Vulkan: system headers + loader ${SPECTRO_VULKAN_LOADER}")
    endif()
    target_include_directories(spectro_vulkan SYSTEM INTERFACE "${SPECTRO_VULKAN_INCLUDE_DIR}")
    target_link_libraries(spectro_vulkan INTERFACE "${SPECTRO_VULKAN_LOADER}")
endif()

# --- GLFW ----------------------------------------------------------------------
add_library(spectro_glfw INTERFACE)
find_package(glfw3 3.3 CONFIG QUIET)
if(TARGET glfw)
    target_link_libraries(spectro_glfw INTERFACE glfw)
    message(STATUS "GLFW: system dev package (${glfw3_VERSION})")
else()
    # Is the GLFW shared library installed without its headers?  Then fetching
    # the headers alone is enough.  They are GLFW 3.4 headers, so the installed
    # library must be at least that version (its real file name carries it).
    find_library(SPECTRO_GLFW_RUNTIME NAMES glfw glfw3 libglfw.so.3)
    set(SPECTRO_GLFW_RUNTIME_USABLE FALSE)
    if(SPECTRO_GLFW_RUNTIME)
        get_filename_component(_glfw_real "${SPECTRO_GLFW_RUNTIME}" REALPATH)
        if(_glfw_real MATCHES "\\.so\\.([0-9]+)\\.([0-9]+)")
            # (Nested on purpose: CMAKE_MATCH_n is only set once MATCHES has run.)
            if("${CMAKE_MATCH_1}.${CMAKE_MATCH_2}" VERSION_GREATER_EQUAL "3.4")
                set(SPECTRO_GLFW_RUNTIME_USABLE TRUE)
            endif()
        endif()
    endif()
    if(SPECTRO_GLFW_RUNTIME_USABLE)
        spectro_fetch_sources(glfw_headers "${SPECTRO_GLFW_URL}" ${SPECTRO_GLFW_SHA256})
        target_include_directories(spectro_glfw SYSTEM INTERFACE
            "${glfw_headers_SOURCE_DIR}/include")
        target_link_libraries(spectro_glfw INTERFACE "${SPECTRO_GLFW_RUNTIME}")
        message(STATUS "GLFW: fetched headers + system runtime ${SPECTRO_GLFW_RUNTIME}")
    else()
        if(NOT SPECTRO_FETCH_MISSING)
            message(FATAL_ERROR "GLFW is not installed (try 'sudo apt install libglfw3-dev').")
        endif()
        # Build GLFW from source.  This needs the X11 and/or Wayland development
        # headers; GLFW's own configure step reports what is missing.
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(glfw
            URL "${SPECTRO_GLFW_URL}"
            URL_HASH SHA256=${SPECTRO_GLFW_SHA256}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            EXCLUDE_FROM_ALL SYSTEM)
        FetchContent_MakeAvailable(glfw)
        target_link_libraries(spectro_glfw INTERFACE glfw)
        message(STATUS "GLFW: building from source")
    endif()
endif()

# --- Dear ImGui ----------------------------------------------------------------
# ImGui is distributed as sources meant to be compiled into the application.
spectro_fetch_sources(imgui "${SPECTRO_IMGUI_URL}" ${SPECTRO_IMGUI_SHA256})
add_library(spectro_imgui STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp")
target_include_directories(spectro_imgui SYSTEM PUBLIC
    "${imgui_SOURCE_DIR}" "${imgui_SOURCE_DIR}/backends")
target_compile_definitions(spectro_imgui PUBLIC
    IMGUI_DISABLE_OBSOLETE_FUNCTIONS
    IMGUI_DISABLE_DEMO_WINDOWS)
target_compile_features(spectro_imgui PUBLIC cxx_std_17)
target_link_libraries(spectro_imgui PUBLIC spectro_vulkan spectro_glfw)
set_target_properties(spectro_imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- GLSL -> SPIR-V compiler ---------------------------------------------------
find_program(SPECTRO_GLSLC NAMES glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES bin)
find_program(SPECTRO_GLSLANG NAMES glslang glslangValidator HINTS ENV VULKAN_SDK PATH_SUFFIXES bin)
set(SPECTRO_SHADER_COMPILER_DEPENDS "")
if(SPECTRO_GLSLC)
    set(SPECTRO_SHADER_COMPILER "${SPECTRO_GLSLC}")
    set(SPECTRO_SHADER_COMPILER_KIND glslc)
    message(STATUS "Shader compiler: ${SPECTRO_GLSLC}")
elseif(SPECTRO_GLSLANG)
    set(SPECTRO_SHADER_COMPILER "${SPECTRO_GLSLANG}")
    set(SPECTRO_SHADER_COMPILER_KIND glslang)
    message(STATUS "Shader compiler: ${SPECTRO_GLSLANG}")
else()
    if(NOT SPECTRO_FETCH_MISSING)
        message(FATAL_ERROR
            "No GLSL compiler found (try 'sudo apt install glslc' or 'glslang-tools').")
    endif()
    # Build the reference compiler as a host tool.  Needs Python 3 at build time.
    set(ENABLE_OPT             OFF CACHE BOOL "" FORCE)  # would need SPIRV-Tools
    set(ENABLE_HLSL            OFF CACHE BOOL "" FORCE)
    set(BUILD_EXTERNAL         OFF CACHE BOOL "" FORCE)
    set(GLSLANG_TESTS          OFF CACHE BOOL "" FORCE)
    set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)
    FetchContent_Declare(glslang
        URL "${SPECTRO_GLSLANG_URL}"
        URL_HASH SHA256=${SPECTRO_GLSLANG_SHA256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        EXCLUDE_FROM_ALL SYSTEM)
    FetchContent_MakeAvailable(glslang)
    set(SPECTRO_SHADER_COMPILER "$<TARGET_FILE:glslang-standalone>")
    set(SPECTRO_SHADER_COMPILER_KIND glslang)
    set(SPECTRO_SHADER_COMPILER_DEPENDS glslang-standalone)
    message(STATUS "Shader compiler: building glslang from source")
endif()
