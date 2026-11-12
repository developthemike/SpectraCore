# spectro_add_shaders(<target> SOURCES <glsl>... [INCLUDES <glsl>...])
#
# Compiles each GLSL source to SPIR-V and embeds the result into <target> as a
# C array, so the executable does not depend on shader files at run time.
#
# shaders/surface.vert  ->  <build>/shaders/surface.vert.spv
#                       ->  <build>/shaders/surface_vert_spv.h
#                              static const uint32_t surface_vert_spv[] = {...};
#
# Include the generated header as "shaders/surface_vert_spv.h".

set(_SPECTRO_EMBED_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/EmbedSpirv.cmake")

function(spectro_add_shaders target)
    cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDES" ${ARGN})

    set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders")
    file(MAKE_DIRECTORY "${out_dir}")

    set(include_deps "")
    foreach(inc IN LISTS ARG_INCLUDES)
        list(APPEND include_deps "${CMAKE_CURRENT_SOURCE_DIR}/${inc}")
    endforeach()

    set(headers "")
    foreach(src IN LISTS ARG_SOURCES)
        get_filename_component(src_abs "${src}" ABSOLUTE)
        get_filename_component(src_dir "${src_abs}" DIRECTORY)
        get_filename_component(name "${src}" NAME)
        string(MAKE_C_IDENTIFIER "${name}_spv" symbol)
        set(spv "${out_dir}/${name}.spv")
        set(hdr "${out_dir}/${symbol}.h")

        if(SPECTRO_SHADER_COMPILER_KIND STREQUAL "glslc")
            set(compile_cmd "${SPECTRO_SHADER_COMPILER}"
                --target-env=vulkan1.0 -O -Werror -I "${src_dir}" -o "${spv}" "${src_abs}")
        else()
            set(compile_cmd "${SPECTRO_SHADER_COMPILER}"
                -V --target-env vulkan1.0 --quiet "-I${src_dir}" -o "${spv}" "${src_abs}")
        endif()

        add_custom_command(
            OUTPUT "${hdr}"
            BYPRODUCTS "${spv}"
            COMMAND ${compile_cmd}
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${spv}" "-DOUTPUT=${hdr}" "-DSYMBOL=${symbol}"
                -P "${_SPECTRO_EMBED_SCRIPT}"
            DEPENDS "${src_abs}" ${include_deps} "${_SPECTRO_EMBED_SCRIPT}"
                    ${SPECTRO_SHADER_COMPILER_DEPENDS}
            COMMENT "Compiling shader ${name}"
            VERBATIM)
        list(APPEND headers "${hdr}")
    endforeach()

    add_custom_target(${target}_shaders DEPENDS ${headers})
    add_dependencies(${target} ${target}_shaders)
    target_sources(${target} PRIVATE ${headers})
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
endfunction()
