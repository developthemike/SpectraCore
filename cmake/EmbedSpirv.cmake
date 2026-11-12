# Script mode: cmake -DINPUT=<file.spv> -DOUTPUT=<file.h> -DSYMBOL=<name> -P EmbedSpirv.cmake
#
# Turns a SPIR-V binary into a header defining
#     static const uint32_t <SYMBOL>[] = { 0x07230203, ... };

if(NOT INPUT OR NOT OUTPUT OR NOT SYMBOL)
    message(FATAL_ERROR "EmbedSpirv.cmake needs INPUT, OUTPUT and SYMBOL")
endif()

file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" hex_len)
math(EXPR remainder "${hex_len} % 8")
if(hex_len EQUAL 0 OR NOT remainder EQUAL 0)
    message(FATAL_ERROR "${INPUT} is not a valid SPIR-V module (size is not a multiple of 4 bytes)")
endif()

# SPIR-V words are stored little-endian: reorder each group of four bytes.
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1," words "${hex}")
# Wrap lines for readability (8 words per line).
string(REGEX REPLACE "(0x........,0x........,0x........,0x........,0x........,0x........,0x........,0x........,)"
       "    \\1\n" words "${words}")

if(NOT words MATCHES "0x07230203")
    message(FATAL_ERROR "${INPUT} does not start with the SPIR-V magic number")
endif()

file(WRITE "${OUTPUT}"
"// Generated from ${INPUT} - do not edit.
#pragma once
#include <cstdint>

static const uint32_t ${SYMBOL}[] = {
${words}
};
")
