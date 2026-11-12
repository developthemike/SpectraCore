#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// The spectrogram surface and the walls that close it off at the sides.
//
// There are no vertex buffers.  The mesh is a regular grid, so every vertex is
// derived from gl_VertexIndex / gl_InstanceIndex, and its height is read from
// the spectrogram texture: one texel per (frequency bin, time row).  The texture
// is used as a ring buffer, so new spectra cost one row upload and nothing else.
//
// Time axis ("slots").  The grid has rows + 1 slots along the time axis:
//
//   slot 0        newest spectrum, pinned to the front edge
//   slot s >= 1   spectrum of age s - 1, at depth (s - 1 + phase) rows
//
// `phase` runs smoothly from 0 to 1 between two spectra arriving, so the surface
// glides away from the front edge instead of jumping one row at a time.  The
// last slot is clamped to the back edge.

layout(set = 0, binding = 1) uniform sampler2D uSpectra;  // R32F: dBFS

layout(push_constant) uniform Push {
    int part;
} pc;

const int PART_SURFACE = 0;
const int PART_FRONT   = 1;
const int PART_BACK    = 2;
const int PART_LEFT    = 3;
const int PART_RIGHT   = 4;

layout(location = 0) out float vLevel;   // 0..1 position on the level scale
layout(location = 1) out vec3  vWorld;
layout(location = 2) out vec3  vNormal;
layout(location = 3) out float vAge;     // 0 = newest row, 1 = oldest
layout(location = 4) out float vWall;    // 1 on the walls, 0 on the surface

float levelAt(int col, int age)
{
    col = clamp(col, 0, g.grid.x - 1);
    age = clamp(age, 0, g.grid.y - 1);
    int row = g.grid.z - age;
    if (row < 0)
        row += g.grid.y;
    float db = texelFetch(uSpectra, ivec2(col, row), 0).r;
    return clamp((db - g.level.x) * g.level.y, 0.0, 1.0);
}

void main()
{
    int  cols   = g.grid.x;
    int  rows   = g.grid.y;
    int  index  = gl_VertexIndex >> 1;
    bool second = (gl_VertexIndex & 1) != 0;

    int   col;
    int   slot;
    float top    = 1.0;                    // 1: on the surface, 0: on the floor
    vec3  normal = vec3(0.0, 1.0, 0.0);

    if (pc.part == PART_SURFACE) {
        // One triangle strip per instance, between two neighbouring slots.
        col  = index;
        slot = gl_InstanceIndex + (second ? 1 : 0);
    } else if (pc.part == PART_FRONT || pc.part == PART_BACK) {
        col    = index;
        slot   = pc.part == PART_FRONT ? 0 : rows;
        top    = second ? 0.0 : 1.0;
        normal = vec3(0.0, 0.0, pc.part == PART_FRONT ? 1.0 : -1.0);
    } else {
        slot   = index;
        col    = pc.part == PART_LEFT ? 0 : cols - 1;
        top    = second ? 0.0 : 1.0;
        normal = vec3(pc.part == PART_LEFT ? -1.0 : 1.0, 0.0, 0.0);
    }

    int   age   = max(slot - 1, 0);
    float depth = slot == 0 ? 0.0 : min(float(slot - 1) + g.misc.x, float(rows - 1));
    float h     = levelAt(col, age);

    if (pc.part == PART_SURFACE) {
        // Shading normal from central differences of the level.  This is a
        // stylised relief rather than the true geometric normal: neighbouring
        // bins of a noisy spectrum differ by far more than their spacing, and
        // physically correct normals would turn the lighting into glitter.
        // scale.w sets how strongly a level step tilts the normal.
        float left  = levelAt(col - 1, age);
        float right = levelAt(col + 1, age);
        float newer = levelAt(col, age - 1);
        float older = levelAt(col, age + 1);
        float dydx  = (right - left) * 0.5 * g.scale.w;
        float dydz  = (newer - older) * 0.5 * g.scale.w;  // +Z points towards newer rows
        normal = normalize(vec3(-dydx, 1.0, -dydz));
    }

    float level = h * top;
    vec3  world = vec3(g.origin.x + float(col) * g.scale.x,
                       g.origin.y + level * g.scale.y,
                       g.origin.z - depth * g.scale.z);

    vLevel  = level;
    vWorld  = world;
    vNormal = normal;
    vAge    = depth / float(rows - 1);
    vWall   = pc.part == PART_SURFACE ? 0.0 : 1.0;

    gl_Position = g.viewProj * vec4(world, 1.0);
}
