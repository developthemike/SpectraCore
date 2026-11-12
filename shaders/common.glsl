// Uniform block shared by every shader.  Mirrors GlobalsUbo in src/gfx/renderer.cpp.

layout(set = 0, binding = 0, std140) uniform Globals {
    mat4  viewProj;
    vec4  eye;         // xyz: camera position (w = 1), or direction towards the camera (w = 0, orthographic)
    vec4  origin;      // world position of column 0, level 0, newest row
    vec4  scale;       // x: world units per column, y: world height of level 1, z: world units per row,
                       // w: how strongly level differences tilt the shading normal
    vec4  level;       // x: dBFS at level 0, y: 1 / (dB span), z: relief strength, w: fog strength
    ivec4 grid;        // x: columns, y: rows, z: texture row holding the newest spectrum
    vec4  misc;        // x: scroll phase 0..1, y: colour map V coordinate, z: wall brightness
    vec4  viewport;    // xy: framebuffer size in pixels, zw: 1 / size
    vec4  background;  // rgb: clear colour, which distant rows fade into
    vec4  light;       // xyz: direction towards the light, w: ambient term
} g;
