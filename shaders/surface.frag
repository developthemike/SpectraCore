#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(set = 0, binding = 2) uniform sampler2D uColormap;  // one colour map per texture row

layout(location = 0) in float vLevel;
layout(location = 1) in vec3  vWorld;
layout(location = 2) in vec3  vNormal;
layout(location = 3) in float vAge;
layout(location = 4) in float vWall;

layout(location = 0) out vec4 outColor;

void main()
{
    // Sample the centre of the colour map texel that corresponds to the level.
    float u    = (clamp(vLevel, 0.0, 1.0) * 255.0 + 0.5) / 256.0;
    vec3  base = texture(uColormap, vec2(u, g.misc.y)).rgb;

    vec3  n = normalize(vNormal);
    float shade;
    if (vWall > 0.5) {
        // Walls: the colour map as a vertical gradient, dimmer than the surface
        // so the two read as different faces of one solid.
        shade = g.misc.z * (0.82 + 0.18 * abs(dot(n, g.light.xyz)));
    } else {
        if (!gl_FrontFacing) {
            // Looking at the surface from underneath.
            n = -n;
            base *= 0.45;
        }
        // Flat areas keep the pure colour map colour; slopes facing the light
        // get brighter, slopes facing away darker.
        float diffuse = max(dot(n, g.light.xyz), 0.0) / max(g.light.y, 0.05);
        shade = mix(1.0, g.light.w + (1.0 - g.light.w) * diffuse, g.level.z);
    }

    vec3 color = base * shade;

    // Old rows fade into the background: a depth cue that also hides the
    // shimmer of sub-pixel detail far away.
    float fog = g.level.w * smoothstep(0.3, 1.0, vAge);
    outColor  = vec4(mix(color, g.background.rgb, fog), 1.0);
}
