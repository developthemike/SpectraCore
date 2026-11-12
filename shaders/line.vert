#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Anti-aliased lines of arbitrary pixel width.
//
// Each instance is one 3D segment; the four vertices of a quad are placed
// around its projection, in screen space.  The fragment shader then shades the
// quad as a capsule, which gives round caps and joins and smooth edges without
// relying on the (optional) wide-line support of the GPU.

layout(location = 0) in vec4 inStart;  // xyz: world position, w: width in pixels
layout(location = 1) in vec4 inEnd;    // xyz: world position, w: depth bias towards the camera
layout(location = 2) in vec4 inColor;

layout(location = 0) noperspective out vec2 vLocal;  // pixels: x along the segment, y across it
layout(location = 1) flat out vec2 vSize;            // x: segment length, y: half width (pixels)
layout(location = 2) flat out vec4 vColor;

void main()
{
    vec4 c0 = g.viewProj * vec4(inStart.xyz, 1.0);
    vec4 c1 = g.viewProj * vec4(inEnd.xyz, 1.0);

    vLocal = vec2(0.0);
    vSize  = vec2(0.0);
    vColor = inColor;

    // Clip the segment against the near plane (z = 0 in clip space) ourselves:
    // an end point behind the camera cannot be projected to the screen.
    if (c0.z < 0.0 && c1.z < 0.0) {
        gl_Position = vec4(0.0, 0.0, -1.0, 1.0);  // outside the clip volume
        return;
    }
    if (c0.z < 0.0)
        c0 = mix(c0, c1, clamp(-c0.z / (c1.z - c0.z) + 1e-5, 0.0, 1.0));
    else if (c1.z < 0.0)
        c1 = mix(c1, c0, clamp(-c1.z / (c0.z - c1.z) + 1e-5, 0.0, 1.0));

    vec2  halfViewport = 0.5 * g.viewport.xy;
    vec2  s0  = c0.xy / c0.w * halfViewport;
    vec2  s1  = c1.xy / c1.w * halfViewport;
    vec2  d   = s1 - s0;
    float len = length(d);
    vec2  dir = len > 1e-4 ? d / len : vec2(1.0, 0.0);
    vec2  nrm = vec2(-dir.y, dir.x);

    float halfWidth = 0.5 * inStart.w;
    float extent    = halfWidth + 1.0;  // room for the cap and the soft edge
    bool  atEnd     = (gl_VertexIndex & 2) != 0;
    float side      = (gl_VertexIndex & 1) != 0 ? 1.0 : -1.0;

    vec4 c      = atEnd ? c1 : c0;
    vec2 offset = dir * (atEnd ? extent : -extent) + nrm * (side * extent);
    c.xy += offset / halfViewport * c.w;
    c.z  -= inEnd.w * c.w;
    gl_Position = c;

    vLocal = vec2(atEnd ? len + extent : -extent, side * extent);
    vSize  = vec2(len, halfWidth);
}
