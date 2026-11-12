#version 450

layout(location = 0) noperspective in vec2 vLocal;
layout(location = 1) flat in vec2 vSize;
layout(location = 2) flat in vec4 vColor;

layout(location = 0) out vec4 outColor;

void main()
{
    // Distance from this pixel to the segment (a capsule's signed distance).
    float beyond = max(max(-vLocal.x, vLocal.x - vSize.x), 0.0);
    float dist   = length(vec2(beyond, vLocal.y));
    float alpha  = clamp(vSize.y + 0.5 - dist, 0.0, 1.0);
    if (alpha <= 0.0)
        discard;
    outColor = vec4(vColor.rgb, vColor.a * alpha);
}
