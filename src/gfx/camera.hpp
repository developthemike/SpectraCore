// Orbit camera: looks at a target point from a given yaw, pitch and distance.
#pragma once

#include "util/math3d.hpp"

namespace spectro {

struct Camera {
    Vec3  target{0.0f, 0.1f, 0.0f};
    float yaw          = 0.0f;   // radians; 0 looks along -Z, positive moves the eye towards +X
    float pitch        = 0.5f;   // radians above the horizon; pi/2 looks straight down
    float distance     = 4.0f;   // eye to target (also sets the zoom of the orthographic view)
    float fov_y        = 0.7f;   // radians
    bool  orthographic = false;
    // Orthographic views only: vertical magnification relative to horizontal, so
    // that a flat 2D view can fill the window whatever the plot's proportions.
    float ortho_stretch = 1.0f;

    Vec3 to_eye() const;   // unit vector from the target towards the eye
    Vec3 right() const;
    Vec3 up() const;
    Vec3 eye() const;

    // Half the height of the visible area at the target, in world units.
    float half_height() const;

    Mat4 view() const;
    Mat4 projection(float aspect) const;

    // The ray through a point of the view, given in normalised device
    // coordinates (x right, y down, both -1..1).
    void ray(float ndc_x, float ndc_y, float aspect, Vec3& origin, Vec3& direction) const;

    // Mouse interaction; deltas are in pixels.
    void orbit(float dx, float dy);
    void pan(float dx, float dy, float viewport_height);
    void zoom(float steps);
};

enum class ViewPreset { Perspective, Top, Front };

// A camera framing a plot of the given world size centred on the origin
// (width along X, height along +Y from 0, depth along Z).
Camera make_view(ViewPreset preset, float plot_width, float plot_height, float plot_depth, float aspect);

}  // namespace spectro
