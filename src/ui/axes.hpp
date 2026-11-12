// The reference frame around the plot: grid walls behind the data, tick marks
// and tick labels along the edges nearest to the viewer.
#pragma once

#include <string>
#include <vector>

#include "gfx/camera.hpp"
#include "gfx/renderer.hpp"
#include "util/math3d.hpp"

namespace spectro {

// A piece of text attached to a point in the 3D scene.  The text is placed
// just beyond the anchor in the `outward` direction as it appears on screen, so
// it clears the plot from any viewing angle.
struct AxisLabel {
    Vec3        anchor;
    Vec3        outward;              // world direction pointing away from the plot
    float       fallback_x = 0.0f;    // screen direction to use when `outward` points
    float       fallback_y = 1.0f;    //   straight at the viewer and has no screen direction
    float       distance   = 0.45f;   // gap between the anchor and the text, in line heights
    int         axis       = 0;       // tick labels of one axis are thinned out together
    bool        title      = false;   // axis title rather than a tick value
    std::string text;
};

struct AxesInput {
    float  plot_width  = 2.0f;
    float  plot_height = 0.5f;
    float  plot_depth  = 2.4f;

    double center_hz       = 0.0;   // tuned frequency
    double span_hz         = 1.0;   // displayed bandwidth (the sample rate)
    double db_floor        = -100.0;
    double db_ceiling      = 0.0;
    double history_seconds = 0.0;   // signal time covered by the depth of the plot; 0 hides the time axis

    Camera camera;
    Mat4   view_proj;
    float  viewport_width  = 1.0f;  // in user-interface units
    float  viewport_height = 1.0f;
    float  pixel_ratio     = 1.0f;  // framebuffer pixels per user-interface unit
    bool   show_grid       = true;
};

enum AxisId : int { kAxisFrequency = 0, kAxisLevel = 1, kAxisTime = 2 };

void build_axes(const AxesInput& input, std::vector<LineInstance>& lines, std::vector<AxisLabel>& labels);

// Projects a world position to user-interface coordinates.  Returns false if
// the point is behind the camera.
bool project_to_screen(const Mat4& view_proj, Vec3 world, float viewport_width, float viewport_height,
                       float& x, float& y);

}  // namespace spectro
