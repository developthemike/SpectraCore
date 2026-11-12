#include "gfx/camera.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace spectro {

namespace {
constexpr float kHalfPi   = static_cast<float>(std::numbers::pi / 2.0);
constexpr float kMinPitch = -0.45f;  // allow a peek from below the floor
constexpr float kMaxPitch = kHalfPi;
// The orthographic eye sits this far out so the plot never crosses the near plane.
constexpr float kOrthoEyeDistance = 30.0f;
}  // namespace

Vec3 Camera::to_eye() const
{
    return {std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)};
}

Vec3 Camera::right() const
{
    return {std::cos(yaw), 0.0f, -std::sin(yaw)};
}

Vec3 Camera::up() const
{
    // right x forward, with forward = -to_eye(); well defined even looking straight down.
    return cross(right(), -to_eye());
}

Vec3 Camera::eye() const
{
    return target + to_eye() * (orthographic ? kOrthoEyeDistance : distance);
}

float Camera::half_height() const
{
    return distance * std::tan(fov_y * 0.5f);
}

Mat4 Camera::view() const
{
    return view_matrix(eye(), right(), up(), -to_eye());
}

Mat4 Camera::projection(float aspect) const
{
    if (orthographic) {
        const float hh = half_height();
        return spectro::orthographic(hh * aspect, hh / ortho_stretch, 0.1f, 2.0f * kOrthoEyeDistance);
    }
    const float z_near = std::max(0.01f, distance * 0.02f);
    return perspective(fov_y, aspect, z_near, distance + 40.0f);
}

void Camera::ray(float ndc_x, float ndc_y, float aspect, Vec3& origin, Vec3& direction) const
{
    const Vec3  forward = -to_eye();
    const float hh      = half_height();
    if (orthographic) {
        origin    = eye() + right() * (ndc_x * hh * aspect) + up() * (-ndc_y * hh / ortho_stretch);
        direction = forward;
    } else {
        const float tan_half = std::tan(fov_y * 0.5f);
        origin    = eye();
        direction = normalize(forward + right() * (ndc_x * tan_half * aspect) + up() * (-ndc_y * tan_half));
    }
}

void Camera::orbit(float dx, float dy)
{
    constexpr float kRadiansPerPixel = 0.005f;
    yaw -= dx * kRadiansPerPixel;
    pitch = std::clamp(pitch + dy * kRadiansPerPixel, kMinPitch, kMaxPitch);
    // Keep yaw in (-pi, pi] so it never loses precision.
    const float pi = static_cast<float>(std::numbers::pi);
    if (yaw > pi)
        yaw -= 2.0f * pi;
    else if (yaw <= -pi)
        yaw += 2.0f * pi;
}

void Camera::pan(float dx, float dy, float viewport_height)
{
    if (viewport_height <= 0.0f)
        return;
    // Move the target so that the point under the cursor follows the cursor.
    const float units_per_pixel = 2.0f * half_height() / viewport_height;
    const float vertical        = orthographic ? units_per_pixel / ortho_stretch : units_per_pixel;
    target = target - right() * (dx * units_per_pixel) + up() * (dy * vertical);
}

void Camera::zoom(float steps)
{
    distance = std::clamp(distance * std::pow(0.88f, steps), 0.15f, 60.0f);
}

Camera make_view(ViewPreset preset, float plot_width, float plot_height, float plot_depth, float aspect)
{
    Camera c;
    c.fov_y = radians(38.0f);
    const float tan_half = std::tan(c.fov_y * 0.5f);
    aspect = std::max(aspect, 0.2f);

    switch (preset) {
    case ViewPreset::Perspective: {
        c.orthographic = false;
        c.yaw          = radians(-20.0f);
        c.pitch        = radians(29.0f);
        c.target       = {0.0f, plot_height * 0.2f, 0.0f};
        // Far enough to see the whole footprint, whichever way the window is shaped.
        const float radius = 0.5f * std::sqrt(plot_width * plot_width + plot_depth * plot_depth);
        c.distance = std::max(radius * 0.93f / tan_half, radius * 1.12f / (tan_half * aspect));
        break;
    }
    case ViewPreset::Top:
    case ViewPreset::Front: {
        // Flat views without perspective: straight down for the classic 2D
        // waterfall, straight on for the classic spectrum analyser display.
        // The plot is scaled independently in both directions to fill the
        // window, leaving room for the axis labels.
        const bool  top      = preset == ViewPreset::Top;
        const float extent_v = top ? plot_depth : plot_height;
        c.orthographic = true;
        c.yaw          = 0.0f;
        c.pitch        = top ? kHalfPi : 0.0f;
        c.target       = top ? Vec3{0.0f, 0.0f, 0.0f} : Vec3{0.0f, plot_height * 0.5f, 0.0f};

        const float half_w = plot_width * 0.5f * 1.16f;
        const float half_v = extent_v * 0.5f * (top ? 1.28f : 1.42f);
        const float hh     = half_w / aspect;  // vertical half extent of an unstretched view this wide
        c.ortho_stretch    = std::clamp(hh / half_v, 0.1f, 10.0f);
        c.distance         = hh / tan_half;
        break;
    }
    }
    return c;
}

}  // namespace spectro
