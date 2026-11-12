#include "ui/axes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "util/units.hpp"

namespace spectro {

namespace {

constexpr uint32_t kGridColor    = pack_rgba(255, 255, 255, 24);
constexpr uint32_t kOutlineColor = pack_rgba(255, 255, 255, 70);
constexpr uint32_t kTickColor    = pack_rgba(255, 255, 255, 150);

void add_line(std::vector<LineInstance>& lines, Vec3 a, Vec3 b, uint32_t color, float width)
{
    LineInstance line{};
    line.start[0]   = a.x;
    line.start[1]   = a.y;
    line.start[2]   = a.z;
    line.width      = width;
    line.end[0]     = b.x;
    line.end[1]     = b.y;
    line.end[2]     = b.z;
    line.depth_bias = 0.0f;
    line.color      = color;
    lines.push_back(line);
}

float screen_distance(const AxesInput& in, Vec3 a, Vec3 b)
{
    float ax, ay, bx, by;
    if (!project_to_screen(in.view_proj, a, in.viewport_width, in.viewport_height, ax, ay) ||
        !project_to_screen(in.view_proj, b, in.viewport_width, in.viewport_height, bx, by))
        return 0.0f;
    return std::hypot(bx - ax, by - ay);
}

std::string format_tick(double value, int decimals)
{
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, value);
    // Avoid "-0".
    if (std::fabs(value) < 0.5 * std::pow(10.0, -decimals))
        std::snprintf(buf, sizeof(buf), "%.*f", decimals, 0.0);
    return buf;
}

}  // namespace

bool project_to_screen(const Mat4& view_proj, Vec3 world, float viewport_width, float viewport_height,
                       float& x, float& y)
{
    const Vec4 clip = view_proj * Vec4{world.x, world.y, world.z, 1.0f};
    if (clip.w <= 1e-6f)
        return false;
    x = (clip.x / clip.w * 0.5f + 0.5f) * viewport_width;
    y = (clip.y / clip.w * 0.5f + 0.5f) * viewport_height;
    return true;
}

void build_axes(const AxesInput& in, std::vector<LineInstance>& lines, std::vector<AxisLabel>& labels)
{
    const float w = in.plot_width, h = in.plot_height, d = in.plot_depth;
    const float x_left = -0.5f * w, x_right = 0.5f * w;
    const float z_front = 0.5f * d, z_back = -0.5f * d;  // front = newest row

    // The walls carrying the grid are the two behind the data as seen from the
    // camera; ticks and labels go on the edges nearest to it.  Which those are
    // follows from the yaw alone (both "nearer" and "lower on screen" flip
    // where its cosine or sine changes sign), which stays well defined even
    // when looking straight down.
    const Vec3  to_eye       = in.camera.to_eye();
    const bool  cam_in_front = std::cos(in.camera.yaw) >= 0.0f;
    const bool  cam_on_right = std::sin(in.camera.yaw) >= 0.0f;
    const bool  top_down     = to_eye.y > 0.96f;
    const float z_far  = cam_in_front ? z_back : z_front;
    const float z_near = cam_in_front ? z_front : z_back;
    const float x_far  = cam_on_right ? x_left : x_right;
    // The time axis runs along the near side edge, except in the flat top view,
    // where it goes on the left of the screen like on a classic waterfall.
    const bool  time_on_right_edge = top_down ? !cam_in_front : cam_on_right;
    const float x_near = time_on_right_edge ? x_right : x_left;
    const float z_out  = cam_in_front ? 1.0f : -1.0f;        // outward direction at the near Z edge
    const float x_out  = time_on_right_edge ? 1.0f : -1.0f;  // outward direction at the time axis

    // The frame sits a hair outside the data so its lines never fight the
    // surface's own walls for the same depth.
    const float eps     = 0.003f * w;
    const float zf_far  = z_far - z_out * eps;
    const float xf_far  = x_far + (x_far < 0.0f ? -eps : eps);
    const float zf_near = z_near + z_out * eps;
    const float xf_near = x_near + x_out * eps;

    const float thin   = 1.0f * in.pixel_ratio;
    const float medium = 1.25f * in.pixel_ratio;
    const float tick   = 0.022f * w;

    // How directly each axis faces the viewer decides whether its labels make sense.
    const bool show_level_axis = !top_down;                                          // not looking straight down
    const bool show_time_axis  = in.history_seconds > 0.0 &&
                                 !(std::fabs(to_eye.z) > 0.97f && std::fabs(to_eye.y) < 0.25f);  // not end-on
    const bool show_freq_axis  = !(std::fabs(to_eye.x) > 0.97f && std::fabs(to_eye.y) < 0.25f);

    // ---- Frequency axis -------------------------------------------------------
    const bool   use_mhz    = in.span_hz >= 200e3 || std::fabs(in.center_hz) >= 10e6;
    const double freq_unit  = use_mhz ? 1e6 : 1e3;
    const double f_lo       = (in.center_hz - 0.5 * in.span_hz) / freq_unit;
    const double f_hi       = (in.center_hz + 0.5 * in.span_hz) / freq_unit;
    const float  freq_px    = screen_distance(in, {x_left, 0.0f, z_near}, {x_right, 0.0f, z_near});
    const int    freq_count = std::clamp(static_cast<int>(freq_px / 85.0f), 2, 14);
    const Ticks  freq_ticks = nice_ticks(f_lo, f_hi, freq_count);
    const auto   freq_to_x  = [&](double f) {
        return x_left + static_cast<float>((f - f_lo) / (f_hi - f_lo)) * w;
    };

    // ---- Level axis -----------------------------------------------------------
    const float level_px    = screen_distance(in, {x_far, 0.0f, z_near}, {x_far, h, z_near});
    const int   level_count = std::clamp(static_cast<int>(level_px / 38.0f), 2, 10);
    const Ticks level_ticks = nice_ticks(in.db_floor, in.db_ceiling, level_count);
    const auto  level_to_y  = [&](double db) {
        return static_cast<float>((db - in.db_floor) / (in.db_ceiling - in.db_floor)) * h;
    };

    // ---- Time axis ------------------------------------------------------------
    const float time_px    = screen_distance(in, {x_near, 0.0f, z_front}, {x_near, 0.0f, z_back});
    const int   time_count = std::clamp(static_cast<int>(time_px / 60.0f), 2, 12);
    const Ticks time_ticks = in.history_seconds > 0.0 ? nice_ticks(0.0, in.history_seconds, time_count) : Ticks{};
    const auto  time_to_z  = [&](double t) {
        return z_front - static_cast<float>(t / in.history_seconds) * d;
    };

    // ---- Lines ----------------------------------------------------------------
    if (in.show_grid) {
        // Floor outline
        add_line(lines, {x_left - eps, 0, zf_near}, {x_right + eps, 0, zf_near}, kOutlineColor, medium);
        add_line(lines, {x_left - eps, 0, zf_far}, {x_right + eps, 0, zf_far}, kOutlineColor, medium);
        add_line(lines, {x_left - eps, 0, z_front + eps}, {x_left - eps, 0, z_back - eps}, kOutlineColor, medium);
        add_line(lines, {x_right + eps, 0, z_front + eps}, {x_right + eps, 0, z_back - eps}, kOutlineColor, medium);

        // Back wall: level lines and frequency lines
        add_line(lines, {x_left, h, zf_far}, {x_right, h, zf_far}, kOutlineColor, medium);
        add_line(lines, {x_left, 0, zf_far}, {x_left, h, zf_far}, kOutlineColor, medium);
        add_line(lines, {x_right, 0, zf_far}, {x_right, h, zf_far}, kOutlineColor, medium);
        for (int i = 0; i < level_ticks.count; ++i) {
            const float y = level_to_y(level_ticks.value(i));
            if (y > 0.002f * h && y < 0.998f * h)
                add_line(lines, {x_left, y, zf_far}, {x_right, y, zf_far}, kGridColor, thin);
        }
        for (int i = 0; i < freq_ticks.count; ++i) {
            const float x = freq_to_x(freq_ticks.value(i));
            add_line(lines, {x, 0, zf_far}, {x, h, zf_far}, kGridColor, thin);
        }

        // Side wall: level lines and time lines
        add_line(lines, {xf_far, h, z_front}, {xf_far, h, z_back}, kOutlineColor, medium);
        add_line(lines, {xf_far, 0, z_near}, {xf_far, h, z_near}, kOutlineColor, medium);
        for (int i = 0; i < level_ticks.count; ++i) {
            const float y = level_to_y(level_ticks.value(i));
            if (y > 0.002f * h && y < 0.998f * h)
                add_line(lines, {xf_far, y, z_front}, {xf_far, y, z_back}, kGridColor, thin);
        }
        for (int i = 0; i < time_ticks.count; ++i) {
            const float z = time_to_z(time_ticks.value(i));
            add_line(lines, {xf_far, 0, z}, {xf_far, h, z}, kGridColor, thin);
        }
    }

    // ---- Ticks and labels -------------------------------------------------------
    // Which way "right" on screen runs along the world X axis.
    const float screen_x_of_world_x = in.camera.right().x >= 0.0f ? 1.0f : -1.0f;

    if (show_freq_axis) {
        const Vec3 outward{0.0f, 0.0f, z_out};
        for (int i = 0; i < freq_ticks.count; ++i) {
            const float x = freq_to_x(freq_ticks.value(i));
            add_line(lines, {x, 0, zf_near}, {x, 0, zf_near + z_out * tick}, kTickColor, medium);

            AxisLabel label;
            label.anchor  = {x, 0.0f, zf_near + z_out * tick};
            label.outward = outward;
            label.axis    = kAxisFrequency;
            label.text    = format_tick(freq_ticks.value(i), freq_ticks.decimals);
            labels.push_back(std::move(label));
        }
        AxisLabel title;
        title.anchor   = {0.0f, 0.0f, zf_near + z_out * tick};
        title.outward  = outward;
        title.distance = 2.0f;
        title.axis     = kAxisFrequency;
        title.title    = true;
        title.text     = use_mhz ? "Frequency (MHz)" : "Frequency (kHz)";
        labels.push_back(std::move(title));
    }

    if (show_level_axis) {
        // Along the vertical edge where the side wall meets the near edge.
        const float sign = x_far < 0.0f ? -1.0f : 1.0f;  // outward
        const Vec3  outward{sign, 0.0f, 0.0f};
        for (int i = 0; i < level_ticks.count; ++i) {
            const float y = level_to_y(level_ticks.value(i));
            add_line(lines, {xf_far, y, z_near}, {xf_far + sign * tick, y, z_near}, kTickColor, medium);

            AxisLabel label;
            label.anchor     = {xf_far + sign * tick, y, z_near};
            label.outward    = outward;
            label.fallback_x = sign * screen_x_of_world_x;
            label.fallback_y = 0.0f;
            label.axis       = kAxisLevel;
            label.text       = format_tick(level_ticks.value(i), level_ticks.decimals);
            labels.push_back(std::move(label));
        }
        AxisLabel title;
        title.anchor     = {xf_far, h, z_near};
        title.outward    = {0.0f, 1.0f, 0.0f};
        title.fallback_x = 0.0f;
        title.fallback_y = -1.0f;
        title.distance   = 0.6f;
        title.axis       = kAxisLevel;
        title.title      = true;
        title.text       = "dBFS";
        labels.push_back(std::move(title));
    }

    if (show_time_axis) {
        const Vec3 outward{x_out, 0.0f, 0.0f};
        for (int i = 0; i < time_ticks.count; ++i) {
            const float z = time_to_z(time_ticks.value(i));
            add_line(lines, {xf_near, 0, z}, {xf_near + x_out * tick, 0, z}, kTickColor, medium);

            AxisLabel label;
            label.anchor     = {xf_near + x_out * tick, 0.0f, z};
            label.outward    = outward;
            label.fallback_x = x_out * screen_x_of_world_x;
            label.fallback_y = 0.0f;
            label.axis       = kAxisTime;
            label.text       = format_tick(time_ticks.value(i), time_ticks.decimals) + " s";
            labels.push_back(std::move(label));
        }
    }
}

}  // namespace spectro
