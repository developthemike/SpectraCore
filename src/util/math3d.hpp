// Just enough 3D math for one camera and one plot.
//
// Conventions: right-handed world with +Y up; matrices are column-major and
// multiply column vectors (clip = proj * view * world).  Projections target
// Vulkan clip space: +Y down, depth 0 (near) to 1 (far).
#pragma once

#include <cmath>

namespace spectro {

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
};

inline Vec3  operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3  operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3  operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3  operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3  cross(Vec3 a, Vec3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3  normalize(Vec3 a)
{
    const float len = length(a);
    return len > 0.0f ? a * (1.0f / len) : a;
}

struct Vec4 {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;
};

struct Mat4 {
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // m[column * 4 + row]

    float&       at(int row, int column) { return m[column * 4 + row]; }
    const float& at(int row, int column) const { return m[column * 4 + row]; }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b)
{
    Mat4 r;
    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
                sum += a.at(row, k) * b.at(k, c);
            r.at(row, c) = sum;
        }
    }
    return r;
}

inline Vec4 operator*(const Mat4& a, Vec4 v)
{
    return {a.at(0, 0) * v.x + a.at(0, 1) * v.y + a.at(0, 2) * v.z + a.at(0, 3) * v.w,
            a.at(1, 0) * v.x + a.at(1, 1) * v.y + a.at(1, 2) * v.z + a.at(1, 3) * v.w,
            a.at(2, 0) * v.x + a.at(2, 1) * v.y + a.at(2, 2) * v.z + a.at(2, 3) * v.w,
            a.at(3, 0) * v.x + a.at(3, 1) * v.y + a.at(3, 2) * v.z + a.at(3, 3) * v.w};
}

// View matrix for a camera at `eye` with an orthonormal basis: `right`, `up`,
// and `forward` (the viewing direction).
inline Mat4 view_matrix(Vec3 eye, Vec3 right, Vec3 up, Vec3 forward)
{
    Mat4 v;
    v.at(0, 0) = right.x;    v.at(0, 1) = right.y;    v.at(0, 2) = right.z;    v.at(0, 3) = -dot(right, eye);
    v.at(1, 0) = up.x;       v.at(1, 1) = up.y;       v.at(1, 2) = up.z;       v.at(1, 3) = -dot(up, eye);
    v.at(2, 0) = -forward.x; v.at(2, 1) = -forward.y; v.at(2, 2) = -forward.z; v.at(2, 3) = dot(forward, eye);
    v.at(3, 0) = 0.0f;       v.at(3, 1) = 0.0f;       v.at(3, 2) = 0.0f;       v.at(3, 3) = 1.0f;
    return v;
}

// Perspective projection; fov_y in radians.
inline Mat4 perspective(float fov_y, float aspect, float z_near, float z_far)
{
    const float f = 1.0f / std::tan(fov_y * 0.5f);
    Mat4        p;
    for (float& v : p.m)
        v = 0.0f;
    p.at(0, 0) = f / aspect;
    p.at(1, 1) = -f;  // Vulkan's Y axis points down
    p.at(2, 2) = z_far / (z_near - z_far);
    p.at(2, 3) = z_near * z_far / (z_near - z_far);
    p.at(3, 2) = -1.0f;
    return p;
}

// Orthographic projection of the box x in [-half_width, half_width],
// y in [-half_height, half_height], view-space depth in [z_near, z_far].
inline Mat4 orthographic(float half_width, float half_height, float z_near, float z_far)
{
    Mat4 p;
    for (float& v : p.m)
        v = 0.0f;
    p.at(0, 0) = 1.0f / half_width;
    p.at(1, 1) = -1.0f / half_height;
    p.at(2, 2) = 1.0f / (z_near - z_far);
    p.at(2, 3) = z_near / (z_near - z_far);
    p.at(3, 3) = 1.0f;
    return p;
}

inline float radians(float degrees) { return degrees * 0.017453292519943295f; }
inline float degrees(float radians) { return radians * 57.29577951308232f; }

}  // namespace spectro
