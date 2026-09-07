#ifndef VIEWER_VIEW_MATRICES_H
#define VIEWER_VIEW_MATRICES_H

#include <cmath>
#include "camera.h"

// 4x4 row-major matrices in the ROW-VECTOR convention (p' = p · M)
// View space: x = cam.u, y = cam.v, z = −cam.w (forward)
// Clip space: D3D perspective, z in [0, 1]

static const float kViewNear = 0.01f, kViewFar = 1e4f;

inline void world_to_view(const camera& cam, float m[16]) {
    vec3 f = -cam.w;
    float t[16] = { (float)cam.u.x(), (float)cam.v.x(), (float)f.x(), 0,
                    (float)cam.u.y(), (float)cam.v.y(), (float)f.y(), 0,
                    (float)cam.u.z(), (float)cam.v.z(), (float)f.z(), 0,
                    -(float)dot(cam.center, cam.u), -(float)dot(cam.center, cam.v), -(float)dot(cam.center, f), 1 };
    for (int k = 0; k < 16; k++) m[k] = t[k];
}

inline void view_to_world(const camera& cam, float m[16]) {
    vec3 f = -cam.w;
    float t[16] = { (float)cam.u.x(), (float)cam.u.y(), (float)cam.u.z(), 0,
                    (float)cam.v.x(), (float)cam.v.y(), (float)cam.v.z(), 0,
                    (float)f.x(),     (float)f.y(),     (float)f.z(),     0,
                    (float)cam.center.x(), (float)cam.center.y(), (float)cam.center.z(), 1 };
    for (int k = 0; k < 16; k++) m[k] = t[k];
}

inline void view_to_clip(const camera& cam, float m[16]) {
    float f = 1.f / tanf((float)degrees_to_radians(cam.vfov) * 0.5f);
    float aspect = (float)cam.image_width / (float)cam.image_height;
    float zn = kViewNear, zf = kViewFar;
    float t[16] = { f / aspect, 0, 0, 0,
                    0, f, 0, 0,
                    0, 0, zf / (zf - zn), 1,
                    0, 0, -zn * zf / (zf - zn), 0 };
    for (int k = 0; k < 16; k++) m[k] = t[k];
}

inline void clip_to_view(const camera& cam, float m[16]) {   // the inverse of view_to_clip
    float f = 1.f / tanf((float)degrees_to_radians(cam.vfov) * 0.5f);
    float aspect = (float)cam.image_width / (float)cam.image_height;
    float zn = kViewNear, zf = kViewFar;
    float a = zf / (zf - zn), b = -zn * zf / (zf - zn);
    float t[16] = { aspect / f, 0, 0, 0,
                    0, 1 / f, 0, 0,
                    0, 0, 0, 1 / b,
                    0, 0, 1, -a / b };
    for (int k = 0; k < 16; k++) m[k] = t[k];
}

inline void mul4(const float a[16], const float b[16], float out[16]) {   // out = a · b
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a[4 * r + k] * b[4 * k + c];
            out[4 * r + c] = s;
        }
}

#endif // VIEWER_VIEW_MATRICES_H
