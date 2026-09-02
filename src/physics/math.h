#ifndef PHYSICS_MATH_H
#define PHYSICS_MATH_H

#include <cmath>

#include "precision.h"   // real
#include "vec3.h"

static inline real clampr(real x, real lo, real hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

inline vec3 any_perpendicular(const vec3& n) {
    vec3 a = (n[0] < real(0.9) && n[0] > real(-0.9)) ? vec3(1, 0, 0) : vec3(0, 1, 0);
    vec3 t = cross(n, a);
    return t / t.length();
}

inline vec3 tangent_from(const vec3& v, const vec3& n) {
    vec3 t = v - n * dot(v, n);
    real tl = t.length();
    if (tl <= real(1e-4) * v.length()) return any_perpendicular(n);
    t = t / tl;
    // re-project and re-normalzie to increase numerical precision
    t = t - n * dot(t, n);
    tl = t.length();
    t = t / tl;
    return t;
}

#endif // PHYSICS_MATH_H
