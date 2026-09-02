#ifndef PHYSICS_DYNAMICS_H
#define PHYSICS_DYNAMICS_H

#include <cmath>

#include "physics/body.h"
#include "precision.h"   // real
#include "vec3.h"

inline real inv_mass(const phys_body& b) {
    return b.motion == DYNAMIC ? real(1) / b.mass : real(0);
}

// change in angular velocity of body `b` due to angular impulse `L`
inline vec3 delta_omega(const phys_body& b, const vec3& L) {
    const real im = inv_mass(b);
    if (im <= real(0)) return vec3(0, 0, 0);
    // sphere
    if (b.shape == COLLIDER_SPHERE) {
        return b.radius > real(0) ? L * (real(2.5) * im / (b.radius * b.radius)) : vec3(0, 0, 0);
    }
    // box
    const vec3& h = b.half;
    const real d0 = h[1]*h[1] + h[2]*h[2];
    const real d1 = h[0]*h[0] + h[2]*h[2];
    const real d2 = h[0]*h[0] + h[1]*h[1];
    const real i0 = d0 > real(0) ? real(3) * im / d0 : real(0);
    const real i1 = d1 > real(0) ? real(3) * im / d1 : real(0);
    const real i2 = d2 > real(0) ? real(3) * im / d2 : real(0);
    const vec3 l_local(dot(L, b.axes[0]), dot(L, b.axes[1]), dot(L, b.axes[2]));
    return b.axes[0] * (l_local[0] * i0)
         + b.axes[1] * (l_local[1] * i1)
         + b.axes[2] * (l_local[2] * i2);
}

// change in relative velocity along `dir` at levers `ra`, `rb` due to unit impulse along `dir`
inline real inv_effective_mass(const phys_body& A, const phys_body& B,
                           const vec3& ra, const vec3& rb, const vec3& dir) {
    return inv_mass(A) + inv_mass(B)    // linear term
         + dot(dir, cross(delta_omega(A, cross(ra, dir)), ra))    // angular term
         + dot(dir, cross(delta_omega(B, cross(rb, dir)), rb));
}

// lever arm from body's center to contact point
inline vec3 contact_lever(const phys_body& b, const vec3& n, const vec3& p, bool is_a) {
    if (b.shape == COLLIDER_SPHERE) return is_a ? -n * b.radius : n * b.radius;
    return p - b.pos;
}

// velocity of point of body `b` at lever arm `r`
inline vec3 velocity_at(const phys_body& b, const vec3& r) {
    return b.vel + cross(b.omega, r);
}

// how a contact's coefficient is derived from that of two surfaces
enum combine_mode { COMBINE_MULTIPLY, COMBINE_MIN, COMBINE_GEOMETRIC,
                    COMBINE_AVERAGE, COMBINE_MAX };

inline real combine(real a, real b, combine_mode m) {
    switch (m) {
        case COMBINE_MULTIPLY:  return a * b;
        case COMBINE_MIN:       return a < b ? a : b;
        case COMBINE_AVERAGE:   return (a + b) * real(0.5);
        case COMBINE_MAX:       return a > b ? a : b;
        case COMBINE_GEOMETRIC:
        default:                { real p = a * b; return p > real(0) ? std::sqrt(p) : real(0); }
    }
}

// world-level simulation settings
struct phys_params {
    real gravity;
    combine_mode friction_combine    = COMBINE_AVERAGE;
    combine_mode restitution_combine = COMBINE_AVERAGE;
};

#endif // PHYSICS_DYNAMICS_H
