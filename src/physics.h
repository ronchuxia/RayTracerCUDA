#ifndef PHYSICS_H
#define PHYSICS_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "precision.h"   // real
#include "vec3.h"

// Host-side rigid-body simulation (workstream D). This is the pacing-agnostic
// INTEGRATOR: `physics_step` advances every body one fixed timestep. The DRIVER
// (how often it's stepped — the viewer's wall-clock accumulator) and the
// COUPLING (writing poses back through the scene's mutation protocol) stay with
// the caller. See docs/plans/physics.md and docs/timestep-and-pacing.md.
//
// MVP scope: spheres of equal mass (= 1) colliding with a ground plane at y=0
// and with each other. Box/hull colliders + GJK/EPA are Phase 3.

// A simulated body. pos/vel/radius are the physics state the step touches; id +
// baseR/baseS are caller bookkeeping (the scene-object id and the transform's
// rotation/scale) so the caller can rebuild the pose as
// transform(child, pos, baseR, baseS). The step never reads them.
struct phys_body {
    int  id;
    vec3 pos, vel;
    real radius;
    vec3 baseR, baseS;
};

struct phys_params {
    real gravity;          // world units / s^2 (negative = down)
    real restitution;      // bounce energy retained (0..1)
    real ground_friction;  // tangential velocity retained per grounded step
    vec3 wall_min, wall_max;  // axis-aligned container walls on x/z (y unused; the
                              // floor is the ground plane). Use +/- huge for "no walls".
    bool has_box;             // one static axis-aligned box obstacle to collide against
    vec3 box_min, box_max;
};

static inline real clampr(real x, real lo, real hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Sphere-sphere collision: if overlapping, split the penetration equally and
// apply an equal-mass restitution impulse along the contact normal (only when
// the pair is approaching, so separating pairs don't stick).
inline void resolve_sphere_pair(phys_body& a, phys_body& b, real e) {
    vec3 d = a.pos - b.pos;                        // b -> a
    real dist2 = d.length_squared();
    real rsum = a.radius + b.radius;
    if (dist2 >= rsum * rsum || dist2 < real(1e-12)) return;
    real dist = std::sqrt(dist2);
    vec3 n = d / dist;
    real pen = rsum - dist;
    vec3 half = n * (pen * real(0.5));
    a.pos += half;                                // positional correction (split)
    b.pos = b.pos - half;
    real vrel = dot(a.vel - b.vel, n);
    if (vrel < 0) {                               // approaching -> impulse
        real j = -(real(1) + e) * vrel / real(2); // (1/m_a + 1/m_b) = 2 at m = 1
        vec3 imp = n * j;
        a.vel += imp;
        b.vel = b.vel - imp;
    }
}

// Sphere vs a STATIC axis-aligned box. The key idea: the closest point on the
// box to the sphere centre is the centre CLAMPED into the box, one axis at a
// time. If that point is within the radius, they touch; the contact normal is
// the direction from the closest point to the centre. Resolution mirrors the
// ground bounce, just with a general normal: push out of penetration, then
// reflect the inward component of velocity (restitution). The centre-inside-the-
// box case (deep penetration, e.g. shoved in by other balls) has no such normal,
// so eject along the least-penetrating face instead.
inline void resolve_sphere_box(phys_body& s, const vec3& bmin, const vec3& bmax, real e) {
    vec3 q(clampr(s.pos[0], bmin[0], bmax[0]),    // closest point on the box to the centre
           clampr(s.pos[1], bmin[1], bmax[1]),
           clampr(s.pos[2], bmin[2], bmax[2]));
    vec3 d = s.pos - q;
    real dist2 = d.length_squared();

    vec3 n; real pen;
    if (dist2 > real(1e-12)) {                    // centre outside the box (common case)
        if (dist2 >= s.radius * s.radius) return; // gap wider than the radius -> no contact
        real dist = std::sqrt(dist2);
        n = d / dist;                             // box surface -> centre
        pen = s.radius - dist;
    } else {                                      // centre inside: eject along nearest face
        real ex = s.pos[0] - bmin[0] < bmax[0] - s.pos[0] ? -(s.pos[0]-bmin[0]) : (bmax[0]-s.pos[0]);
        real ey = s.pos[1] - bmin[1] < bmax[1] - s.pos[1] ? -(s.pos[1]-bmin[1]) : (bmax[1]-s.pos[1]);
        real ez = s.pos[2] - bmin[2] < bmax[2] - s.pos[2] ? -(s.pos[2]-bmin[2]) : (bmax[2]-s.pos[2]);
        real ax = ex<0?-ex:ex, ay = ey<0?-ey:ey, az = ez<0?-ez:ez;  // exit distances
        if (ax <= ay && ax <= az)      { n = vec3(ex<0?real(-1):real(1), 0, 0); pen = s.radius + ax; }
        else if (ay <= az)             { n = vec3(0, ey<0?real(-1):real(1), 0); pen = s.radius + ay; }
        else                           { n = vec3(0, 0, ez<0?real(-1):real(1)); pen = s.radius + az; }
    }
    s.pos += n * pen;                             // push out along the contact normal
    real vn = dot(s.vel, n);
    if (vn < 0) s.vel = s.vel - n * ((real(1) + e) * vn);  // reflect the inward component
}

// Advance ALL bodies one fixed step `dt`: semi-implicit Euler + ground-plane
// (y=0) bounce + tangential ground friction + container walls + an optional
// static box obstacle, then sphere-sphere collisions (brute-force pairs, relaxed
// 2 iterations). Returns the max |vel| after the step, for the caller's sleep policy.
inline real physics_step(std::vector<phys_body>& bodies, const phys_params& p, real dt) {
    for (phys_body& b : bodies) {
        b.vel[1] += p.gravity * dt;                // gravity (y only)
        b.pos    += b.vel * dt;                    // integrate with the new velocity
        if (b.pos[1] - b.radius <= 0) {           // ground plane
            b.pos[1] = b.radius;
            b.vel[1] = -p.restitution * b.vel[1];
        }
        if (b.pos[1] <= b.radius + real(1e-3)) {  // grounded: tangential friction
            b.vel[0] *= p.ground_friction;
            b.vel[2] *= p.ground_friction;
        }
        for (int k = 0; k <= 2; k += 2) {         // container walls on x (0) and z (2)
            if (b.pos[k] - b.radius < p.wall_min[k]) {
                b.pos[k] = p.wall_min[k] + b.radius;
                b.vel[k] = -p.restitution * b.vel[k];
            }
            if (b.pos[k] + b.radius > p.wall_max[k]) {
                b.pos[k] = p.wall_max[k] - b.radius;
                b.vel[k] = -p.restitution * b.vel[k];
            }
        }
        if (p.has_box) resolve_sphere_box(b, p.box_min, p.box_max, p.restitution);  // static obstacle
    }
    for (int it = 0; it < 2; it++)
        for (std::size_t i = 0; i < bodies.size(); i++)
            for (std::size_t j = i + 1; j < bodies.size(); j++)
                resolve_sphere_pair(bodies[i], bodies[j], p.restitution);

    real maxv = 0;
    for (phys_body& b : bodies) {
        real v = b.vel.length();
        if (v > maxv) maxv = v;
    }
    return maxv;
}

#endif // PHYSICS_H
