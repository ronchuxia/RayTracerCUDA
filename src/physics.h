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
};

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

// Advance ALL bodies one fixed step `dt`: semi-implicit Euler + ground-plane
// (y=0) bounce + tangential ground friction, then sphere-sphere collisions
// (brute-force pairs, relaxed 2 iterations). Returns the max |vel| after the
// step, so the caller can drive its sleep policy.
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
