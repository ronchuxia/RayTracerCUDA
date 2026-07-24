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
// Collision is split into a shared NARROW PHASE (build_contacts: what overlaps,
// and each contact's normal + penetration) and a SOLVER (how to respond). Two
// solvers are compiled in, chosen by -DPHYS_SOLVER:
//   0           SPLIT       — the original one-pass positional split + equal-mass
//                             impulse (resolve_sphere_pair / _box), pairs relaxed
//                             twice. Cheap; jitters under crowding.
//   1 (default) SEQ_IMPULSE — sequential impulse: build the contact set once, then
//                             iterate accumulated normal impulses (Gauss-Seidel)
//                             for velocity + a projected position pass. Crowded
//                             piles settle instead of buzz. This is the seam the
//                             convex-convex (GJK/EPA) detector will emit into.
//
// MVP scope: spheres of equal mass (= 1) vs a ground plane at y=0, container
// walls, an optional static box, and each other. Convex hulls are Phase 3.

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

// ---- shape queries -------------------------------------------------------

// Sphere vs a STATIC axis-aligned box. The closest point on the box to the
// sphere centre is the centre CLAMPED into the box, one axis at a time. If that
// point is within the radius they touch; the contact normal is the direction
// from the closest point to the centre (works for face/edge/corner). The centre-
// inside case (deep penetration, e.g. shoved in by other balls) has no such
// normal, so eject along the least-penetrating face instead. Detection only:
// fills n (unit, box surface -> centre) and pen (>= 0); returns false if no hit.
inline bool sphere_box_contact(const phys_body& s, const vec3& bmin, const vec3& bmax,
                               vec3& n, real& pen) {
    vec3 q(clampr(s.pos[0], bmin[0], bmax[0]),     // closest point on the box to the centre
           clampr(s.pos[1], bmin[1], bmax[1]),
           clampr(s.pos[2], bmin[2], bmax[2]));
    vec3 d = s.pos - q;
    real dist2 = d.length_squared();
    if (dist2 > real(1e-12)) {                     // centre outside the box (common case)
        if (dist2 >= s.radius * s.radius) return false;  // gap wider than the radius
        real dist = std::sqrt(dist2);
        n = d / dist;                              // box surface -> centre
        pen = s.radius - dist;
    } else {                                       // centre inside: eject along nearest face
        real ex = s.pos[0]-bmin[0] < bmax[0]-s.pos[0] ? -(s.pos[0]-bmin[0]) : (bmax[0]-s.pos[0]);
        real ey = s.pos[1]-bmin[1] < bmax[1]-s.pos[1] ? -(s.pos[1]-bmin[1]) : (bmax[1]-s.pos[1]);
        real ez = s.pos[2]-bmin[2] < bmax[2]-s.pos[2] ? -(s.pos[2]-bmin[2]) : (bmax[2]-s.pos[2]);
        real ax = ex<0?-ex:ex, ay = ey<0?-ey:ey, az = ez<0?-ez:ez;  // exit distances
        if (ax <= ay && ax <= az) { n = vec3(ex<0?real(-1):real(1), 0, 0); pen = s.radius + ax; }
        else if (ay <= az)        { n = vec3(0, ey<0?real(-1):real(1), 0); pen = s.radius + ay; }
        else                      { n = vec3(0, 0, ez<0?real(-1):real(1)); pen = s.radius + az; }
    }
    return true;
}

// ---- SPLIT solver primitives (the original fused impulse solver) ----------
// Kept intact: the tests call these directly, and solve_split() is built from
// them. Each both detects AND resolves in one shot (position + velocity).

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

// Sphere vs static box: push out of penetration, then reflect the inward
// component of velocity (restitution). Mirrors the ground bounce with a general
// normal from sphere_box_contact().
inline void resolve_sphere_box(phys_body& s, const vec3& bmin, const vec3& bmax, real e) {
    vec3 n; real pen;
    if (!sphere_box_contact(s, bmin, bmax, n, pen)) return;
    s.pos += n * pen;                             // push out along the contact normal
    real vn = dot(s.vel, n);
    if (vn < 0) s.vel = s.vel - n * ((real(1) + e) * vn);  // reflect the inward component
}

// The SPLIT solver: static colliders (ground / walls / box) resolved once each,
// then sphere-sphere relaxed twice. (Gravity + integrate + friction live in
// physics_step; this is only the collision response.)
inline void solve_split(std::vector<phys_body>& bodies, const phys_params& p) {
    for (phys_body& b : bodies) {
        if (b.pos[1] - b.radius <= 0) {           // ground plane y=0
            b.pos[1] = b.radius;
            b.vel[1] = -p.restitution * b.vel[1];
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
        if (p.has_box) resolve_sphere_box(b, p.box_min, p.box_max, p.restitution);
    }
    for (int it = 0; it < 2; it++)
        for (std::size_t i = 0; i < bodies.size(); i++)
            for (std::size_t j = i + 1; j < bodies.size(); j++)
                resolve_sphere_pair(bodies[i], bodies[j], p.restitution);
}

// ---- contact abstraction + SEQUENTIAL-IMPULSE solver ---------------------

// One overlapping contact. b < 0 => body a vs a static collider (ground / wall /
// box); otherwise dynamic bodies a and b. n is unit, pointing from b (or the
// static surface) toward a; pen >= 0.
struct contact {
    int  a, b;
    vec3 n;
    real pen;
};

// Shared NARROW PHASE: emit every overlapping contact (ground, walls, box, and
// sphere pairs) from the current body positions. This is the seam Phase 3's
// convex-convex (GJK/EPA) detector will emit contacts into — the solver below
// does not care what shape produced a contact.
inline void build_contacts(const std::vector<phys_body>& bodies, const phys_params& p,
                           std::vector<contact>& out) {
    out.clear();
    for (std::size_t i = 0; i < bodies.size(); i++) {
        const phys_body& b = bodies[i];
        if (b.pos[1] - b.radius < 0)              // ground plane y=0
            out.push_back({ (int)i, -1, vec3(0, 1, 0), b.radius - b.pos[1] });
        for (int k = 0; k <= 2; k += 2) {         // container walls on x (0) and z (2)
            if (b.pos[k] - b.radius < p.wall_min[k]) {
                vec3 n(0, 0, 0); n[k] = real(1);
                out.push_back({ (int)i, -1, n, p.wall_min[k] - (b.pos[k] - b.radius) });
            }
            if (b.pos[k] + b.radius > p.wall_max[k]) {
                vec3 n(0, 0, 0); n[k] = real(-1);
                out.push_back({ (int)i, -1, n, (b.pos[k] + b.radius) - p.wall_max[k] });
            }
        }
        if (p.has_box) {                          // static box obstacle
            vec3 n; real pen;
            if (sphere_box_contact(b, p.box_min, p.box_max, n, pen))
                out.push_back({ (int)i, -1, n, pen });
        }
    }
    for (std::size_t i = 0; i < bodies.size(); i++)   // sphere-sphere
        for (std::size_t j = i + 1; j < bodies.size(); j++) {
            vec3 d = bodies[i].pos - bodies[j].pos;
            real dist2 = d.length_squared();
            real rsum = bodies[i].radius + bodies[j].radius;
            if (dist2 < rsum * rsum && dist2 >= real(1e-12)) {
                real dist = std::sqrt(dist2);
                out.push_back({ (int)i, (int)j, d / dist, rsum - dist });
            }
        }
}

// Sequential-impulse solver tunables.
static constexpr int  SEQ_VEL_ITERS = 8;            // velocity (impulse) iterations
static constexpr int  SEQ_POS_ITERS = 4;            // position-correction iterations
static constexpr real SEQ_POS_BETA  = real(0.8);    // fraction of penetration corrected per pass
static constexpr real SEQ_POS_SLOP  = real(1e-4);   // penetration left uncorrected (kills jitter)
static constexpr real SEQ_REST_VEL  = real(0.5);    // approach speed below which we don't bounce

// The SEQUENTIAL-IMPULSE solver: build the contact set ONCE, then
//   (1) VELOCITY — iterate accumulated normal impulses (Gauss-Seidel) so
//       contacts share load instead of fighting; the clamp jn >= 0 keeps a
//       contact from ever pulling. Restitution comes from each contact's initial
//       approach speed, and slow/resting contacts get target 0 so they don't
//       buzz.
//   (2) POSITION — a projected Gauss-Seidel pass, re-detecting penetration each
//       iteration, that pushes remaining overlap out (leaving a small slop).
// Separating velocity from position and iterating both is what lets a crowded
// pile settle where the SPLIT solver's 2 pair-relaxes buzz.
inline void solve_sequential(std::vector<phys_body>& bodies, const phys_params& p) {
    std::vector<contact> C;
    build_contacts(bodies, p, C);
    const std::size_t n = C.size();

    // Restitution target per contact, from the initial approach speed.
    std::vector<real> vbias(n), jn(n, real(0));
    for (std::size_t c = 0; c < n; c++) {
        const contact& k = C[c];
        real vn = k.b < 0 ? dot(bodies[k.a].vel, k.n)
                          : dot(bodies[k.a].vel - bodies[k.b].vel, k.n);
        vbias[c] = vn < -SEQ_REST_VEL ? -p.restitution * vn : real(0);
    }
    // (1) velocity: accumulated normal impulses, clamped non-negative.
    for (int it = 0; it < SEQ_VEL_ITERS; it++)
        for (std::size_t c = 0; c < n; c++) {
            const contact& k = C[c];
            real invSum = k.b < 0 ? real(1) : real(2);   // 1/m_a (+ 1/m_b), m = 1
            real vn = k.b < 0 ? dot(bodies[k.a].vel, k.n)
                              : dot(bodies[k.a].vel - bodies[k.b].vel, k.n);
            real dJ = (vbias[c] - vn) / invSum;
            real jn_new = jn[c] + dJ; if (jn_new < 0) jn_new = real(0);  // no sticking
            vec3 imp = k.n * (jn_new - jn[c]);
            jn[c] = jn_new;
            bodies[k.a].vel += imp;
            if (k.b >= 0) bodies[k.b].vel = bodies[k.b].vel - imp;
        }
    // (2) position: projected Gauss-Seidel, re-detecting penetration each pass.
    for (int it = 0; it < SEQ_POS_ITERS; it++) {
        build_contacts(bodies, p, C);
        for (std::size_t c = 0; c < C.size(); c++) {
            const contact& k = C[c];
            real corr = (k.pen - SEQ_POS_SLOP) * SEQ_POS_BETA;
            if (corr <= 0) continue;
            if (k.b < 0) bodies[k.a].pos += k.n * corr;  // static: full push-out
            else {                                       // pair: split the correction
                vec3 h = k.n * (corr * real(0.5));
                bodies[k.a].pos += h;
                bodies[k.b].pos = bodies[k.b].pos - h;
            }
        }
    }
}

// Advance ALL bodies one fixed step `dt`: semi-implicit Euler, then the selected
// collision solver, then tangential ground friction. Returns the max |vel| after
// the step, for the caller's sleep policy.
inline real physics_step(std::vector<phys_body>& bodies, const phys_params& p, real dt) {
    for (phys_body& b : bodies) {                 // gravity (y only) + integrate
        b.vel[1] += p.gravity * dt;
        b.pos    += b.vel * dt;
    }
#if defined(PHYS_SOLVER) && PHYS_SOLVER == 0
    solve_split(bodies, p);
#else
    solve_sequential(bodies, p);
#endif
    for (phys_body& b : bodies)                   // grounded: tangential friction
        if (b.pos[1] <= b.radius + real(1e-3)) {
            b.vel[0] *= p.ground_friction;
            b.vel[2] *= p.ground_friction;
        }
    real maxv = 0;
    for (phys_body& b : bodies) {
        real v = b.vel.length();
        if (v > maxv) maxv = v;
    }
    return maxv;
}

#endif // PHYSICS_H
