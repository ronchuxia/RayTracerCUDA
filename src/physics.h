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
// EVERYTHING THAT COLLIDES IS A BODY. The ground, the container walls and any
// obstacle are `phys_body`s with STATIC motion and a sphere/box collider, exactly
// like the falling spheres — there is no separate notion of "world geometry".
// That is what lets one code path cover every case: a contact is always a pair
// of real body indices, immovability is always just inv_mass() == 0, and a
// MOVING kinematic surface transfers its velocity for free (an implicit
// zero-velocity world could not express that).
//
// Collision is split into a NARROW PHASE (build_contacts: which pairs overlap,
// and each contact's normal + penetration, dispatched by collider shape) and a
// SOLVER (how to respond). The solver is SEQUENTIAL IMPULSE: build the contact
// set once, then iterate accumulated normal + friction impulses (Gauss-Seidel)
// for velocity, followed by a projected position pass. Crowded piles settle
// instead of buzzing. This is the seam the convex-convex (GJK/EPA) detector
// will emit into.
//
// Every body carries a physics ROLE (`motion` + `collidable`), a `mass` and a
// surface `friction`, which the solver reads only through inv_mass() and
// combine_friction(), so STATIC / KINEMATIC / DYNAMIC and any mass ratio all
// share one code path. See docs/plans/physics.md (Phase 3).
//
// Current scope: SPHERE colliders against boxes and each other. Boxes are
// ORIENTED (they carry their own axes), so a rotated box collides as itself.
// Box-box and angular dynamics are Phase 3B (GJK/EPA + quaternions).
//
// THERE IS NO PLANE COLLIDER. Every collider is read from the scene object it is
// attached to, and no object is an infinite plane — a bounded surface is a box
// (the ball pit's walls are its own wall quads) and an unbounded one is whatever
// large shape actually renders (its floor is a radius-1000 sphere, and collides
// as that sphere). So a collider can never disagree with the thing you see.

// How a body moves — the physics ROLE (with `collidable`, the 2-field model
// USD/PhysX use). The solver never switches on this directly; it reads the
// derived inv_mass() below, which is what makes one code path cover all three.
// Ordered by how much of the body the sim owns, least to most:
//   STATIC     never moves. Infinite mass. Ground, walls, fixed obstacles.
//   KINEMATIC  moves under EXTERNAL control (a viewer drag, an animation curve).
//              Infinite mass, so it PUSHES dynamic bodies but is never pushed
//              back — the integrator leaves its pose to the driver.
//   DYNAMIC    moves under gravity + contact impulses. Finite mass.
// STATIC is deliberately 0, so a zero-initialised body is inert rather than
// live; the viewer's motion combo indexes this enum directly, so its label
// array must stay in this order.
enum motion_type { STATIC, KINEMATIC, DYNAMIC };

// What a body collides AS, independent of what it renders as. Prefixed because
// a bare SPHERE/BOX would collide with hittable.h's HittableType at global
// scope in the viewer, which includes both headers.
enum collider_type { COLLIDER_SPHERE, COLLIDER_BOX };

// A simulated body. pos/vel + the collider fields are the physics state the step
// touches; scene_id + baseR/baseS are caller bookkeeping (which scene object
// this body drives, and that object's rotation/scale) so the caller can rebuild
// the pose as transform(child, pos, baseR, baseS). scene_id < 0 means UNLINKED —
// no render counterpart — and the step never reads any of the three.
//
// It is named scene_id, not id, because a body is addressed two different ways
// and both are plain ints: `contact.a`/`contact.b` are indices INTO the body
// vector, while this is an id in the SCENE's object list. They are not
// interchangeable and nothing in the type system says so.
//
// The collider parameters sit FLAT rather than in a tagged union: this is
// host-side code over a handful of bodies, so readability beats the few dead
// bytes a sphere carries for the box fields, and appending them keeps
// every existing aggregate initialiser valid.
struct phys_body {
    int  scene_id;
    vec3 pos, vel;
    real radius;                          // COLLIDER_SPHERE
    vec3 baseR, baseS;
    motion_type   motion     = DYNAMIC;   // role: how it moves (see motion_type)
    bool          collidable = true;      // orthogonal: does it collide at all?
    real          mass       = real(1);   // authored; MUST be > 0. Only DYNAMIC uses it.
    real          friction   = real(0.5); // surface property; combined per contact
    real          restitution= real(0.7); // surface property; combined per contact
    collider_type shape      = COLLIDER_SPHERE;
    vec3          half       = vec3(0, 0, 0);  // COLLIDER_BOX half-extents about pos
    // COLLIDER_BOX orientation: the box's own x/y/z axes in world space (the
    // columns of its rotation matrix — unit and mutually perpendicular).
    // Identity means axis-aligned. Stored as axes rather than derived from
    // baseR because the Euler-angle convention lives in transforms.h, which
    // physics.h deliberately does not include: the caller converts once, so the
    // collider and the rendered pose cannot drift apart. B3's quaternion
    // becomes the authoritative orientation and these are derived from it.
    vec3          axes[3]    = { vec3(1,0,0), vec3(0,1,0), vec3(0,0,1) };
};

// Inverse mass — the ONLY channel through which a role reaches the solver: a
// heavier body simply takes less of each contact impulse, and an immovable one
// (STATIC / KINEMATIC) takes none, dropping out of the contact's effective mass
// entirely.
//
// The ROLE ALWAYS WINS over the authored mass, and this is DERIVED rather than
// stored precisely so that it cannot go stale: a body whose motion changes at
// runtime (the viewer dragging a box to KINEMATIC) can never be left behind
// with a finite mass. Authoring keeps `mass` because that is the meaningful UI
// quantity; switching a body back to DYNAMIC restores the mass it was given.
inline real inv_mass(const phys_body& b) {
    return b.motion == DYNAMIC ? real(1) / b.mass : real(0);
}

// How a CONTACT's coefficient is derived from the two surfaces that meet there.
// Friction and restitution are both pair properties in reality — "the friction
// of rubber" is meaningless, only "rubber on steel" is — so any rule that
// synthesises one from two per-object numbers is an approximation, which is why
// this is a choice rather than a law. Engines differ: Bullet multiplies both,
// PhysX defaults to average with min/multiply/max selectable per material,
// MuJoCo takes the max for friction and has no restitution at all.
//
// Listed in order of increasing result: for coefficients in [0,1],
//     a*b  <=  min  <=  sqrt(a*b)  <=  average  <=  max
// so the dropdown reads from deadest/slipperiest to bounciest/grippiest. The
// viewer's combo indexes this enum directly, so its label array must match.
enum combine_mode { COMBINE_MULTIPLY, COMBINE_MIN, COMBINE_GEOMETRIC,
                    COMBINE_AVERAGE, COMBINE_MAX };

// Three properties separate these, and they are what the choice turns on:
//   IDENTITY WHEN EQUAL — equal surfaces resolve to that value, so a uniform
//     scene behaves exactly as authored. All but MULTIPLY have it (0.7*0.7=0.49).
//   ABSORBING ZERO — a 0 on either side forces 0, so a genuinely dead/slippery
//     object can be authored. MULTIPLY, MIN and GEOMETRIC have it.
//   BOTH DIRECTIONS — responds to "make this one bouncier" AND "make this one
//     deader". MAX ignores the second, MIN the first.
// GEOMETRIC is the only rule with all three (it is MULTIPLY renormalised so
// equal surfaces round-trip). We nonetheless DEFAULT TO AVERAGE, which is
// PhysX's default, so our behaviour lines up with the engine most people will
// compare against. The trade is that AVERAGE has no absorbing zero: a surface
// authored at 0 still bounces (or grips) at half its partner's value, so a
// scene that needs a genuinely dead or frictionless object should select
// GEOMETRIC in the viewer's Physics panel. Note both rules agree wherever the
// two surfaces match, so a uniform scene is unaffected by the choice.
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

// World-level simulation settings. Everything here is a property of the SIM;
// per-surface properties (mass, friction, restitution) live on the body, which
// is why restitution is no longer here.
struct phys_params {
    real gravity;                                          // world units / s^2 (negative = down)
    combine_mode friction_combine    = COMBINE_AVERAGE;    // PhysX's default
    combine_mode restitution_combine = COMBINE_AVERAGE;
};

static inline real clampr(real x, real lo, real hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Any unit vector perpendicular to n — the fallback tangent for a contact with
// no sliding yet (the direction is arbitrary; the friction clamp is symmetric).
inline vec3 any_perpendicular(const vec3& n) {
    vec3 a = (n[0] < real(0.9) && n[0] > real(-0.9)) ? vec3(1, 0, 0) : vec3(0, 1, 0);
    vec3 t = cross(n, a);
    return t / t.length();
}

// ---- shape queries -------------------------------------------------------

// Sphere vs an axis-aligned box. The closest point on the box to the sphere
// centre is the centre CLAMPED into the box, one axis at a time. If that point
// is within the radius they touch; the contact normal is the direction from the
// closest point to the centre (works for face/edge/corner). The centre-inside
// case (deep penetration, e.g. shoved in by other balls) has no such normal, so
// eject along the least-penetrating face instead. Detection only: fills n (unit,
// box surface -> centre) and pen (>= 0); returns false if no hit.
//
// Frame-agnostic: it reads only the coordinates it is handed, so an ORIENTED box
// reuses it by passing the sphere centre in the box's own frame (contact_between
// below does exactly that, then rotates n back to world space).
inline bool sphere_box_contact(const vec3& c, real r, const vec3& bmin, const vec3& bmax,
                               vec3& n, real& pen) {
    vec3 q(clampr(c[0], bmin[0], bmax[0]),         // closest point on the box to the centre
           clampr(c[1], bmin[1], bmax[1]),
           clampr(c[2], bmin[2], bmax[2]));
    vec3 d = c - q;
    real dist2 = d.length_squared();
    if (dist2 > real(1e-12)) {                     // centre outside the box (common case)
        if (dist2 >= r * r) return false;          // gap wider than the radius
        real dist = std::sqrt(dist2);
        n = d / dist;                              // box surface -> centre
        pen = r - dist;
    } else {                                       // centre inside: eject along nearest face
        real ex = c[0]-bmin[0] < bmax[0]-c[0] ? -(c[0]-bmin[0]) : (bmax[0]-c[0]);
        real ey = c[1]-bmin[1] < bmax[1]-c[1] ? -(c[1]-bmin[1]) : (bmax[1]-c[1]);
        real ez = c[2]-bmin[2] < bmax[2]-c[2] ? -(c[2]-bmin[2]) : (bmax[2]-c[2]);
        real ax = ex<0?-ex:ex, ay = ey<0?-ey:ey, az = ez<0?-ez:ez;  // exit distances
        if (ax <= ay && ax <= az) { n = vec3(ex<0?real(-1):real(1), 0, 0); pen = r + ax; }
        else if (ay <= az)        { n = vec3(0, ey<0?real(-1):real(1), 0); pen = r + ay; }
        else                      { n = vec3(0, 0, ez<0?real(-1):real(1)); pen = r + az; }
    }
    return true;
}

// ---- contact abstraction + SEQUENTIAL-IMPULSE solver ---------------------

// One overlapping contact between bodies a and b. n is unit and points from b
// toward a; pen >= 0. BOTH indices are always valid body indices — the ground
// and the walls are bodies too, so there is no sentinel. The orientation of n is
// the only asymmetry left, and it is intrinsic: contact(a,b,n) and
// contact(b,a,-n) describe the same collision, but a solver has to be told
// which way "apart" is.
struct contact {
    int  a, b;
    vec3 n;
    real pen;
};

// Narrow phase for ONE ORDERED pair: fills n (from B toward A) and pen. Every
// implemented test is sphere-vs-X, so build_contacts canonicalises the sphere
// into A before calling. Box-box returns false until B2's GJK/EPA lands.
inline bool contact_between(const phys_body& A, const phys_body& B, vec3& n, real& pen) {
    if (A.shape != COLLIDER_SPHERE) return false;

    if (B.shape == COLLIDER_SPHERE) {              // sphere vs sphere
        vec3 d = A.pos - B.pos;
        real dist2 = d.length_squared();
        real rsum = A.radius + B.radius;
        if (dist2 >= rsum * rsum || dist2 < real(1e-12)) return false;
        real dist = std::sqrt(dist2);
        n = d / dist;
        pen = rsum - dist;
        return true;
    }
    // Sphere vs ORIENTED box: move the sphere centre into the box's own frame
    // (project the separation onto each box axis), run the axis-aligned test
    // there against +/-half, then rotate the normal back out. The box's axes
    // absorb the rotation, so no separate oriented-box test is needed — the same
    // move B2's support functions generalise to every convex shape. For an
    // axis-aligned box the axes are identity and this reduces to the plain test,
    // now expressed relative to the box centre.
    vec3 d = A.pos - B.pos;
    vec3 c_local(dot(d, B.axes[0]), dot(d, B.axes[1]), dot(d, B.axes[2]));
    vec3 n_local;
    if (!sphere_box_contact(c_local, A.radius, -B.half, B.half, n_local, pen)) return false;
    n = n_local.x() * B.axes[0] + n_local.y() * B.axes[1] + n_local.z() * B.axes[2];
    return true;   // unit: n_local is unit and the axes are orthonormal
}

// Shared NARROW PHASE: every overlapping pair, as a contact. This is the seam
// Phase 3B's convex-convex (GJK/EPA) detector will emit into — the solver below
// does not care what shape produced a contact.
//
// ROLES filter here rather than in the solver: a non-collidable body has no
// contacts at all, and a pair that cannot move EITHER side is never worth
// emitting. Movability is tested through inv_mass() with the same `<= 0`
// comparison the solver applies to invSum, so the narrow phase and the solver
// can never disagree about what is resolvable.
//
// Brute-force O(N^2) over bodies; the world BVH becomes the broad phase when
// body count warrants it (Phase 3B).
inline void build_contacts(const std::vector<phys_body>& bodies, std::vector<contact>& out) {
    out.clear();
    for (std::size_t i = 0; i < bodies.size(); i++)
        for (std::size_t j = i + 1; j < bodies.size(); j++) {
            if (!bodies[i].collidable || !bodies[j].collidable) continue;
            if (inv_mass(bodies[i]) + inv_mass(bodies[j]) <= real(0)) continue;  // both immovable
            // Canonicalise so the sphere is the first operand (see contact_between).
            int a = (int)i, b = (int)j;
            if (bodies[a].shape != COLLIDER_SPHERE && bodies[b].shape == COLLIDER_SPHERE)
                { a = (int)j; b = (int)i; }
            vec3 n; real pen;
            if (contact_between(bodies[a], bodies[b], n, pen))
                out.push_back({ a, b, n, pen });
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
//       buzz. Each contact also gets a FRICTION impulse opposing sliding,
//       clamped to the Coulomb cone |jt| <= mu*jn against that contact's own
//       accumulated normal impulse — so friction follows contact by
//       construction, and a body touching nothing is never damped.
//   (2) POSITION — a projected Gauss-Seidel pass, re-detecting penetration each
//       iteration, that pushes remaining overlap out (leaving a small slop),
//       shared by inverse mass so the movable side takes its share.
// Separating velocity from position and iterating both is what lets a crowded
// pile settle.
inline void solve_sequential(std::vector<phys_body>& bodies, const phys_params& p) {
    std::vector<contact> C;
    build_contacts(bodies, C);
    const std::size_t n = C.size();

    std::vector<real> vbias(n), jn(n, real(0)), jt(n, real(0)), mu(n);
    std::vector<vec3> tang(n);
    for (std::size_t c = 0; c < n; c++) {
        const contact& k = C[c];
        vec3 vrel = bodies[k.a].vel - bodies[k.b].vel;
        real vn = dot(vrel, k.n);
        real e   = combine(bodies[k.a].restitution, bodies[k.b].restitution, p.restitution_combine);
        vbias[c] = vn < -SEQ_REST_VEL ? -e * vn : real(0);
        mu[c]    = combine(bodies[k.a].friction, bodies[k.b].friction, p.friction_combine);
        // Tangent is fixed for the whole step, taken from the sliding direction
        // at its start: accumulating jt along an axis that rotated every
        // iteration would make the cone clamp meaningless.
        vec3 vt  = vrel - k.n * vn;
        real vtl = vt.length();
        tang[c]  = vtl > real(1e-8) ? vt / vtl : any_perpendicular(k.n);
    }

    // (1) velocity: accumulated normal + friction impulses. Mass enters ONLY as
    // inverse mass, so an immovable body contributes 0 to the effective mass and
    // absorbs no impulse (imp * 0) — that one weighting is what makes STATIC and
    // KINEMATIC bodies "free" to push against.
    for (int it = 0; it < SEQ_VEL_ITERS; it++)
        for (std::size_t c = 0; c < n; c++) {
            const contact& k = C[c];
            real ima = inv_mass(bodies[k.a]);
            real imb = inv_mass(bodies[k.b]);
            real invSum = ima + imb;                    // 1/m_a + 1/m_b along n
            if (invSum <= real(0)) continue;            // nothing here can move

            real vn = dot(bodies[k.a].vel - bodies[k.b].vel, k.n);
            real dJ = (vbias[c] - vn) / invSum;
            real jn_new = jn[c] + dJ; if (jn_new < 0) jn_new = real(0);  // no sticking
            vec3 imp = k.n * (jn_new - jn[c]);
            jn[c] = jn_new;
            bodies[k.a].vel += imp * ima;
            bodies[k.b].vel -= imp * imb;

            if (mu[c] > real(0)) {                      // Coulomb friction
                real vt   = dot(bodies[k.a].vel - bodies[k.b].vel, tang[c]);
                real lim  = mu[c] * jn[c];              // cone limit from THIS contact's load
                real jt_new = jt[c] - vt / invSum;
                if (jt_new >  lim) jt_new =  lim;
                if (jt_new < -lim) jt_new = -lim;
                vec3 impt = tang[c] * (jt_new - jt[c]);
                jt[c] = jt_new;
                bodies[k.a].vel += impt * ima;
                bodies[k.b].vel -= impt * imb;
            }
        }

    // (2) position: projected Gauss-Seidel, re-detecting penetration each pass.
    for (int it = 0; it < SEQ_POS_ITERS; it++) {
        build_contacts(bodies, C);
        for (std::size_t c = 0; c < C.size(); c++) {
            const contact& k = C[c];
            real corr = (k.pen - SEQ_POS_SLOP) * SEQ_POS_BETA;
            if (corr <= 0) continue;
            real ima = inv_mass(bodies[k.a]);
            real imb = inv_mass(bodies[k.b]);
            real invSum = ima + imb;
            if (invSum <= real(0)) continue;
            vec3 push = k.n * (corr / invSum);
            bodies[k.a].pos += push * ima;
            bodies[k.b].pos -= push * imb;
        }
    }
}

// Advance the sim one fixed step `dt`: semi-implicit Euler, then the collision
// solver. Returns the max |vel| over the movable bodies, for the caller's sleep
// policy.
//
// Both loops gate on inv_mass() rather than on `motion`, so THE INTEGRATOR MOVES
// EXACTLY THE BODIES THE SOLVER CAN PUSH. That equivalence is the point: a body
// that fell under gravity but absorbed no impulse would be unpushable and yet in
// motion. Testing the same function everywhere makes the two impossible to
// disagree.
// (`mass` is authored strictly positive, so a movable body's reciprocal is
// always > 0; the viewer's field clamps it to [0.01, 1000].)
//
// STATIC bodies never move, and a KINEMATIC body's pose belongs to the driver
// (the viewer writes it from a drag or an animation), so the integrator must not
// fight it. Both still collide — that happens in the solver. Tangential damping
// is no longer applied here: friction is a per-contact force now, so it acts
// only where something is actually touching.
inline real physics_step(std::vector<phys_body>& bodies, const phys_params& p, real dt) {
    for (phys_body& b : bodies) {                 // gravity (y only) + integrate
        if (inv_mass(b) <= real(0)) continue;
        b.vel[1] += p.gravity * dt;
        b.pos    += b.vel * dt;
    }
    solve_sequential(bodies, p);
    real maxv = 0;
    for (phys_body& b : bodies) {
        if (inv_mass(b) <= real(0)) continue;
        real v = b.vel.length();
        if (v > maxv) maxv = v;
    }
    return maxv;
}

#endif // PHYSICS_H
