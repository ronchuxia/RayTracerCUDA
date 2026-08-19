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
// instead of buzzing. A contact is a normal plus a depth and nothing else, which
// is what lets the analytic tests here and the convex-convex detector in gjk.h
// feed the same solver.
//
// Every body carries a physics ROLE (`motion` + `collidable`), a `mass` and a
// surface `friction`, which the solver reads only through inv_mass() and
// combine_friction(), so STATIC / KINEMATIC / DYNAMIC and any mass ratio all
// share one code path. See docs/plans/physics.md (Phase 3).
//
// Current scope: SPHERE and BOX colliders, every pairing. Boxes are ORIENTED
// (they carry their own axes), so a rotated box collides as itself. Sphere-sphere
// and sphere-box have exact analytic tests; everything else goes through the
// support-function detector in gjk.h.
//
// SPHERES ROTATE, BOXES DO NOT YET. A sphere carries angular velocity and a
// scalar inverse inertia, so friction spins it up and a ball ROLLS instead of
// being scrubbed to a halt — the missing piece that made any real friction
// coefficient over-damp. A box's inverse inertia is zero, so it neither takes
// nor gives angular impulse; its inertia tensor, its orientation and the
// multi-point contact manifold it needs to rest without wobbling are all B3b/B3c.
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
    // ROLLING RESISTANCE, dimensionless, combined per contact like friction.
    // Coulomb friction alone cannot stop a rolling ball: once it rolls, its
    // contact point is stationary, so there is nothing left for friction to
    // oppose and a rigid sphere coasts forever. Real balls stop because they and
    // the surface deform, which this stands in for.
    //
    // A rolling sphere decelerates at (5/7) * rolling_friction * g. The 5/7 is
    // the rolling constraint: angular momentum about the CONTACT POINT is
    // (7/5) m v R, since contact forces have no moment arm there, so a resisting
    // couple bleeds off velocity 7/5 more slowly than the same force applied to
    // the centre would. The default 0.01 is rubber on concrete and gives
    // 0.070 m/s^2 — 14 s to stop from 1 m/s, measured 14.15 s.
    //
    // This resists spin ORTHOGONAL to the contact normal only — the component
    // that makes a ball travel. See spinning_friction for the rest.
    real     rolling_friction= real(0.01);
    // SPINNING RESISTANCE: the same idea for spin ABOUT the contact normal — a
    // ball turning on the spot like a top, which moves nowhere. Bullet splits
    // these two and so do we, because they have different lever arms and one
    // coefficient over-damps the spinning one.
    //
    // Rolling resists through deformation over the ball's RADIUS. Spinning
    // resists through torsion across the CONTACT PATCH, whose radius is far
    // smaller for a stiff contact. Both are clamped here against the same
    // `rad` (the lever arm, so the ball's radius), so the ratio of the two
    // coefficients is what carries the ratio of the levers: the default is
    // 0.2 * rolling_friction, taking the patch as roughly a fifth of the radius.
    //
    // A sphere spinning on the spot decelerates at 2.5 * spinning_friction * g / R
    // — no linear coupling, so unlike rolling there is no 5/7. At the defaults
    // that is 0.098 rad/s^2, against 0.49 when one coefficient covered both.
    real     spinning_friction = real(0.002);
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
    // ANGULAR VELOCITY, radians/s about each world axis. Written only by the
    // friction impulse (see solve_sequential) and read only through the lever
    // term, so a body whose inverse inertia is zero can never acquire spin.
    //
    // There is deliberately no ORIENTATION beside it. A sphere is the only shape
    // that spins today and a sphere's shape does not depend on how it is turned,
    // so nothing — collision or rendering — has anything to read an orientation
    // for. Boxes need one, and that is B3c along with the inertia tensor and the
    // quaternion path in transforms.h.
    vec3          omega      = vec3(0, 0, 0);
};

// Convex-convex detection, DEFINED IN gjk.h, which is included at the bottom of
// this file once phys_body is complete — the same wiring hittable.h uses for its
// composite shapes. Declared here because contact_between below calls it.
inline vec3 support(const phys_body& b, const vec3& dir);
inline bool gjk_epa_contact(const phys_body& A, const phys_body& B, vec3& n, real& pen);

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

// I^-1 * L — the angular counterpart of inv_mass, and the only channel through
// which rotation reaches the solver. Same derived-not-stored discipline, and the
// same gate: a body the solver cannot push linearly cannot be spun either.
//
// A solid sphere's inertia is isotropic, I = (2/5) m r^2, so its inverse is the
// SCALAR 5 / (2 m r^2) = 2.5 * inv_mass / r^2 and no orientation is needed to
// apply it — turning a sphere does not change how it resists being spun. That is
// what makes spheres the cheap half of B3.
//
// A BOX RETURNS ZERO, so it neither receives nor responds to angular impulse.
// That is deliberate and must stay consistent: giving a box inverse inertia
// while nothing integrates its orientation would let it absorb spin that never
// becomes motion, which is the same trap a DYNAMIC plane fell into (see
// docs/plans/phase3-status.md item 4) — moved by one channel and immovable in
// another. Box inertia arrives with box orientation in B3c, together.
inline vec3 inv_inertia_apply(const phys_body& b, const vec3& L) {
    if (b.shape != COLLIDER_SPHERE) return vec3(0, 0, 0);
    const real im = inv_mass(b);
    if (im <= real(0) || b.radius <= real(0)) return vec3(0, 0, 0);
    return L * (real(2.5) * im / (b.radius * b.radius));
}

// The LEVER ARM: contact point minus centre of mass, for the body in slot `a`
// (is_a) or slot `b` of a contact whose normal is `n`.
//
// A sphere touches along its own radius, so the contact point is exactly one
// radius from the centre against the normal — no contact point needs to be
// stored or estimated. Two consequences fall out of r being PARALLEL to n:
// cross(r, n) is exactly zero, so the normal impulse exerts no torque and the
// normal effective mass is unchanged; and only FRICTION, which acts along the
// tangent, can spin a ball.
//
// Any other shape returns zero, matching inv_inertia_apply. When B3b adds
// contact manifolds the stored contact point supersedes this, and the sphere
// case must keep agreeing with it.
inline vec3 contact_lever(const phys_body& b, const vec3& n, bool is_a) {
    if (b.shape != COLLIDER_SPHERE) return vec3(0, 0, 0);
    return is_a ? -n * b.radius : n * b.radius;
}

// Velocity of the material point of `b` currently at lever arm `r`. Rolling is
// entirely this: the surface of a rolling ball is momentarily still, because the
// spin term cancels the linear one.
inline vec3 velocity_at(const phys_body& b, const vec3& r) {
    return b.vel + cross(b.omega, r);
}

// Effective inverse mass of a contact along `dir` — how much relative velocity
// one unit of impulse buys. The two angular terms are what rolling costs: for a
// sphere along the TANGENT each contributes 2.5 * inv_mass, so a ball resting on
// an immovable surface has 3.5x the effective mobility it had when it could only
// slide, and the friction impulse that used to stop it now mostly spins it up.
// Along the NORMAL both terms are exactly zero for spheres (r parallel to n), so
// bouncing is untouched.
inline real inv_effective_mass(const phys_body& A, const phys_body& B,
                           const vec3& ra, const vec3& rb, const vec3& dir) {
    return inv_mass(A) + inv_mass(B)
         + dot(dir, cross(inv_inertia_apply(A, cross(ra, dir)), ra))
         + dot(dir, cross(inv_inertia_apply(B, cross(rb, dir)), rb));
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

// A unit vector in the contact's TANGENT PLANE, from the part of `v` orthogonal
// to `n`; an arbitrary perpendicular when `v` has no such part. Both the friction
// tangent and the rolling axis are built with this, and both are impulse
// directions whose clamp only means what it says if they really are orthogonal
// to the normal.
//
// TWO THINGS HERE ARE NOT OPTIONAL, and an absolute epsilon gets both wrong.
// `v - n*dot(v,n)` is exactly orthogonal to n only when n is exactly unit, and a
// float normal is unit to about 1e-7 — sphere_box_contact's n = d/dist rounds,
// and the oriented-box path then recombines it from three axes. So the residue
// left along n scales with |v|: measured at 1.9e-6 for a ball spinning at
// 10 rad/s against a normal of (0, 0.99999994, 0), which is 190x an absolute
// 1e-8 threshold. The axis came back as the NORMAL itself, and the rolling
// impulse clamped to it then damped spin about the normal at full strength —
// exactly the axis it is meant to leave alone.
//   1. The "is there any tangential part" test is RELATIVE to |v|, so it scales
//      with the residue it has to reject.
//   2. The result is projected a SECOND time, which removes the residue that
//      survives the first pass when the tangential part is small but real.
inline vec3 tangent_from(const vec3& v, const vec3& n) {
    vec3 t = v - n * dot(v, n);
    real tl = t.length();
    if (tl <= real(1e-4) * v.length()) return any_perpendicular(n);
    t  = t / tl;
    t  = t - n * dot(t, n);
    tl = t.length();
    return tl > real(0.5) ? t / tl : any_perpendicular(n);
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

// Narrow phase for ONE ORDERED pair: fills n (from B toward A) and pen.
//
// Sphere-sphere and sphere-box have exact closed-form answers, so they are taken
// directly; build_contacts canonicalises the sphere into A so those two cases
// are what an ordered pair usually is. EVERY OTHER PAIRING FALLS THROUGH TO
// GJK/EPA, which needs nothing but each shape's support function and so is
// already correct for box-box, for a box handed in as A with a sphere as B, and
// for B4's convex hulls without further cases here.
//
// The two paths are interchangeable, not layered: both fill the same n and pen
// in the same convention, and tests/test_physics.cu pins them against each other
// on the same configurations. Keeping the analytic pair is a cost decision (a
// handful of operations against an iterative search) and an exactness one, not a
// correctness one.
inline bool contact_between(const phys_body& A, const phys_body& B, vec3& n, real& pen) {
    if (A.shape == COLLIDER_SPHERE && B.shape == COLLIDER_SPHERE) {   // sphere vs sphere
        vec3 d = A.pos - B.pos;
        real dist2 = d.length_squared();
        real rsum = A.radius + B.radius;
        if (dist2 >= rsum * rsum || dist2 < real(1e-12)) return false;
        real dist = std::sqrt(dist2);
        n = d / dist;
        pen = rsum - dist;
        return true;
    }
    if (A.shape == COLLIDER_SPHERE && B.shape == COLLIDER_BOX) {
        // Sphere vs ORIENTED box: move the sphere centre into the box's own
        // frame (project the separation onto each box axis), run the
        // axis-aligned test there against +/-half, then rotate the normal back
        // out. The box's axes absorb the rotation, so no separate oriented-box
        // test is needed. For an axis-aligned box the axes are identity and this
        // reduces to the plain test, expressed relative to the box centre.
        vec3 d = A.pos - B.pos;
        vec3 c_local(dot(d, B.axes[0]), dot(d, B.axes[1]), dot(d, B.axes[2]));
        vec3 n_local;
        if (!sphere_box_contact(c_local, A.radius, -B.half, B.half, n_local, pen)) return false;
        n = n_local.x() * B.axes[0] + n_local.y() * B.axes[1] + n_local.z() * B.axes[2];
        return true;   // unit: n_local is unit and the axes are orthonormal
    }
    return gjk_epa_contact(A, B, n, pen);
}

// Shared NARROW PHASE: every overlapping pair, as a contact. The solver below
// does not care what shape produced a contact, nor which of contact_between's
// two paths found it.
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
            // Canonicalise so the sphere is the first operand, which is what
            // puts a sphere/box pair on contact_between's analytic fast path.
            // Correctness does not depend on it — GJK/EPA handles either order.
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

    std::vector<real> vbias(n), jn(n, real(0)), jt(n, real(0)), jr(n, real(0)), js(n, real(0));
    std::vector<real> mu(n), mur(n), mus(n), rad(n);
    std::vector<vec3> tang(n), roll(n), lev_a(n), lev_b(n);
    for (std::size_t c = 0; c < n; c++) {
        const contact& k = C[c];
        // Lever arms are fixed for the whole step alongside the tangent: both
        // describe the contact's geometry, and re-deriving them mid-iteration
        // from poses the solver is itself moving would make the accumulated
        // impulses inconsistent with the frame they were accumulated in.
        lev_a[c] = contact_lever(bodies[k.a], k.n, true);
        lev_b[c] = contact_lever(bodies[k.b], k.n, false);
        // Relative velocity AT THE CONTACT POINT, not between the centres. For a
        // ball already rolling, the two differ completely: its centre is moving
        // and its contact point is not.
        vec3 vrel = velocity_at(bodies[k.a], lev_a[c]) - velocity_at(bodies[k.b], lev_b[c]);
        real vn = dot(vrel, k.n);
        real e   = combine(bodies[k.a].restitution, bodies[k.b].restitution, p.restitution_combine);
        vbias[c] = vn < -SEQ_REST_VEL ? -e * vn : real(0);
        mu[c]    = combine(bodies[k.a].friction, bodies[k.b].friction, p.friction_combine);
        // Rolling resistance shares friction's combine rule — it is a friction
        // coefficient, and a pair property for the same reason. Its clamp is an
        // ANGULAR impulse, so it needs a length: the contact's own lever arm,
        // which is the radius for a sphere and zero for anything that cannot spin.
        mur[c]   = combine(bodies[k.a].rolling_friction, bodies[k.b].rolling_friction,
                           p.friction_combine);
        mus[c]   = combine(bodies[k.a].spinning_friction, bodies[k.b].spinning_friction,
                           p.friction_combine);
        real la = lev_a[c].length(), lb = lev_b[c].length();
        rad[c]   = la > lb ? la : lb;
        // The ROLLING axis, fixed for the step for the same reason the tangent
        // is: it is the part of the relative spin ORTHOGONAL to the normal, and
        // accumulating an impulse along an axis that rotated every iteration
        // would make its clamp meaningless. The spinning axis needs no such
        // treatment — it IS the contact normal, already fixed. With no rolling
        // spin the direction is arbitrary and the impulse comes out zero.
        roll[c]  = tangent_from(bodies[k.a].omega - bodies[k.b].omega, k.n);
        // Tangent is fixed for the whole step, taken from the sliding direction
        // at its start: accumulating jt along an axis that rotated every
        // iteration would make the cone clamp meaningless.
        tang[c]  = tangent_from(vrel, k.n);
    }

    // (1) velocity: accumulated normal + friction impulses. Mass enters ONLY as
    // inverse mass, so an immovable body contributes 0 to the effective mass and
    // absorbs no impulse (imp * 0) — that one weighting is what makes STATIC and
    // KINEMATIC bodies "free" to push against.
    for (int it = 0; it < SEQ_VEL_ITERS; it++)
        for (std::size_t c = 0; c < n; c++) {
            const contact& k = C[c];
            phys_body& A = bodies[k.a];
            phys_body& B = bodies[k.b];
            const real ima = inv_mass(A), imb = inv_mass(B);
            if (ima + imb <= real(0)) continue;         // nothing here can move
            const vec3& ra = lev_a[c];
            const vec3& rb = lev_b[c];

            // NORMAL. For spheres the two angular terms in inv_effective_mass are
            // exactly zero (the lever arm is parallel to the normal), so this is
            // arithmetically the same impulse it was before rotation existed —
            // which is why a frictionless scene is unchanged to the last bit.
            const real kn = inv_effective_mass(A, B, ra, rb, k.n);
            if (kn > real(0)) {
                real vn = dot(velocity_at(A, ra) - velocity_at(B, rb), k.n);
                real jn_new = jn[c] + (vbias[c] - vn) / kn;
                if (jn_new < 0) jn_new = real(0);       // no sticking
                vec3 imp = k.n * (jn_new - jn[c]);
                jn[c] = jn_new;
                A.vel += imp * ima;   A.omega += inv_inertia_apply(A,  cross(ra, imp));
                B.vel -= imp * imb;   B.omega -= inv_inertia_apply(B,  cross(rb, imp));
            }

            // FRICTION, and the only thing that can spin a ball. The impulse is
            // unchanged in form; what changed is that it now also torques, and
            // that the tangential effective mass includes the cost of spinning
            // up, so most of it goes into rotation instead of into stopping.
            const real kt = inv_effective_mass(A, B, ra, rb, tang[c]);
            if (mu[c] > real(0) && kt > real(0)) {
                real vt   = dot(velocity_at(A, ra) - velocity_at(B, rb), tang[c]);
                real lim  = mu[c] * jn[c];              // cone limit from THIS contact's load
                real jt_new = jt[c] - vt / kt;
                if (jt_new >  lim) jt_new =  lim;
                if (jt_new < -lim) jt_new = -lim;
                vec3 impt = tang[c] * (jt_new - jt[c]);
                jt[c] = jt_new;
                A.vel += impt * ima;  A.omega += inv_inertia_apply(A, cross(ra, impt));
                B.vel -= impt * imb;  B.omega -= inv_inertia_apply(B, cross(rb, impt));
            }

            // ROLLING and SPINNING RESISTANCE. Both are PURE angular impulses —
            // no linear component, because what is resisted is rotation itself,
            // not sliding. Once a ball rolls its contact point is stationary and
            // Coulomb friction has nothing left to act on, so without these a
            // rigid sphere turns forever.
            //
            // TWO AXES, TWO COEFFICIENTS. Split the relative spin at the contact
            // into the part ORTHOGONAL to the normal, which is what carries a
            // ball along, and the part ABOUT the normal, which is a ball turning
            // on the spot. They resist through different mechanisms over
            // different lengths — deformation across the ball's radius against
            // torsion across the contact patch — so one coefficient for both
            // over-damps the spinning one.
            //
            // Each is clamped like friction, against the same accumulated normal
            // load times the lever arm, so both scale with how hard the surfaces
            // are pressed together and vanish the instant they separate. A body
            // that cannot spin has zero angular effective mass and is skipped, so
            // this costs nothing on scenes without rotation.
            if (rad[c] > real(0)) {
                const vec3 wrel = A.omega - B.omega;
                if (mur[c] > real(0)) {                 // rolling: orthogonal to n
                    real kw = dot(roll[c], inv_inertia_apply(A, roll[c]))
                            + dot(roll[c], inv_inertia_apply(B, roll[c]));
                    if (kw > real(0)) {
                        real lim = mur[c] * jn[c] * rad[c];
                        real jr_new = jr[c] - dot(wrel, roll[c]) / kw;
                        if (jr_new >  lim) jr_new =  lim;
                        if (jr_new < -lim) jr_new = -lim;
                        vec3 impr = roll[c] * (jr_new - jr[c]);
                        jr[c] = jr_new;
                        A.omega += inv_inertia_apply(A, impr);
                        B.omega -= inv_inertia_apply(B, impr);
                    }
                }
                if (mus[c] > real(0)) {                 // spinning: about n
                    real kw = dot(k.n, inv_inertia_apply(A, k.n))
                            + dot(k.n, inv_inertia_apply(B, k.n));
                    if (kw > real(0)) {
                        real lim = mus[c] * jn[c] * rad[c];
                        real js_new = js[c] - dot(wrel, k.n) / kw;
                        if (js_new >  lim) js_new =  lim;
                        if (js_new < -lim) js_new = -lim;
                        vec3 imps = k.n * (js_new - js[c]);
                        js[c] = js_new;
                        A.omega += inv_inertia_apply(A, imps);
                        B.omega -= inv_inertia_apply(B, imps);
                    }
                }
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
// solver. Returns the max SURFACE SPEED over the movable bodies, for the
// caller's sleep policy — see the note on the return value below.
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
//
// NO ORIENTATION IS INTEGRATED. `omega` persists between steps and drives
// contacts through the lever term, but nothing turns because a sphere is the
// only shape that can spin and its shape and appearance are the same whichever
// way it is turned. B3c integrates a quaternion for boxes.
//
// The RETURN VALUE is a surface speed, not a centre speed: `|v|` and
// `|omega| * radius` are both linear speeds and either can keep a body awake on
// its own. Taking only `|v|` would let a ball spinning on the spot count as
// asleep, and it is not — the moment it meets anything, friction converts that
// spin into motion.
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
        real w = b.omega.length() * b.radius;     // radius is 0 for a box, which cannot spin
        if (w > v) v = w;
        if (v > maxv) maxv = v;
    }
    return maxv;
}

// The convex-convex detector, wired in at the bottom so it can see the complete
// phys_body it reads colliders from (hittable.h does the same for the composite
// shapes its dispatch calls). Include physics.h, never gjk.h.
#include "gjk.h"

#endif // PHYSICS_H
