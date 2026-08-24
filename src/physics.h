#ifndef PHYSICS_H
#define PHYSICS_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "precision.h"   // real
#include "quat.h"        // orientation
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
// EVERYTHING ROTATES. A sphere carries angular velocity and a scalar inverse
// inertia, so friction spins it up and a ball ROLLS instead of being scrubbed to
// a halt — the missing piece that made any real friction coefficient over-damp
// (B3a). A box carries a quaternion orientation and a full inertia tensor, and
// its contacts are multi-point MANIFOLDS rather than a single point (B3b/B3c):
// the tensor is what lets it take torque, and the manifold is what lets it rest
// flat instead of rocking on one pivot. The two had to land together — a box
// with inertia but a one-point contact is a see-saw, and a box with a manifold
// but no inertia cannot use it.
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
    // ORIENTATION, and the AUTHORITATIVE one (B3c). A quaternion rather than the
    // matrix, because only a quaternion can be integrated: nine numbers nudged
    // by omega*dt are no longer a rotation, and re-orthonormalising them every
    // step is both expensive and arbitrary, where a quaternion needs one
    // normalise. See quat.h.
    quat          orient;
    // The box's own x/y/z axes in world space — the COLUMNS of `orient`'s
    // rotation matrix. A DERIVED CACHE, not state: GJK's support function and
    // the sphere-box test project onto these on every iteration, and rebuilding
    // them from the quaternion each time would be the hot path paying for the
    // representation. Write them only through set_orientation(); anything that
    // sets one without the other has produced a body whose collider and pose
    // disagree, which is exactly what this file is arranged to prevent.
    vec3          axes[3]    = { vec3(1,0,0), vec3(0,1,0), vec3(0,0,1) };
    // ANGULAR VELOCITY, radians/s about each world axis. Written by the friction
    // and contact impulses (see solve_sequential), read through the lever term,
    // and integrated into `orient` by physics_step — so a body whose inverse
    // inertia is zero can never acquire spin, and one that does acquire it
    // actually turns.
    vec3          omega      = vec3(0, 0, 0);
};

// The ONE write path for orientation: set the quaternion, rebuild the axis
// cache. Everything that turns a body goes through here — the integrator, the
// factories in viewer/physics_utils.h, and the tests — so the two
// representations cannot drift apart.
inline void set_orientation(phys_body& b, const quat& q) {
    b.orient  = normalize(q);
    b.axes[0] = quat_axis(b.orient, 0);
    b.axes[1] = quat_axis(b.orient, 1);
    b.axes[2] = quat_axis(b.orient, 2);
}

// Authoring from a MATRIX, which is how a pose arrives from transforms.h: the
// caller hands over the three world-space axes it already has and the
// quaternion is recovered from them, rather than the caller learning this
// file's convention. The axes are re-derived from that quaternion rather than
// copied, so a slightly non-orthonormal input is cleaned up here instead of
// silently becoming a shearing "rotation".
inline void set_orientation_from_axes(phys_body& b, const vec3 axes[3]) {
    set_orientation(b, quat_from_axes(axes));
}

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
// what makes spheres the cheap half of B3: no frame, no tensor, no rebuild.
//
// A BOX IS ANISOTROPIC, and that is the whole difference. Its inertia is
// diagonal only in ITS OWN frame — a long box resists turning about its long
// axis far less than across it — so applying I^-1 means rotating the angular
// momentum into the body frame, scaling per axis, and rotating back:
// I^-1_world = R I^-1_local R^T, with R's columns being the cached axes. That is
// why the orientation had to arrive first: without it there is no frame to
// rotate into, which is what "box inertia arrives with box orientation, together"
// meant (the trap recorded as docs/plans/phase3-status.md item 4 — a body moved
// by one channel and immovable in another).
//
// For a solid box of half-extents h, I_xx = (1/12) m ((2h_y)^2 + (2h_z)^2)
// = (1/3) m (h_y^2 + h_z^2), so the inverse component is 3 * inv_mass /
// (h_y^2 + h_z^2). A degenerate (zero-thickness) extent would divide by zero, so
// each component is guarded; a body with no extent at all cannot spin.
inline vec3 inv_inertia_apply(const phys_body& b, const vec3& L) {
    const real im = inv_mass(b);
    if (im <= real(0)) return vec3(0, 0, 0);
    if (b.shape == COLLIDER_SPHERE) {
        if (b.radius <= real(0)) return vec3(0, 0, 0);
        return L * (real(2.5) * im / (b.radius * b.radius));
    }
    // Box: into the body frame, scale, back out.
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

// The LEVER ARM: contact point minus centre of mass, for the body in slot `a`
// (is_a) or slot `b` of a contact whose normal is `n` and whose stored world
// contact point is `p`.
//
// A SPHERE IS COMPUTED, NOT READ, and deliberately so. It touches along its own
// radius, so its lever is exactly one radius from the centre against the normal
// — no stored point can be more accurate than that, and one shared point cannot
// be right for both bodies at once anyway (the two surface points differ by the
// penetration). Two consequences fall out of r being PARALLEL to n: cross(r, n)
// is exactly zero, so the normal impulse exerts no torque and the normal
// effective mass is unchanged; and only FRICTION, which acts along the tangent,
// can spin a ball. Keeping this branch is also what makes B3b bit-exact for
// scenes of spheres — the measured B3a rolling results still hold.
//
// Everything else reads the manifold point, which is what lets a box take
// torque: a ball landing on the corner of a crate now tips it.
inline vec3 contact_lever(const phys_body& b, const vec3& n, const vec3& p, bool is_a) {
    if (b.shape == COLLIDER_SPHERE) return is_a ? -n * b.radius : n * b.radius;
    return p - b.pos;
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
    // The world-space CONTACT POINT (B3b). One pair can emit several of these —
    // a box resting on the ground emits four, one per corner of the overlap —
    // and each is solved as its own contact, which is what stops a resting box
    // from pivoting about a single point. `mcount` is how many the pair emitted,
    // and it exists only so the position pass can share one overlap between
    // them instead of correcting it once per point.
    vec3 p;
    int  mcount;
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

// ---- contact manifolds (B3b) ---------------------------------------------
//
// A normal and a depth say how to separate two shapes; they do not say WHERE
// they touch, and for anything that can rotate that is the more important half.
// One point is a pivot: a crate resting on the floor at a single point is a
// see-saw, and it will rock forever no matter how good the solver is. A box rests
// only when its contact is described by the POLYGON its face shares with the
// floor, sampled at the corners.
//
// The construction is the standard one — pick the face most square to the
// normal on each body, clip one against the other, keep what is behind the
// reference plane:
//
//     reference face (the flatter of the two)      incident face (the other box)
//     ┌───────────────┐                            ┌───────────┐
//     │               │  clip the incident face    │  ·     ·  │  4 corners in,
//     │   ·       ·   │  against these 4 side   →  │           │  up to 8 out,
//     │               │  planes, keep points       │  ·     ·  │  keep the 4
//     │   ·       ·   │  below the face plane      └───────────┘  deepest
//     └───────────────┘
//
// Each surviving point becomes its OWN contact with its own penetration and its
// own accumulated impulse, so the solver needs no notion of a manifold — it just
// sees four contacts that happen to share a normal. That is what makes this
// change small: the Gauss-Seidel loop is untouched.

// The face of box `b` most square to `dir`: fills the axis and which end of it,
// and returns the alignment (1 = exactly square, 0 = edge-on).
inline real box_best_face(const phys_body& b, const vec3& dir, int& axis, real& sign) {
    real best = real(-1);
    axis = 0; sign = real(1);
    for (int i = 0; i < 3; i++) {
        const real d = dot(b.axes[i], dir);
        const real a = d < real(0) ? -d : d;
        if (a > best) { best = a; axis = i; sign = d < real(0) ? real(-1) : real(1); }
    }
    return best;
}

// Sutherland-Hodgman: keep the part of the polygon with dot(p, nrm) <= off,
// inserting the crossing point on every edge that leaves the half-space. A quad
// clipped by four planes yields at most 8 points, which is why the callers'
// buffers are 16.
inline int clip_polygon(const vec3* in, int n, const vec3& nrm, real off, vec3* out) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        const vec3& p = in[i];
        const vec3& q = in[(i + 1) % n];
        const real dp = dot(p, nrm) - off;
        const real dq = dot(q, nrm) - off;
        const bool pin = dp <= real(0), qin = dq <= real(0);
        if (pin) out[m++] = p;
        if (pin != qin) {
            const real t = dp / (dp - dq);        // dp != dq whenever they straddle
            out[m++] = p + (q - p) * t;
        }
    }
    return m;
}

// Up to 4 contact points for two boxes already known to overlap along `n`.
// Returns 0 if the clip degenerates (edge-on contact, where no face pair
// describes the touch) and the caller should fall back to a single point.
inline int box_box_manifold(const phys_body& A, const phys_body& B, const vec3& n,
                            vec3 pts[4], real pens[4]) {
    int  ai, bi;
    real as, bs;
    // n points from B toward A, so A's touching face looks along -n and B's
    // along +n. The REFERENCE is whichever is more square to the normal: the
    // flatter face makes the better clipping frame, and the other box's face is
    // the one that gets clipped.
    const real align_a = box_best_face(A, -n, ai, as);
    const real align_b = box_best_face(B,  n, bi, bs);
    const bool ref_is_a = align_a >= align_b;
    const phys_body& R = ref_is_a ? A : B;
    const phys_body& I = ref_is_a ? B : A;
    const int  raxis = ref_is_a ? ai : bi;
    const real rsign = ref_is_a ? as : bs;

    const vec3 rn = R.axes[raxis] * rsign;                 // reference face outward normal
    const vec3 rc = R.pos + rn * R.half[raxis];            // its centre
    const int  rj = (raxis + 1) % 3, rk = (raxis + 2) % 3; // its two in-plane axes

    // Incident face: the face of the other box most anti-parallel to rn.
    int  iaxis = 0;
    real isign = real(1), best = real(-1);
    for (int i = 0; i < 3; i++) {
        const real d = dot(I.axes[i], rn);
        const real a = d < real(0) ? -d : d;
        if (a > best) { best = a; iaxis = i; isign = d < real(0) ? real(1) : real(-1); }
    }
    const vec3 ic = I.pos + I.axes[iaxis] * (isign * I.half[iaxis]);
    const int  ij = (iaxis + 1) % 3, ik = (iaxis + 2) % 3;
    const vec3 u = I.axes[ij] * I.half[ij], v = I.axes[ik] * I.half[ik];

    vec3 buf0[16], buf1[16];
    buf0[0] = ic + u + v;  buf0[1] = ic + u - v;
    buf0[2] = ic - u - v;  buf0[3] = ic - u + v;
    int np = 4;

    // Clip against the reference face's four side planes.
    const vec3 side[2] = { R.axes[rj], R.axes[rk] };
    const real ext[2]  = { R.half[rj], R.half[rk] };
    vec3* src = buf0; vec3* dst = buf1;
    for (int s = 0; s < 2 && np > 0; s++) {
        np = clip_polygon(src, np, side[s],  dot(rc, side[s]) + ext[s], dst);
        vec3* t = src; src = dst; dst = t;
        if (np == 0) break;
        np = clip_polygon(src, np, -side[s], -dot(rc, side[s]) + ext[s], dst);
        t = src; src = dst; dst = t;
    }

    // Keep the points that are actually behind the reference face, deepest first
    // when there are more than four — the shallow ones carry the least load and
    // are the ones a 4-point manifold can afford to drop.
    int cnt = 0;
    for (int i = 0; i < np; i++) {
        const real d = dot(src[i] - rc, rn);
        if (d > real(0)) continue;                 // not penetrating
        const real pen = -d;
        if (cnt < 4) { pts[cnt] = src[i]; pens[cnt] = pen; cnt++; continue; }
        int shallow = 0;
        for (int q = 1; q < 4; q++) if (pens[q] < pens[shallow]) shallow = q;
        if (pen > pens[shallow]) { pts[shallow] = src[i]; pens[shallow] = pen; }
    }
    return cnt;
}

// Contact POINTS for an ordered pair whose normal and depth are already known.
// A sphere touches at exactly one point and needs no clipping; two boxes get the
// face clip above; anything else (and a degenerate clip) falls back to the
// deepest support point, which is always on the surface and always inside the
// overlap.
inline int contact_manifold(const phys_body& A, const phys_body& B,
                            const vec3& n, real pen, vec3 pts[4], real pens[4]) {
    if (A.shape == COLLIDER_SPHERE) { pts[0] = A.pos - n * A.radius; pens[0] = pen; return 1; }
    if (B.shape == COLLIDER_SPHERE) { pts[0] = B.pos + n * B.radius; pens[0] = pen; return 1; }
    if (A.shape == COLLIDER_BOX && B.shape == COLLIDER_BOX) {
        const int c = box_box_manifold(A, B, n, pts, pens);
        if (c > 0) return c;
    }
    pts[0] = support(A, -n); pens[0] = pen; return 1;
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
            if (contact_between(bodies[a], bodies[b], n, pen)) {
                // One overlap, up to four contacts — each manifold point is
                // solved independently, sharing only the normal.
                vec3 pts[4]; real pens[4];
                const int m = contact_manifold(bodies[a], bodies[b], n, pen, pts, pens);
                for (int q = 0; q < m; q++)
                    out.push_back({ a, b, n, pens[q], pts[q], m });
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
        lev_a[c] = contact_lever(bodies[k.a], k.n, k.p, true);
        lev_b[c] = contact_lever(bodies[k.b], k.n, k.p, false);
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
        // The ROLLING RADIUS, and it must be read from the SHAPE rather than
        // from the lever arms. Taking max(|lev_a|, |lev_b|) gave the same answer
        // while a box's lever was zero, but B3b made a box's lever real: the
        // ground is a 100-deep box, so its lever is 100 and the clamp below came
        // out 200x too strong — a rolling ball stopped in 0.2 s instead of 7.
        // Rolling and spinning resistance stand in for a CURVED body deforming
        // against a surface, so a box contributes nothing to it; two boxes get
        // no rolling resistance at all, and settle through the manifold instead.
        rad[c]   = real(0);
        if (bodies[k.a].shape == COLLIDER_SPHERE) rad[c] = bodies[k.a].radius;
        if (bodies[k.b].shape == COLLIDER_SPHERE && bodies[k.b].radius > rad[c])
            rad[c] = bodies[k.b].radius;
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
            // Divided by the manifold size: four points of one box-on-floor
            // contact describe ONE overlap, so correcting the full depth at each
            // would push the box out four times over and pop it into the air.
            real corr = (k.pen - SEQ_POS_SLOP) * SEQ_POS_BETA / real(k.mcount);
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
// ORIENTATION IS INTEGRATED HERE (B3c), on the same gate as position: a body the
// solver cannot push cannot turn either. `omega` drives contacts through the
// lever term AND advances the quaternion, so a box that acquires spin actually
// ends up facing a different way — which is the whole difference between
// absorbing an angular impulse and expressing one.
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
        // ORIENTATION, integrated for every movable body on the same gate as
        // position. A sphere's own shape does not depend on how it is turned, so
        // this changes nothing it collides or looks like; it is integrated
        // anyway because the alternative is a shape test here that would have to
        // agree with the one in inv_inertia_apply, and a body that spins but
        // does not turn is the exact inconsistency this phase exists to remove.
        if (b.omega.length_squared() > real(0))
            set_orientation(b, quat_integrate(b.orient, b.omega, dt));
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
