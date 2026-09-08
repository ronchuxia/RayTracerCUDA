// Physics module tests — call the REAL src/physics.h (physics_step /
// solve_sequential / sphere_aabb_contact), not a reimplementation. The whole
// point of extracting physics into a header was that tests exercise the
// shipping code directly.
//
//   nvcc tests/test_physics.cu -o build/test_physics -std=c++14 -arch=sm_86 -rdc=true -Isrc
//
// Everything that collides is a body, so these build their world the same way
// the viewer does: STATIC box bodies for the ground and obstacles, DYNAMIC
// spheres for the movers.
#include <cstdio>
#include <cmath>
#include <vector>

#include "physics.h"
#include "physics/hull.h"

#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } \
                              else printf("ok: %s\n", msg); } while (0)

// --- world builders, mirroring the viewer's body scan ---
// The ground is a large flat static BOX with its top face at y = 0. There is no
// plane collider any more: every collider is read from the object it belongs to,
// and no scene object is an infinite plane. A box top gives the same contact
// normal and penetration a plane did, so the assertions below are unchanged.
static phys_body ground_plane(real friction = real(0.5), real restitution = real(0.7)) {
    phys_body g{ -1, vec3(0, -100, 0), vec3(0,0,0), vec3() };
    g.motion = STATIC; g.shape = COLLIDER_BOX; g.half = vec3(1000, 100, 1000);
    g.friction = friction; g.restitution = restitution;
    return g;
}
static phys_body static_box(const vec3& centre, const vec3& half,
                            real friction = real(0.5), real restitution = real(0.7)) {
    phys_body b{ -1, centre, vec3(0,0,0), vec3() };
    b.motion = STATIC; b.shape = COLLIDER_BOX; b.half = half;
    b.friction = friction; b.restitution = restitution;
    return b;
}
// A box that FALLS. Box-box contact is convex-convex (gjk.h), so this is the
// only body type that exercises it end to end; before B2 a dynamic box fell
// through everything because the narrow phase had no test for it.
static phys_body dynamic_box(const vec3& centre, const vec3& half,
                             real friction = real(0.5), real restitution = real(0.7)) {
    phys_body b = static_box(centre, half, friction, restitution);
    b.motion = DYNAMIC;
    return b;
}
// The same box turned `deg` degrees about y, through the one orientation write
// path — the viewer seeds bodies from a transform's Euler angles the same way.
static phys_body rotated_box_y(const vec3& centre, const vec3& half, real deg) {
    phys_body b = static_box(centre, half);
    set_orientation(b, quat_from_euler_zyx_degrees(vec3(0, deg, 0)));
    return b;
}
// A box as a HULL (B4): the same eight corners, six faces with counter-clockwise
// loops seen from outside. The analytic box is the oracle for every hull assertion.
static hull_shape box_hull(const vec3& half) {
    hull_shape h;
    for (int i = 0; i < 8; i++)
        h.verts.push_back(vec3(i & 1 ? half[0] : -half[0], i & 2 ? half[1] : -half[1], i & 4 ? half[2] : -half[2]));
    for (int a = 0; a < 3; a++)
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            hull_shape::face f;
            f.normal = vec3(a == 0 ? sgn : 0, a == 1 ? sgn : 0, a == 2 ? sgn : 0);
            const int u = (a + 1) % 3, v = (a + 2) % 3;              // u x v = normal for sgn > 0
            for (int k = 0; k < 4; k++) {                            // counter-clockwise about +normal
                const int cu = (k == 1 || k == 2), cv = (k >= 2);
                int idx = 0;
                if (a == 0 ? sgn > 0 : (u == 0 ? cu : cv)) idx |= 1;
                if (a == 1 ? sgn > 0 : (u == 1 ? cu : cv)) idx |= 2;
                if (a == 2 ? sgn > 0 : (u == 2 ? cu : cv)) idx |= 4;
                f.loop.push_back(idx);
            }
            if (sgn < 0) { int t = f.loop[1]; f.loop[1] = f.loop[3]; f.loop[3] = t; }   // mirror the winding for -normal
            h.faces.push_back(f);
        }
    hull_mass_properties(h);
    return h;
}
// A right prism with a regular `sides`-gon of circumradius r as its cap (y = ±h):
// the first hull with faces of more than four vertices, for the face clip.
static hull_shape prism_hull(int sides, real r, real h) {
    hull_shape hs;
    for (int i = 0; i < sides; i++) {
        const real t = real(2 * M_PI) * i / sides;
        hs.verts.push_back(vec3(r * std::cos(t), -h, r * std::sin(t)));   // bottom_i = 2i, top_i = 2i + 1
        hs.verts.push_back(vec3(r * std::cos(t),  h, r * std::sin(t)));
    }
    hull_shape::face top, bottom;
    top.normal = vec3(0, 1, 0); bottom.normal = vec3(0, -1, 0);
    for (int i = 0; i < sides; i++) { top.loop.push_back(2 * (sides - 1 - i) + 1); bottom.loop.push_back(2 * i); }
    hs.faces.push_back(top); hs.faces.push_back(bottom);
    for (int i = 0; i < sides; i++) {
        const int j = (i + 1) % sides;
        const real t = real(2 * M_PI) * (i + real(0.5)) / sides;
        hull_shape::face f;
        f.normal = vec3(std::cos(t), 0, std::sin(t));
        f.loop = { 2 * i, 2 * i + 1, 2 * j + 1, 2 * j };
        hs.faces.push_back(f);
    }
    hull_mass_properties(hs);
    return hs;
}
static bool wound_outward(const hull_shape& hs) {                    // every loop counter-clockwise about its normal
    bool ok = true;
    for (const hull_shape::face& f : hs.faces) {
        const vec3 e0 = hs.verts[f.loop[1]] - hs.verts[f.loop[0]], e1 = hs.verts[f.loop[2]] - hs.verts[f.loop[0]];
        ok &= dot(cross(e0, e1), f.normal) > real(0);
    }
    return ok;
}
static phys_body static_hull(const vec3& centre, const hull_shape& hull) {
    phys_body b{ -1, centre, vec3(0,0,0), vec3() };
    b.motion = STATIC; b.shape = COLLIDER_HULL; b.hull = hull;
    return b;
}
static phys_body dynamic_hull(const vec3& centre, const hull_shape& hull) {
    phys_body b = static_hull(centre, hull);
    b.motion = DYNAMIC;
    return b;
}
// two manifolds equal as sets: every point of C0 has a match in C1
static double manifold_distance(const std::vector<contact>& C0, const std::vector<contact>& C1) {
    double worst = C0.size() == C1.size() ? 0 : 1e9;
    for (const contact& k : C0) {
        double best = 1e9;
        for (const contact& l : C1) {
            const double e = (double)(k.p - l.p).length() + std::fabs((double)k.pen - (double)l.pen);
            if (e < best) best = e;
        }
        if (best > worst) worst = best;
    }
    return worst;
}
static phys_body ball(int scene_id, const vec3& pos, const vec3& vel, real r,
                      real m = real(1), real friction = real(0.5),
                      real restitution = real(0.7)) {
    phys_body b{ scene_id, pos, vel, vec3() };
    b.radius = r;
    b.mass = m; b.friction = friction; b.restitution = restitution;
    return b;
}

int main() {
    int fails = 0;
    const real h = real(1.0 / 240);
    const phys_params p_bounce{ real(-9.8) };   // bodies default to restitution 0.7

    // 1. Head-on, equal mass, e = 0.7: two unit spheres overlapping and
    //    approaching at +/-1 along x. Analytic: each bounces back at 0.7 and the
    //    position pass pushes the centres to just-touching (2.0 apart, less slop).
    {
        std::vector<phys_body> b;
        b.push_back(ball(0, vec3(-0.9, 5, 0), vec3( 1,0,0), real(1)));
        b.push_back(ball(1, vec3( 0.9, 5, 0), vec3(-1,0,0), real(1)));
        solve_sequential(b, p_bounce);
        real sep = (b[0].pos - b[1].pos).length();
        CHECK(std::fabs((double)b[0].vel[0] + 0.7) < 1e-4 &&
              std::fabs((double)b[1].vel[0] - 0.7) < 1e-4 &&
              std::fabs((double)sep - 2.0)         < 1e-3,
              "head-on collision: restitution bounce + push to just-touching");
    }

    // 2. Three clustered spheres dropped under gravity settle on the ground with
    //    no interpenetration (exercises physics_step end to end: integrate +
    //    ground contacts + friction + pair collisions).
    {
        std::vector<phys_body> bodies;
        bodies.push_back(ground_plane());
        const int N = 3; const real drop = 3;
        for (int i = 0; i < N; i++) {
            real ox = real(0.7) * ((i & 1) ? real(1) : real(-1));
            real oz = real(0.4) * (real(i) - real(N - 1) * real(0.5));
            bodies.push_back(ball(i, vec3(ox, drop + real(1.8) * i, oz), vec3(0,0,0), real(1)));
        }
        real maxv = 0;
        for (int s = 0; s < 20000; s++) maxv = physics_step(bodies, p_bounce, h);

        bool on_ground = true, no_overlap = true;
        for (int i = 1; i <= N; i++)
            if (bodies[i].pos[1] < bodies[i].radius - real(1e-2)) on_ground = false;
        for (int i = 1; i <= N; i++)
            for (int j = i + 1; j <= N; j++)
                if ((bodies[i].pos - bodies[j].pos).length()
                    < bodies[i].radius + bodies[j].radius - real(1e-2)) no_overlap = false;

        CHECK(maxv < real(0.05), "dropped spheres settle (max |v| -> 0)");
        CHECK(on_ground,         "all bodies rest on the ground plane");
        CHECK(no_overlap,        "no interpenetration at rest");
    }

    // 3. Sphere vs an axis-aligned box [-1,1]^3 — the detection helper directly,
    //    since it is what build_contacts dispatches to for a sphere/box pair.
    const vec3 bmin(-1,-1,-1), bmax(1,1,1);
    {
        // (a) FACE: closest point is on the +x face -> axis-aligned normal.
        vec3 n; real pen;
        bool hit = sphere_aabb_contact(vec3(1.4, 0, 0), real(0.5), bmin, bmax, n, pen);
        CHECK(hit && std::fabs((double)n[0] - 1.0) < 1e-6 &&
              std::fabs((double)pen - 0.1) < 1e-5,
              "sphere-vs-box FACE: axis normal, penetration = r - gap");
    }
    {
        // (b) CORNER: past the (+,+,+) vertex along the diagonal -> diagonal normal.
        real off = real(1) + real(0.5) / std::sqrt(3.0) - real(0.05);
        vec3 n; real pen;
        bool hit = sphere_aabb_contact(vec3(off, off, off), real(0.5), bmin, bmax, n, pen);
        vec3 diag = unit_vector(vec3(1,1,1));
        CHECK(hit && dot(n, diag) > real(0.999),
              "sphere-vs-box CORNER: closest point is the vertex, normal is the diagonal");
    }
    {
        // (c) INSIDE: centre inside the box, nearest the +x face -> ejected out +x.
        vec3 n; real pen;
        bool hit = sphere_aabb_contact(vec3(0.8, 0, 0), real(0.5), bmin, bmax, n, pen);
        CHECK(hit && std::fabs((double)n[0] - 1.0) < 1e-6 &&
              std::fabs((double)pen - 0.7) < 1e-5,      // r + exit distance = 0.5 + 0.2
              "sphere-vs-box INSIDE: ejected along the nearest face");
    }
    {
        // (d) the same pair through the real narrow phase: a ball resting on top
        //     of a static box must produce one contact with an upward normal.
        std::vector<phys_body> b;
        b.push_back(static_box(vec3(0, 0.5, 0), vec3(0.5, 0.5, 0.5)));   // box spans y=[0,1]
        b.push_back(ball(0, vec3(0, real(1.4), 0), vec3(), real(0.5)));  // resting on its top
        std::vector<contact> C;
        build_contacts(b, C);
        CHECK(C.size() == 1 && C[0].a == 1 && C[0].b == 0 && C[0].n[1] > real(0.99),
              "narrow phase: sphere-on-box yields one contact, sphere is `a`, normal is up");
    }

    // 3r. ORIENTED boxes. A box turned 45 degrees about y has a world AABB 41%
    //     wider than itself, so a collider read from that AABB stops balls short
    //     of the visible surface. The unit box below and the probe point are
    //     chosen so the two answers DISAGREE: the point sits inside the enclosing
    //     AABB but 0.35 clear of the box, well past the 0.2 radius.
    const vec3 unit_half(0.5, 0.5, 0.5);
    const vec3 probe(0.6, 0, 0.6);            // local (0, 0, 0.849) once turned 45 deg
    {
        // (a) axis-aligned: the probe DOES touch (closest point (0.5,0,0.5),
        //     gap 0.141 < 0.2). This is the control for (b).
        std::vector<phys_body> b;
        b.push_back(static_box(vec3(0,0,0), unit_half));
        b.push_back(ball(0, probe, vec3(), real(0.2)));
        std::vector<contact> C;
        build_contacts(b, C);
        CHECK(C.size() == 1, "oriented box control: axis-aligned box DOES touch the probe");
    }
    {
        // (b) turned 45 degrees, same probe: no contact. Reading the world AABB
        //     would put the probe inside the box entirely and report a deep hit.
        std::vector<phys_body> b;
        b.push_back(rotated_box_y(vec3(0,0,0), unit_half, real(45)));
        b.push_back(ball(0, probe, vec3(), real(0.2)));
        std::vector<contact> C;
        build_contacts(b, C);
        CHECK(C.size() == 0, "oriented box: 45-degree turn moves the surface away from the probe");
    }
    {
        // (c) the normal must come back out in world space: approach along the
        //     turned box's own +z axis, 0.1 deep.
        vec3 axis_z(real(std::sqrt(0.5)), 0, real(std::sqrt(0.5)));   // Ry(45) * (0,0,1)
        std::vector<phys_body> b;
        b.push_back(rotated_box_y(vec3(0,0,0), unit_half, real(45)));
        b.push_back(ball(0, axis_z * real(0.6), vec3(), real(0.2)));
        std::vector<contact> C;
        build_contacts(b, C);
        CHECK(C.size() == 1 && dot(C[0].n, axis_z) > real(0.9999) &&
              std::fabs((double)C[0].pen - 0.1) < 1e-5,
              "oriented box: contact normal is the box's turned axis, not a world axis");
    }
    {
        // (d) end to end: a ball dropped over the corner region of a turned box
        //     must fall PAST it to the ground. With an AABB-derived collider it
        //     would rest on top of the box (y ~ 1.2) — on nothing visible.
        std::vector<phys_body> b;
        b.push_back(ground_plane());
        b.push_back(rotated_box_y(vec3(0, 0.5, 0), unit_half, real(45)));   // spans y=[0,1]
        b.push_back(ball(0, vec3(probe[0], real(3), probe[2]), vec3(), real(0.2)));
        for (int s = 0; s < 4000; s++) physics_step(b, p_bounce, h);
        CHECK(b[2].pos[1] < real(0.3),
              "oriented box: a ball over the turned box's corner falls to the ground");
    }

    // 4. Physics ROLES (motion + collidable). Gravity is off and the bodies sit
    //    in free space, so ONLY the pair contact acts.
    {
        const phys_params p{ real(0) };

        // (a) STATIC neighbour, overlapping by 0.5: it must not budge, and the
        //     dynamic body must take the WHOLE push-out (two dynamics would
        //     split it and meet in the middle).
        {
            std::vector<phys_body> b;
            b.push_back(ball(0, vec3(0,   5, 0), vec3(), real(1), real(1), real(0.5), real(0)));
            b.push_back(ball(1, vec3(1.5, 5, 0), vec3(), real(1), real(1), real(0.5), real(0)));
            b[0].motion = STATIC;
            for (int s = 0; s < 200; s++) physics_step(b, p, h);
            real sep = (b[1].pos - b[0].pos).length();
            CHECK((b[0].pos - vec3(0, 5, 0)).length() < real(1e-9) &&  // immovable
                  std::fabs((double)sep - 2.0) < 1e-3 &&               // separated to touching
                  b[1].pos[0] > real(1.9),                             // the DYNAMIC one moved
                  "role STATIC: immovable; the dynamic neighbour takes the full push-out");
        }

        // (b) KINEMATIC neighbour carrying a velocity: the integrator must leave
        //     its pose to the driver (no drift) and no impulse may land on it,
        //     while it still shoves the dynamic body clear.
        {
            std::vector<phys_body> b;
            b.push_back(ball(0, vec3(0,   5, 0), vec3(1,0,0), real(1), real(1), real(0.5), real(0)));
            b.push_back(ball(1, vec3(1.5, 5, 0), vec3(),      real(1), real(1), real(0.5), real(0)));
            b[0].motion = KINEMATIC;
            for (int s = 0; s < 200; s++) physics_step(b, p, h);
            real sep = (b[1].pos - b[0].pos).length();
            CHECK((b[0].pos - vec3(0, 5, 0)).length() < real(1e-9) &&  // driver owns the pose
                  std::fabs((double)b[0].vel[0] - 1.0) < 1e-6 &&       // absorbed no impulse
                  sep > real(1.99) && b[1].pos[0] > real(1.5),         // shoved clear, never overlapping
                  "role KINEMATIC: driver-owned pose, pushes but is never pushed");
        }

        // (c) collidable = false: still falls under gravity, but ignores every
        //     contact — so it drops straight through the ground plane.
        {
            const phys_params g{ real(-9.8) };
            std::vector<phys_body> b;
            b.push_back(ground_plane());
            b.push_back(ball(0, vec3(0, 5, 0), vec3(), real(1)));
            b[1].collidable = false;
            for (int s = 0; s < 600; s++) physics_step(b, g, h);
            CHECK(b[1].pos[1] < real(0),
                  "role collidable=false: falls straight through the ground");
        }
    }

    // 5. MASS. A 1 kg body moving at +1 hits a 3 kg body moving at -1 head-on,
    //    e = 0.5, in free space. The analytic 1-D result is
    //      v' = (m1 u1 + m2 u2 +- m e (u - u)) / (m1 + m2)  ->  -1.25 and -0.25.
    {
        const phys_params p{ real(0) };
        std::vector<phys_body> b;
        b.push_back(ball(0, vec3(0,   5, 0), vec3( 1,0,0), real(1), real(1), real(0.5), real(0.5)));
        b.push_back(ball(1, vec3(1.9, 5, 0), vec3(-1,0,0), real(1), real(3), real(0.5), real(0.5)));
        solve_sequential(b, p);
        CHECK(std::fabs((double)b[0].vel[0] + 1.25) < 1e-5 &&
              std::fabs((double)b[1].vel[0] + 0.25) < 1e-5,
              "mass ratio 1:3 head-on reproduces the analytic impulse");

        // The heavy body also takes the smaller share of the position push-out:
        // the correction splits as 1/m, so 3:1 here.
        std::vector<phys_body> q;
        q.push_back(ball(0, vec3(0,   5, 0), vec3(), real(1), real(1)));
        q.push_back(ball(1, vec3(1.9, 5, 0), vec3(), real(1), real(3)));
        solve_sequential(q, p);
        real moved_light = std::fabs((double)q[0].pos[0] - 0.0);
        real moved_heavy = std::fabs((double)q[1].pos[0] - 1.9);
        CHECK(moved_light > moved_heavy * real(2.5) &&
              moved_light < moved_heavy * real(3.5),
              "position push-out splits by inverse mass (light body moves ~3x)");
    }

    // 6. FRICTION is a per-CONTACT force with a surface coefficient, not a global
    //    damp — the property that fixes both bugs the height-test version had.
    {
        const phys_params p{ real(-9.8) };

        // (a) A ball sliding on the ground decelerates at about mu*g — but it does
        //     NOT stop, it starts rolling. mu = 0.5 -> a = 4.9 m/s^2 while it
        //     slides, and sliding ends at 5/7 of the launch speed (test 9 pins
        //     that transition exactly). Before B3a this asserted the ball stopped,
        //     which was the missing-rotation bug rather than a property worth
        //     keeping: friction was scrubbing away energy a real ball puts into
        //     spin. What stops it now is rolling resistance, over ~14 s.
        {
            std::vector<phys_body> b;
            b.push_back(ground_plane(real(0.5), real(0)));
            b.push_back(ball(0, vec3(0, real(0.5), 0), vec3(2,0,0), real(0.5), real(1), real(0.5), real(0)));
            b[0].rolling_friction = real(0);   // isolate friction: no rolling decay
            b[1].rolling_friction = real(0);
            for (int s = 0; s < 240; s++) physics_step(b, p, h);   // 1 s: sliding is long over
            CHECK(std::fabs((double)b[1].vel[0] - 10.0/7.0) < 1e-4,
                  "friction: a sliding ball decelerates at ~mu*g, then rolls at 5/7 of its speed");
        }
        // (b) A frictionless surface never damps it — but only under a rule with
        //     an absorbing zero, so this one names its rule instead of taking the
        //     AVERAGE default, under which mu = (0 + 0.5)/2 = 0.25 and the ball
        //     WOULD slow down. That is the trade PhysX's default makes.
        {
            phys_params pg{ real(-9.8) }; pg.friction_combine = COMBINE_GEOMETRIC;
            std::vector<phys_body> b;
            b.push_back(ground_plane(real(0), real(0)));
            b.push_back(ball(0, vec3(0, real(0.5), 0), vec3(2,0,0), real(0.5), real(1), real(0.5), real(0)));
            for (int s = 0; s < 240; s++) physics_step(b, pg, h);
            CHECK(std::fabs((double)b[1].vel[0] - 2.0) < 1e-3,
                  "friction: under GEOMETRIC a zero coefficient slides forever");
        }
        // (b2) The same setup on the AVERAGE default DOES damp — pinning the one
        //      behavioural consequence of matching PhysX.
        {
            std::vector<phys_body> b;
            b.push_back(ground_plane(real(0), real(0)));
            b.push_back(ball(0, vec3(0, real(0.5), 0), vec3(2,0,0), real(0.5), real(1), real(0.5), real(0)));
            for (int s = 0; s < 240; s++) physics_step(b, p, h);
            CHECK(b[1].vel[0] < real(1.9),
                  "friction: under the AVERAGE default a zero surface still grips");
        }
        // (c) A ball in FREE FALL is never damped, however fast it moves sideways
        //     — the old height-test damp did this to anything below y = radius.
        {
            std::vector<phys_body> b;
            b.push_back(ball(0, vec3(0, real(-5), 0), vec3(2,0,0), real(0.5)));  // below y=0, touching nothing
            for (int s = 0; s < 240; s++) physics_step(b, p, h);
            CHECK(std::fabs((double)b[0].vel[0] - 2.0) < 1e-6,
                  "friction: a body touching nothing is never damped");
        }
        // (d) Friction acts wherever contact happens, not just near y=0: a ball
        //     launched on TOP of a box goes through the same slide-then-roll
        //     transition (the old height test missed this contact entirely).
        {
            std::vector<phys_body> b;
            b.push_back(static_box(vec3(0, 0.5, 0), vec3(4, 0.5, 4), real(0.5), real(0)));  // wide slab, top y=1
            b.push_back(ball(0, vec3(0, real(1.5), 0), vec3(2,0,0), real(0.5), real(1), real(0.5), real(0)));
            b[0].rolling_friction = real(0);
            b[1].rolling_friction = real(0);
            for (int s = 0; s < 240; s++) physics_step(b, p, h);
            CHECK(std::fabs((double)b[1].vel[0] - 10.0/7.0) < 1e-4,
                  "friction: a ball on a box slides and then rolls too, not just on the ground");
        }
    }

    // 7. COMBINE RULES. Friction and restitution are pair properties synthesised
    //    from two per-surface numbers, so the rule is configurable. These pin the
    //    three properties the choice actually turns on.
    {
        const real a = real(0.8), b = real(0.2);
        CHECK(std::fabs((double)combine(a, b, COMBINE_MULTIPLY)  - 0.16) < 1e-5 &&
              std::fabs((double)combine(a, b, COMBINE_MIN)       - 0.2)  < 1e-5 &&
              std::fabs((double)combine(a, b, COMBINE_GEOMETRIC) - 0.4)  < 1e-5 &&
              std::fabs((double)combine(a, b, COMBINE_AVERAGE)   - 0.5)  < 1e-5 &&
              std::fabs((double)combine(a, b, COMBINE_MAX)       - 0.8)  < 1e-5,
              "combine: each rule computes its formula");

        // The enum is ordered by increasing result, so the dropdown reads
        // deadest -> bounciest. Holds for any coefficients in [0,1].
        CHECK(combine(a,b,COMBINE_MULTIPLY) <= combine(a,b,COMBINE_MIN) &&
              combine(a,b,COMBINE_MIN)      <= combine(a,b,COMBINE_GEOMETRIC) &&
              combine(a,b,COMBINE_GEOMETRIC)<= combine(a,b,COMBINE_AVERAGE) &&
              combine(a,b,COMBINE_AVERAGE)  <= combine(a,b,COMBINE_MAX),
              "combine: enum order is monotonic in the result");

        // IDENTITY WHEN EQUAL — a uniform scene behaves exactly as authored.
        // Everything but MULTIPLY has it (0.7*0.7 = 0.49, a scene-wide change).
        const real e = real(0.7);
        CHECK(std::fabs((double)combine(e,e,COMBINE_GEOMETRIC) - 0.7) < 1e-5 &&
              std::fabs((double)combine(e,e,COMBINE_AVERAGE)   - 0.7) < 1e-5 &&
              std::fabs((double)combine(e,e,COMBINE_MIN)       - 0.7) < 1e-5 &&
              std::fabs((double)combine(e,e,COMBINE_MAX)       - 0.7) < 1e-5 &&
              std::fabs((double)combine(e,e,COMBINE_MULTIPLY)  - 0.49)< 1e-5,
              "combine: identity when both surfaces match (all but multiply)");

        // ABSORBING ZERO — a dead/slippery object can be authored whatever it
        // touches. MULTIPLY, MIN and GEOMETRIC have it; AVERAGE and MAX do not.
        CHECK(combine(real(0), e, COMBINE_GEOMETRIC) == real(0) &&
              combine(real(0), e, COMBINE_MULTIPLY)  == real(0) &&
              combine(real(0), e, COMBINE_MIN)       == real(0) &&
              combine(real(0), e, COMBINE_AVERAGE)   >  real(0) &&
              combine(real(0), e, COMBINE_MAX)       >  real(0),
              "combine: zero on one surface forces zero (multiply/min/geometric only)");

        // And it reaches the solver: the SAME pair bounces differently under two
        // rules. A bouncy ball (0.9) dropped on a dead floor (0.0) rebounds under
        // MAX and stays put under GEOMETRIC.
        auto rebound = [&](combine_mode m) {
            phys_params pp{ real(-9.8) }; pp.restitution_combine = m;
            std::vector<phys_body> v;
            v.push_back(ground_plane(real(0.5), real(0)));                      // dead floor
            v.push_back(ball(0, vec3(0, real(2), 0), vec3(), real(0.5),
                             real(1), real(0.5), real(0.9)));                   // bouncy ball
            real top = 0;
            for (int s = 0; s < 480; s++) {
                physics_step(v, pp, h);
                if (v[1].vel[1] > real(0) && v[1].pos[1] > top) top = v[1].pos[1];
            }
            return top;
        };
        real max_top = rebound(COMBINE_MAX), geo_top = rebound(COMBINE_GEOMETRIC);
        printf("  rebound height: MAX %.3f, GEOMETRIC %.3f\n", (double)max_top, (double)geo_top);
        CHECK(max_top > real(0.8) && geo_top < real(0.6),
              "combine: the rule reaches the solver (bouncy ball on a dead floor)");
    }

    // 8. CONVEX-CONVEX (B2): support functions + GJK/EPA, in src/physics/gjk.h. These
    //    are what let ANY pair of convex colliders collide from one code path,
    //    where before every pair needed its own analytic test and box-box
    //    silently had none.
    {
        const double SQRT2 = 1.41421356237309504880;

        // 8a. support(): the only thing GJK and EPA ever ask a shape. Everything
        //     downstream is wrong if this is wrong, and it is three lines, so it
        //     is pinned directly.
        {
            phys_body s = ball(-1, vec3(1, 2, 3), vec3(), real(2));
            vec3 sp = support(s, vec3(0, 5, 0));                 // length must not matter
            CHECK(std::fabs((double)sp[0] - 1) < 1e-5 &&
                  std::fabs((double)sp[1] - 4) < 1e-5 &&
                  std::fabs((double)sp[2] - 3) < 1e-5,
                  "support: a sphere answers with its centre plus one radius along the direction");

            phys_body bx = static_box(vec3(0, 0, 0), vec3(1, 2, 3));
            vec3 bp = support(bx, vec3(1, -1, 1));
            CHECK(bp[0] == real(1) && bp[1] == real(-2) && bp[2] == real(3),
                  "support: a box answers with the corner extreme along each of its own axes");

            // The turned box is the case that makes the whole approach worth it:
            // its support reaches a CORNER at sqrt(2), where a world bounding box
            // would report a flat face there. Same fact B1e fixed by hand for
            // sphere-box, now falling out of the support function for every pair.
            phys_body r45 = rotated_box_y(vec3(0, 0, 0), vec3(1, 1, 1), real(45));
            vec3 rp = support(r45, vec3(1, 0, 0));
            CHECK(std::fabs((double)rp[0] - SQRT2) < 1e-4,
                  "support: a turned box reaches out to its corner, not its face");
        }

        // 8b. GJK's yes/no, on the pair that had no test at all before B2.
        {
            vec3 n; real pen;
            CHECK(!gjk_epa_contact(static_box(vec3(0,0,0), vec3(1,1,1)),
                                   static_box(vec3(3,0,0), vec3(1,1,1)), n, pen),
                  "gjk: two separated boxes do not touch");
            CHECK(gjk_epa_contact(static_box(vec3(0,0,0), vec3(1,1,1)),
                                  static_box(vec3(real(1.9),0,0), vec3(1,1,1)), n, pen),
                  "gjk: two overlapping boxes do touch");
        }

        // 8c. EPA's depth and normal. Each case has an answer computable by hand,
        //     which is the point of choosing them.
        {
            vec3 n; real pen;
            // Overlapping 0.1 along x. A is at the origin, so A must move -x.
            gjk_epa_contact(static_box(vec3(0,0,0), vec3(1,1,1)),
                            static_box(vec3(real(1.9),0,0), vec3(1,1,1)), n, pen);
            CHECK(std::fabs((double)n[0] + 1) < 1e-3 && std::fabs((double)pen - 0.1) < 1e-3,
                  "epa: box overlapping a box along x gives normal -x, depth 0.1");

            // Face-to-face and axis-aligned — the configuration most likely to
            // degenerate, because so many support points tie.
            gjk_epa_contact(static_box(vec3(0, real(1.95), 0), vec3(1,1,1)),
                            static_box(vec3(0, 0, 0), vec3(1,1,1)), n, pen);
            CHECK(std::fabs((double)n[1] - 1) < 1e-3 && std::fabs((double)pen - 0.05) < 1e-3,
                  "epa: a box resting squarely on a box gives normal +y, depth 0.05");

            // A 45-degree box driven corner-first into a face. Its corner reaches
            // x = sqrt(2); the other box's face is at 1.3.
            gjk_epa_contact(rotated_box_y(vec3(0,0,0), vec3(1,1,1), real(45)),
                            static_box(vec3(real(2.3),0,0), vec3(1,1,1)), n, pen);
            CHECK(std::fabs((double)n[0] + 1) < 1e-3 &&
                  std::fabs((double)pen - (SQRT2 - 1.3)) < 2e-3,
                  "epa: a turned box driven corner-first reports the corner's depth");

            // COINCIDENT CENTRES. A small box entirely inside a big one, sharing
            // its centre exactly. The shallowest way out is +/-y, through the
            // 0.8 half-height, so the depth is 0.8 + 0.3 = 1.1.
            //
            // This is the case that caught EPA orienting its faces from the
            // ORIGIN: with the centres exactly equal, the origin can land on a
            // starting face's plane, which leaves that face's direction
            // undetermined and its distance 0 — permanently the nearest face, and
            // one the convergence test accepts immediately. It reported zero
            // penetration for a 1.98-deep overlap, and moving the centres apart
            // by 0.001 hid it completely. Faces are oriented from the
            // tetrahedron's centroid now, which cannot lie on one of its own
            // faces.
            bool concentric = gjk_epa_contact(
                static_box(vec3(0,0,0), vec3(real(0.5), real(0.3), real(0.6))),
                static_box(vec3(0,0,0), vec3(real(1), real(0.8), real(1.2))), n, pen);
            CHECK(concentric && std::fabs((double)pen - 1.1) < 1e-3 &&
                  std::fabs((double)std::fabs((double)n[1]) - 1) < 1e-3,
                  "epa: two boxes sharing a centre exactly report the shallowest way out");
        }

        // 8d. The two paths must agree. contact_detect takes the analytic
        //     sphere-box test; gjk_epa_contact takes the general one. Same
        //     configuration, same answer — that is what makes the fast path a
        //     cost decision rather than a second implementation to keep in sync.
        //
        //     The normal tolerance is 0.01, not 1e-5 like the depth: EPA's
        //     tolerance is on DISTANCE, which bounds the ANGLE only to
        //     sqrt(2*TOL/r) on a curved surface. Measured worst case here is
        //     0.0063 rad (0.36 degrees), identical in float and double. See the
        //     table at EPA_TOL in src/physics/gjk.h.
        {
            double worst_n = 0, worst_pen = 0, worst_swapped = 0;
            int disagreements = 0, touching = 0;
            for (int i = 0; i < 12; i++)
            for (int j = 0; j < 12; j++)
            for (int k = 0; k < 12; k++) {
                vec3 c(real(-1.6 + 0.29*i), real(-1.6 + 0.29*j), real(-1.6 + 0.29*k));
                phys_body S = ball(-1, c, vec3(), real(0.5));
                phys_body B = rotated_box_y(vec3(0,0,0), vec3(1, real(0.6), real(0.8)), real(30));

                vec3 n1, n2; real p1, p2;
                bool h1 = contact_detect(S, B, n1, p1);      // analytic fast path
                bool h2 = gjk_epa_contact(S, B, n2, p2);      // general path
                if (h1 != h2) { disagreements++; continue; }
                if (!h1) continue;
                touching++;
                double dn = (double)(n1 - n2).length();
                double dp = std::fabs((double)p1 - (double)p2);
                if (dn > worst_n)   worst_n = dn;
                if (dp > worst_pen) worst_pen = dp;

                // Handed in the other way round, the box lands in slot A and
                // there is no analytic test for it, so it goes through GJK/EPA.
                // The answer must be the same collision with the normal flipped.
                vec3 n3; real p3;
                if (contact_detect(B, S, n3, p3)) {
                    double d = (double)(n1 + n3).length() + std::fabs((double)p1 - (double)p3);
                    if (d > worst_swapped) worst_swapped = d;
                } else disagreements++;
            }
            printf("  sphere-box: %d touching of 1728, worst |dn| %.5f, worst |dpen| %.7f,"
                   " worst swapped-order error %.5f\n",
                   touching, worst_n, worst_pen, worst_swapped);
            CHECK(disagreements == 0 && touching > 400,
                  "analytic and convex-convex agree on WHETHER a sphere and box touch");
            CHECK(worst_n < 0.01 && worst_pen < 1e-4,
                  "analytic and convex-convex agree on the normal and the depth");
            CHECK(worst_swapped < 0.02,
                  "the pair order does not change the collision, only the normal's sign");
        }

        // 8e. The narrow phase emits box-box now. Before B2 contact_detect
        //     returned false for any pair without a sphere, so two overlapping
        //     boxes produced NO contact and passed through each other.
        {
            std::vector<phys_body> b;
            b.push_back(static_box(vec3(0, 0, 0), vec3(1, 1, 1)));
            b.push_back(dynamic_box(vec3(0, real(1.9), 0), vec3(1, 1, 1)));
            std::vector<contact> C;
            build_contacts(b, C);
            // Neither body is a sphere, so there is nothing to canonicalise and
            // the pair keeps scan order. Assert the CONVENTION the solver relies
            // on instead of that order: n points from b toward a, so it agrees
            // in sign with the separation of their centres.
            bool oriented = C.size() >= 1;
            for (const contact& k : C)
                oriented = oriented && dot(k.n, b[k.a].pos - b[k.b].pos) > real(0) &&
                           std::fabs((double)std::fabs((double)k.n[1]) - 1) < 1e-3;
            CHECK(oriented,
                  "build_contacts emits a box-box contact, with n from b toward a");

            // B3b: face-on, the overlap is a SQUARE, not a point, so the pair
            // emits one contact per corner of it. That is the whole reason a
            // resting box stops rocking — a single point is a pivot.
            CHECK(C.size() == 4,
                  "a face-on box-box overlap emits a 4-point manifold");
            bool on_face = true;
            for (const contact& k : C) {
                // Every point lies on the shared face (y = 1 between boxes whose
                // faces are at y = 1 and y = 0.9) and inside the square.
                on_face = on_face && std::fabs((double)k.p[1] - 1.0) < 0.11 &&
                          std::fabs((double)k.p[0]) <= 1.001 && std::fabs((double)k.p[2]) <= 1.001 &&
                          k.mcount == 4;
                }
            CHECK(on_face,
                  "and each point is on the shared face, inside the overlap square");
        }

        // 8f. End to end: a box falls under gravity and comes to rest on another
        //     box. Half-extent 0.5 on a top face at y = 0, so it rests at 0.5.
        {
            std::vector<phys_body> b;
            b.push_back(ground_plane(real(0.5), real(0)));
            b.push_back(dynamic_box(vec3(0, real(3), 0), vec3(real(0.5), real(0.5), real(0.5)),
                                    real(0.5), real(0)));
            const phys_params p{ real(-9.8) };
            real maxv = 0;
            for (int s = 0; s < 2400; s++) maxv = physics_step(b, p, h);
            printf("  dropped box rests at y = %.5f (expect 0.5), max |v| = %.5f\n",
                   (double)b[1].pos[1], (double)maxv);
            CHECK(std::fabs((double)b[1].pos[1] - 0.5) < 2e-3,
                  "a dropped box comes to rest on the surface, not through it");
            CHECK(maxv < real(0.02), "and settles rather than buzzing");
        }
    }

    // 9. ROTATION: spheres carry angular velocity, friction spins them up, and a
    //    ball ROLLS instead of being scrubbed to a halt (B3a). Boxes carry a
    //    quaternion and a real inertia tensor, and take torque through their
    //    contact manifold (B3b/B3c).
    //
    //    Most of these have closed-form answers, because a uniform sphere on a
    //    flat surface is one of the few rigid-body problems that does.
    {
        const real G = real(9.8);
        const phys_params p{ -G };

        // 9a. Inverse inertia, the angular counterpart of inv_mass. A solid
        //     sphere is I = (2/5) m r^2, so I^-1 = 2.5 * inv_mass / r^2 — and the
        //     same role gate applies, since a body the solver cannot push must
        //     not be spinnable either.
        {
            phys_body s = ball(0, vec3(0,0,0), vec3(), real(0.5), real(2));  // m = 2, r = 0.5
            vec3 got = delta_omega(s, vec3(1, 0, 0));
            CHECK(std::fabs((double)got[0] - 2.5 * 0.5 / 0.25) < 1e-6,
                  "inv_inertia: a solid sphere's is 2.5 * inv_mass / r^2");

            phys_body st = s; st.motion = STATIC;
            CHECK(delta_omega(st, vec3(1,0,0)).near_zero(),
                  "inv_inertia: an immovable body cannot be spun, whatever its mass");

            // B3c: a box now has a real tensor. STATIC still returns zero — the
            // role gate is above the shape test and outranks it.
            CHECK(delta_omega(static_box(vec3(0,0,0), vec3(1,1,1)), vec3(1,0,0)).near_zero(),
                  "inv_inertia: an immovable box is still rotation-free (role outranks shape)");

            // A box of half-extents h has I_xx = (1/3) m (h_y^2 + h_z^2), so the
            // inverse is 3 * inv_mass / (h_y^2 + h_z^2) — ANISOTROPIC, unlike a
            // sphere: this one resists turning about x (the long axis) least.
            {
                phys_body bx = dynamic_box(vec3(0,0,0), vec3(2, 1, 1));   // m = 1
                vec3 gx = delta_omega(bx, vec3(1, 0, 0));
                vec3 gy = delta_omega(bx, vec3(0, 1, 0));
                // 1e-6, not 1e-9: 3/5 is not exactly representable in binary, so
                // the float build lands an ulp off. The sphere assertions above
                // use the same bound for the same reason.
                CHECK(std::fabs((double)gx[0] - 3.0 / (1.0 + 1.0)) < 1e-6,
                      "inv_inertia: a box about its own x is 3 * inv_mass / (hy^2 + hz^2)");
                CHECK(std::fabs((double)gy[1] - 3.0 / (4.0 + 1.0)) < 1e-6,
                      "inv_inertia: and about y is 3 * inv_mass / (hx^2 + hz^2) — anisotropic");
                CHECK((double)gx[0] > (double)gy[1],
                      "inv_inertia: a long box turns most easily about its long axis");
            }

            // The tensor lives in the BODY frame: turn the box 90 degrees about
            // y and the axis that was easy to spin about is now world z.
            {
                phys_body bx = dynamic_box(vec3(0,0,0), vec3(2, 1, 1));
                phys_body rb = bx;
                set_orientation(rb, quat_from_axis_angle(vec3(0,1,0), real(1.5707963267948966)));
                vec3 turned = delta_omega(rb, vec3(0, 0, 1));   // world z == body x now
                vec3 flat   = delta_omega(bx, vec3(1, 0, 0));   // body x when unturned
                CHECK(std::fabs((double)turned[2] - (double)flat[0]) < 1e-6,
                      "inv_inertia: the tensor rotates with the body (R I^-1 R^T)");
            }

            // The Euler pair is an exact inverse: what the viewer seeds from is
            // what the write-back hands back to the transform.
            {
                const vec3 e(10, 20, 30);
                const vec3 back = quat_to_euler_zyx_degrees(quat_from_euler_zyx_degrees(e));
                CHECK((back - e).length() < real(1e-4),
                      "euler -> quat -> euler round-trips (10, 20, 30) degrees");
            }
        }

        // 9b. A sphere's lever arm is PARALLEL to the contact normal, so the
        //     normal impulse exerts no torque and the normal effective mass is
        //     untouched. This is why bouncing is bit-for-bit what it was, and why
        //     only friction can spin a ball.
        {
            phys_body s = ball(0, vec3(0, real(0.5), 0), vec3(), real(0.5));
            vec3 up(0, 1, 0);
            vec3 cp(0, 0, 0);                      // the contact point, on the floor
            vec3 r = contact_lever(s, up, cp, true);
            phys_body none = ground_plane();
            vec3 rb = contact_lever(none, up, cp, false);
            CHECK(cross(r, up).near_zero() && std::fabs((double)r[1] + 0.5) < 1e-9,
                  "lever arm: one radius against the normal, so it exerts no torque along n");
            CHECK(std::fabs((double)inv_effective_mass(s, none, r, rb, up) - 1.0) < 1e-9,
                  "effective mass along the normal is unchanged by rotation (1/m exactly)");
            // Along the tangent each spinnable side adds 2.5 * inv_mass, so a ball
            // on an immovable floor is 3.5x as mobile as it was when it could only
            // slide. That factor is the whole of the friction change.
            CHECK(std::fabs((double)inv_effective_mass(s, none, r, rb, vec3(1,0,0)) - 3.5) < 1e-9,
                  "effective mass along the tangent gains 2.5/m: the cost of spinning up");
        }

        // 9c. THE ANALYTIC CASE. A ball launched sliding at v0 on friction mu is
        //     decelerated at mu*g while its spin builds, until the contact point
        //     stops slipping. Conserving angular momentum about the contact point
        //     gives rolling at exactly 5/7 of v0, reached at t = 2*v0/(7*mu*g) —
        //     independent of mass and radius. Rolling resistance is off here so
        //     the coast afterwards is exact.
        {
            const real v0 = 2, mu = real(0.5);
            std::vector<phys_body> b;
            b.push_back(ground_plane(mu, real(0)));
            b.push_back(ball(0, vec3(0, real(0.5), 0), vec3(v0,0,0), real(0.5), real(1), mu, real(0)));
            b[0].rolling_friction = real(0); b[1].rolling_friction = real(0);
            for (int s = 0; s < 240; s++) physics_step(b, p, h);      // 1 s >> t_roll = 0.117 s
            const double v_roll = 5.0 * (double)v0 / 7.0;
            printf("  slide->roll: vx = %.6f (analytic %.6f), rolling error %.2e\n",
                   (double)b[1].vel[0], v_roll,
                   std::fabs((double)(b[1].vel[0] + b[1].omega[2] * b[1].radius)));
            CHECK(std::fabs((double)b[1].vel[0] - v_roll) < 1e-4,
                  "a sliding ball transitions to rolling at exactly 5/7 of its launch speed");
            CHECK(std::fabs((double)(b[1].vel[0] + b[1].omega[2] * b[1].radius)) < 1e-6,
                  "and rolls without slipping: the contact point is stationary");

            // Coulomb friction alone can never stop it. Once the contact point is
            // stationary there is nothing left for friction to oppose, so with
            // rolling resistance off the ball coasts forever.
            for (int s = 0; s < 4800; s++) physics_step(b, p, h);     // 20 more seconds
            CHECK(std::fabs((double)b[1].vel[0] - v_roll) < 1e-4,
                  "friction alone never stops a rolling ball — it coasts indefinitely");
        }

        // 9d. Which is what ROLLING RESISTANCE is for. A rolling sphere sheds
        //     speed at (5/7) * mu_r * g: the resisting couple acts about the
        //     contact point, where the ball's angular momentum is (7/5) m v R.
        {
            const real mur = real(0.02);
            std::vector<phys_body> b;
            b.push_back(ground_plane(real(0.5), real(0)));
            b.push_back(ball(0, vec3(0, real(0.5), 0), vec3(1,0,0), real(0.5), real(1),
                             real(0.5), real(0)));
            b[0].rolling_friction = mur; b[1].rolling_friction = mur;
            b[1].omega = vec3(0, 0, real(-2));                        // launched already rolling
            int stopped = -1;
            for (int s = 1; s <= 4800 && stopped < 0; s++) {
                physics_step(b, p, h);
                if (b[1].vel.length() < real(0.01)) stopped = s;
            }
            const double predicted = 1.0 / (5.0 / 7.0 * (double)mur * (double)G);
            printf("  rolling resistance mu_r = %.3f: stops from 1 m/s in %.2f s (analytic %.2f s)\n",
                   (double)mur, stopped < 0 ? -1.0 : stopped * (double)h, predicted);
            CHECK(stopped > 0 && std::fabs(stopped * (double)h - predicted) < 0.2,
                  "rolling resistance decelerates a rolling ball at (5/7) * mu_r * g");
        }

        // 9d1. The resistance lever is the ROLLING body's radius. The viewer's
        //      floors are radius-1000 spheres; taking the larger radius of the
        //      pair made the clamp 2000x too strong and pinned every ball's spin
        //      to zero, so balls slid to a stop like blocks. Same run as 9d on a
        //      sphere floor. The floor curves away under the ball, adding a
        //      downhill push (5/7) g x/R, so with u = mu_r R - x the motion is
        //      u'' = w^2 u, w = sqrt((5/7) g / R): the ball stops at
        //      t = atanh(v0 / (mu_r R w)) / w, 8.24 s instead of the flat 7.14 s.
        {
            const real mur = real(0.02), Rf = real(1000);
            std::vector<phys_body> b;
            b.push_back(ball(-1, vec3(0, -Rf, 0), vec3(0,0,0), Rf, real(1),
                             real(0.5), real(0)));
            b[0].motion = STATIC;
            b.push_back(ball(0, vec3(0, real(0.5), 0), vec3(1,0,0), real(0.5), real(1),
                             real(0.5), real(0)));
            b[0].rolling_friction = mur; b[1].rolling_friction = mur;
            b[1].omega = vec3(0, 0, real(-2));
            int stopped = -1;
            for (int s = 1; s <= 4800 && stopped < 0; s++) {
                physics_step(b, p, h);
                if (b[1].vel.length() < real(0.01)) stopped = s;
            }
            const double w = std::sqrt(5.0 / 7.0 * (double)G / (double)Rf);
            const double predicted = std::atanh(1.0 / ((double)mur * (double)Rf * w)) / w;
            printf("  on a radius-1000 sphere floor: stops in %.2f s (analytic %.2f s)\n",
                   stopped < 0 ? -1.0 : stopped * (double)h, predicted);
            CHECK(stopped > 0 && std::fabs(stopped * (double)h - predicted) < 0.2,
                  "rolling resistance uses the rolling ball's radius, not a sphere floor's");
        }

        // 9d2. SPINNING resistance is a separate axis from rolling. Rolling
        //      resists the spin ORTHOGONAL to the contact normal — the part that
        //      carries a ball along. Spinning resists the spin ABOUT it — a ball
        //      turning on the spot, which goes nowhere.
        //
        //      They resist through different mechanisms across different lengths
        //      (deformation over the ball's radius against torsion over the
        //      contact patch), so one coefficient for both over-damps the
        //      spinning one. With no linear coupling to slow it, top-spin decays
        //      at 2.5 * mu_s * g / R — no 5/7, which only appears when the
        //      rolling constraint is being maintained.
        {
            // Each coefficient must be UNABLE to touch the other's axis: this is
            // what makes them two knobs rather than one with extra steps.
            for (int which = 0; which < 2; which++) {
                const bool spinning = (which == 0);
                std::vector<phys_body> b;
                b.push_back(ground_plane(real(0.5), real(0)));
                b.push_back(ball(0, vec3(0, real(0.5), 0),
                                 spinning ? vec3(0,0,0) : vec3(0,0,5), real(0.5), real(1),
                                 real(0.5), real(0)));
                // Zero the coefficient that should NOT be able to act here.
                for (phys_body& q : b) {
                    q.rolling_friction  = spinning ? real(0.01) : real(0);
                    q.spinning_friction = spinning ? real(0)    : real(0.01);
                }
                b[1].omega = spinning ? vec3(0, 10, 0)    // about the normal
                                      : vec3(10, 0, 0);   // across it
                for (int s = 0; s < 240 * 120; s++) physics_step(b, p, h);
                CHECK(b[1].omega.length() > real(5),
                      spinning ? "rolling resistance cannot slow spin about the normal"
                               : "spinning resistance cannot slow spin across the normal");
            }

            // And the rate on the axis that IS resisted.
            const real mus = real(0.01);
            std::vector<phys_body> b;
            b.push_back(ground_plane(real(0.5), real(0)));
            b.push_back(ball(0, vec3(0, real(0.5), 0), vec3(), real(0.5), real(1),
                             real(0.5), real(0)));
            for (phys_body& q : b) q.spinning_friction = mus;
            b[1].omega = vec3(0, 10, 0);
            int stopped = -1;
            for (int s = 1; s <= 240 * 120 && stopped < 0; s++) {
                physics_step(b, p, h);
                if (b[1].omega.length() < real(0.5)) stopped = s;
            }
            const double predicted = 9.5 / (2.5 * (double)mus * (double)G / 0.5);
            printf("  top-spin mu_s = %.3f: 10 -> 0.5 rad/s in %.1f s (analytic %.1f s)\n",
                   (double)mus, stopped < 0 ? -1.0 : stopped * (double)h, predicted);
            CHECK(stopped > 0 && std::fabs(stopped * (double)h - predicted) < 0.5,
                  "spinning resistance decelerates a top-spinning ball at 2.5 * mu_s * g / r");
        }

        // 9e. A box TAKES SPIN now (B3c), and the manifold (B3b) is what keeps
        //     that from turning every slide into a tumble. Before this phase the
        //     first assertion read "a box never acquires spin" — the flip is the
        //     deliberate change the old comment was reserving.
        {
            // A sliding box still stops dead, and stays flat while doing it. The
            // friction impulse acts at the FLOOR, below the centre of mass, so it
            // torques the box forward; four contact points, each free to carry
            // its own normal load, are what resist the tip.
            std::vector<phys_body> b;
            b.push_back(ground_plane(real(0.5), real(0)));
            b.push_back(dynamic_box(vec3(0, real(0.5), 0), vec3(real(0.5), real(0.5), real(0.5)),
                                    real(0.5), real(0)));
            b[1].vel = vec3(2, 0, 0);
            for (int s = 0; s < 480; s++) physics_step(b, p, h);
            printf("  sliding box: vx = %.4f, |omega| = %.4f, up.y = %.5f\n",
                   (double)b[1].vel[0], (double)b[1].omega.length(), (double)b[1].axes[1][1]);
            CHECK(std::fabs((double)b[1].vel[0]) < 0.05,
                  "a sliding box still stops dead");
            CHECK((double)b[1].axes[1][1] > 0.999,
                  "and stays flat doing it: the 4-point manifold resists the friction torque");

            // Off-centre load DOES turn a box: one hanging over the edge of a
            // platform tips off it. This is the pairing B3b+B3c exists for —
            // the manifold says WHERE the support is, the tensor turns that into
            // rotation.
            std::vector<phys_body> t;
            t.push_back(static_box(vec3(0, real(0.5), 0), vec3(1, real(0.5), 1),
                                   real(0.8), real(0)));            // platform, top at y = 1
            t.push_back(dynamic_box(vec3(real(1.3), real(1.5), 0),  // overhanging its +x edge
                                    vec3(real(0.5), real(0.5), real(0.5)), real(0.8), real(0)));
            // Half a second — while it is still ON the platform. Left to run it
            // slides off and free-falls, which would satisfy any "did it turn"
            // test for the wrong reason.
            for (int s = 0; s < 120; s++) physics_step(t, p, h);
            printf("  overhanging box after 0.5 s: up.y = %.4f, |omega| = %.4f\n",
                   (double)t[1].axes[1][1], (double)t[1].omega.length());
            CHECK(t[1].omega.length() > real(0.1) && (double)t[1].axes[1][1] < 0.999,
                  "a box supported off-centre tips: the manifold's geometry becomes torque");
        }

        // 9f. Spin alone keeps a body awake. A ball spinning on the spot has zero
        //     centre velocity, and reporting it as at rest would be wrong: the
        //     moment it touches anything, friction turns that spin into motion.
        {
            std::vector<phys_body> b;
            b.push_back(ball(0, vec3(0, 5, 0), vec3(), real(0.5)));
            b[0].omega = vec3(0, 0, 4);                               // 4 rad/s, r = 0.5
            const phys_params zero_g{ real(0) };
            real maxv = physics_step(b, zero_g, h);
            CHECK(std::fabs((double)maxv - 2.0) < 1e-6,
                  "the sleep metric is a SURFACE speed: |omega| * r counts, not just |v|");
        }
    }

    // 10. CONVEX HULL (B4 step 1): a hull is a shared vertex/face list in the body's
    //     frame; support() scans its vertices. A box-shaped hull must be
    //     indistinguishable from the analytic box wherever GJK/EPA is the path.
    {
        const hull_shape cube = box_hull(vec3(1, 2, 3));
        CHECK(cube.faces.size() == 6 && cube.faces[0].loop.size() == 4, "hull: a box hull has six four-vertex faces");
        CHECK(wound_outward(cube), "hull: every face loop is counter-clockwise seen from outside");

        phys_body hb = static_hull(vec3(0, 0, 0), cube), bx = static_box(vec3(0, 0, 0), vec3(1, 2, 3));
        // On a direction with no ties the support POINT must match; along a face
        // normal several corners tie and the two shapes may pick different ones, so
        // there only the support VALUE (the projection GJK consumes) is compared.
        const vec3 exact[] = { vec3(1, -1, 1), vec3(-3, 2, real(0.5)) }, tied[] = { vec3(0, 1, 0), vec3(0, 0, -1) };
        bool same = true;
        for (const vec3& d : exact) same &= (support(hb, d) - support(bx, d)).length() < real(1e-9);
        for (const vec3& d : tied)  same &= std::fabs((double)(dot(support(hb, d), d) - dot(support(bx, d), d))) < 1e-9;
        CHECK(same, "hull support: a box hull answers with the analytic box's corners");

        phys_body hr = hb, br = rotated_box_y(vec3(0, 0, 0), vec3(1, 2, 3), real(45));
        set_orientation(hr, quat_from_euler_zyx_degrees(vec3(0, 45, 0)));
        CHECK(std::fabs((double)(dot(support(hr, vec3(1, 0, 0)), vec3(1, 0, 0)) - dot(support(br, vec3(1, 0, 0)), vec3(1, 0, 0)))) < 1e-6,
              "hull support: turned 45 degrees it reaches as far as the turned box's corner");

        vec3 nh, nb; real ph, pb;                                    // sphere overlapping the top face by 0.5
        phys_body sp = ball(-1, vec3(0, real(2.5), 0), vec3(), real(1));
        bool hit_h = gjk_epa_contact(sp, hb, nh, ph), hit_b = gjk_epa_contact(sp, bx, nb, pb);
        CHECK(hit_h && hit_b && (nh - nb).length() < real(1e-4) && std::fabs((double)(ph - pb)) < 1e-4,
              "gjk/epa: sphere vs box hull gives the box's normal and depth");
        CHECK(std::fabs((double)nh[1] - 1) < 1e-3 && std::fabs((double)ph - 0.5) < 1e-3,
              "gjk/epa: sphere on the hull's top face: normal +y, depth 0.5");

        hull_shape unit = box_hull(vec3(1, 1, 1));                   // two hulls overlapping 0.1 along x, as test 8c's boxes
        vec3 n; real pen;
        CHECK(gjk_epa_contact(static_hull(vec3(0, 0, 0), unit), static_hull(vec3(real(1.9), 0, 0), unit), n, pen)
              && std::fabs((double)n[0] + 1) < 1e-3 && std::fabs((double)pen - 0.1) < 1e-3,
              "gjk/epa: hull overlapping a hull along x gives normal -x, depth 0.1");
        CHECK(!gjk_epa_contact(static_hull(vec3(0, 0, 0), unit), static_hull(vec3(3, 0, 0), unit), n, pen),
              "gjk: two separated hulls do not touch");

        // 10b. Mass properties (B4 step 2): the polyhedron integrals must reproduce
        //      the box's closed forms, and delta_omega must then match the box's
        //      branch in every frame — the tensor is the only thing that changes
        //      between a hull and the box it was built from.
        {
            hull_shape off = box_hull(vec3(2, 1, 1));            // authored 1,2,3 away from its own centre
            for (vec3& v : off.verts) v += vec3(1, 2, 3);
            const vec3 com = hull_mass_properties(off);
            CHECK((com - vec3(1, 2, 3)).length() < real(1e-6) && std::fabs((double)off.radius - std::sqrt(6.0)) < 1e-6,
                  "hull mass: the centre of mass of an off-centre box hull is where the box was, and the vertices move onto it");
            printf("  off-centre box hull: inv_inertia diag %.7f %.7f %.7f, off-diag %.2e %.2e %.2e\n",
                   (double)off.inv_inertia[0][0], (double)off.inv_inertia[1][1], (double)off.inv_inertia[2][2],
                   (double)off.inv_inertia[0][1], (double)off.inv_inertia[1][2], (double)off.inv_inertia[0][2]);
            // 1e-5: the float build integrates about the authored origin (terms up to
            // 5^3) and subtracts the centre-of-mass shift, so ~6 digits survive.
            CHECK(std::fabs((double)off.inv_inertia[0][0] - 3.0 / (1.0 + 1.0)) < 1e-5 && std::fabs((double)off.inv_inertia[1][1] - 3.0 / (4.0 + 1.0)) < 1e-5
                  && std::fabs((double)off.inv_inertia[0][1]) < 1e-5 && std::fabs((double)off.inv_inertia[1][2]) < 1e-5,
                  "hull mass: a box hull's inverse tensor is 3 / (hy^2 + hz^2) on the diagonal, zero off it");

            phys_body hb2 = static_hull(vec3(0, 0, 0), off), bx2 = dynamic_box(vec3(0, 0, 0), vec3(2, 1, 1));
            hb2.motion = DYNAMIC; hb2.mass = real(2); bx2.mass = real(2);
            const quat q = quat_from_euler_zyx_degrees(vec3(20, 50, -10));
            set_orientation(hb2, q); set_orientation(bx2, q);
            const vec3 L(real(0.3), -1, real(0.7));
            CHECK((delta_omega(hb2, L) - delta_omega(bx2, L)).length() < real(1e-5),
                  "delta_omega: a turned box hull spins exactly as the turned box does (R I^-1 R^T, mass 2)");
            CHECK(delta_omega(static_hull(vec3(0, 0, 0), off), L).near_zero(),
                  "delta_omega: an immovable hull is rotation-free (role outranks shape)");
        }

        // 10c. Face clip (B4 step 3): one generic clip serves box-box, box-hull and
        //      hull-hull, with the box as a 6-face view of it. The box-box manifold
        //      of test 8e is the oracle for the hull pairings.
        {
            std::vector<phys_body> bb, bh, hh;
            bb.push_back(static_box(vec3(0, 0, 0), vec3(1, 1, 1)));  bb.push_back(dynamic_box(vec3(0, real(1.9), 0), vec3(1, 1, 1)));
            bh.push_back(static_box(vec3(0, 0, 0), vec3(1, 1, 1)));  bh.push_back(dynamic_hull(vec3(0, real(1.9), 0), unit));
            hh.push_back(static_hull(vec3(0, 0, 0), unit));         hh.push_back(dynamic_hull(vec3(0, real(1.9), 0), unit));
            std::vector<contact> Cbb, Cbh, Chh;
            build_contacts(bb, Cbb); build_contacts(bh, Cbh); build_contacts(hh, Chh);
            CHECK(Cbb.size() == 4 && manifold_distance(Cbb, Cbh) < 1e-6 && manifold_distance(Cbb, Chh) < 1e-6,
                  "face clip: box-hull and hull-hull give the box-box 4-point manifold, same points and depths");

            // A hexagonal prism on the ground: six bottom vertices survive the clip.
            // Tilted half a degree about z, the deepest four cluster on the low side
            // (x extent 1.5 of 2); the spread rule keeps both ends of the footprint.
            const hull_shape hex = prism_hull(6, real(1), real(0.5));
            CHECK(wound_outward(hex) && hex.faces[0].loop.size() == 6, "hull: a hexagonal prism has six-vertex caps wound outward");
            std::vector<phys_body> pb;
            pb.push_back(static_box(vec3(0, -1, 0), vec3(10, 1, 10)));
            pb.push_back(dynamic_hull(vec3(0, real(0.47), 0), hex));
            set_orientation(pb[1], quat_from_euler_zyx_degrees(vec3(0, 0, real(0.5))));
            std::vector<contact> Cp;
            build_contacts(pb, Cp);
            real lo = 1, hi = -1, zlo = 1, zhi = -1;
            for (const contact& k : Cp) {
                lo = std::fmin(lo, k.p[0]); hi = std::fmax(hi, k.p[0]);
                zlo = std::fmin(zlo, k.p[2]); zhi = std::fmax(zhi, k.p[2]);
            }
            printf("  tilted hexagonal prism: %d contacts, x extent %.3f, z extent %.3f\n", (int)Cp.size(), (double)(hi - lo), (double)(zhi - zlo));
            CHECK(Cp.size() == 4 && hi - lo > real(1.99) && zhi - zlo > real(1.7),
                  "face clip: the spread rule keeps the whole footprint of a tilted hexagonal prism");
        }
    }

    // 11. HULL BUILDER (B4 step 4): build_hull turns a point cloud into the
    //     face-list hull the steps above consume. The hand-built hulls are the
    //     oracle: a cube's 24 quad corners (every corner three times) plus
    //     inside, on-face and on-edge points must give box_hull.
    {
        const vec3 half(1, 2, 3);
        const hull_shape cube = box_hull(half);
        std::vector<vec3> pts;
        for (const hull_shape::face& f : cube.faces) for (int i : f.loop) pts.push_back(cube.verts[i]);
        pts.push_back(vec3(0, 0, 0)); pts.push_back(vec3(1, 0, 0)); pts.push_back(vec3(real(0.5), 2, 1)); pts.push_back(vec3(1, 2, 0));
        hull_shape built;
        const vec3 com = build_hull(pts, built);
        bool quads = built.faces.size() == 6;
        for (const hull_shape::face& f : built.faces) quads = quads && f.loop.size() == 4;
        CHECK(built.verts.size() == 8 && quads && wound_outward(built) && com.length() < real(1e-6)
              && std::fabs((double)built.radius - std::sqrt(14.0)) < 1e-6,
              "build_hull: a cube cloud with repeated, inside, on-face and on-edge points gives 8 vertices and six quads");
        phys_body hb = static_hull(vec3(0, 0, 0), built), bx = static_box(vec3(0, 0, 0), half);
        bool same = true;
        for (int k = 0; k < 50; k++) {
            const vec3 d(std::sin(real(k)), std::cos(real(3 * k)), std::sin(real(7 * k + 1)));
            same = same && std::fabs((double)(dot(support(hb, d), d) - dot(support(bx, d), d))) < 1e-6;
        }
        CHECK(same, "build_hull: the built hull supports exactly like the analytic box");
        bool tensor = true;
        for (int r = 0; r < 3; r++) tensor = tensor && (built.inv_inertia[r] - cube.inv_inertia[r]).length() < real(1e-5);
        CHECK(tensor, "build_hull: same inverse inertia as the hand-built box hull");

        // the same cloud turned off-axis: the coplanarity test must still merge each face into one quad
        phys_body turn = static_box(vec3(0, 0, 0), half);
        set_orientation(turn, quat_from_euler_zyx_degrees(vec3(20, 50, -10)));
        for (vec3& v : pts) v = turn.axes[0] * v[0] + turn.axes[1] * v[1] + turn.axes[2] * v[2];
        hull_shape turned;
        const vec3 com_t = build_hull(pts, turned);
        quads = turned.faces.size() == 6;
        for (const hull_shape::face& f : turned.faces) quads = quads && f.loop.size() == 4;
        CHECK(turned.verts.size() == 8 && quads && wound_outward(turned) && com_t.length() < real(1e-6),
              "build_hull: a turned cube still merges into six quads");

        // an octahedron has no coplanar triangles: eight faces stay triangles
        std::vector<vec3> oct = { vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1) };
        hull_shape o;
        build_hull(oct, o);
        bool tris = o.faces.size() == 8 && o.verts.size() == 6;
        for (const hull_shape::face& f : o.faces) tris = tris && f.loop.size() == 3;
        CHECK(tris && wound_outward(o), "build_hull: an octahedron keeps eight triangles");

        // a hexagonal prism cloud gives prism_hull's faces and tensor
        const hull_shape hexp = prism_hull(6, real(1), real(0.5));
        hull_shape hb2;
        build_hull(hexp.verts, hb2);
        int caps = 0, sides = 0;
        for (const hull_shape::face& f : hb2.faces) { caps += f.loop.size() == 6; sides += f.loop.size() == 4; }
        tensor = true;
        for (int r = 0; r < 3; r++) tensor = tensor && (hb2.inv_inertia[r] - hexp.inv_inertia[r]).length() < real(1e-5);
        CHECK(caps == 2 && sides == 6 && hb2.verts.size() == 12 && tensor,
              "build_hull: a hexagonal prism cloud gives two hexagons, six quads and the hand-built tensor");
    }

    printf(fails ? "PHYSICS TESTS FAILED (%d)\n" : "ALL PHYSICS TESTS PASSED\n", fails);
    return fails ? 1 : 0;
}
