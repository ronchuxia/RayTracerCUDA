// Physics module tests — call the REAL src/physics.h (physics_step /
// solve_sequential / sphere_box_contact), not a reimplementation. The whole
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

#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } \
                              else printf("ok: %s\n", msg); } while (0)

// --- world builders, mirroring the viewer's body scan ---
// The ground is a large flat static BOX with its top face at y = 0. There is no
// plane collider any more: every collider is read from the object it belongs to,
// and no scene object is an infinite plane. A box top gives the same contact
// normal and penetration a plane did, so the assertions below are unchanged.
static phys_body ground_plane(real friction = real(0.5), real restitution = real(0.7)) {
    phys_body g{ -1, vec3(0, -100, 0), vec3(0,0,0), real(0), vec3(), vec3() };
    g.motion = STATIC; g.shape = COLLIDER_BOX; g.half = vec3(1000, 100, 1000);
    g.friction = friction; g.restitution = restitution;
    return g;
}
static phys_body static_box(const vec3& centre, const vec3& half,
                            real friction = real(0.5), real restitution = real(0.7)) {
    phys_body b{ -1, centre, vec3(0,0,0), real(0), vec3(), vec3() };
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
// The same box turned `deg` degrees about y. physics.h reads the box's axes, not
// an angle, so the test writes them directly — it has no dependency on
// transforms.h. These are the columns of Ry(deg), matching what
// physics_utils.h's box_collider_of() hands over from transform::apply_R.
static phys_body rotated_box_y(const vec3& centre, const vec3& half, real deg) {
    phys_body b = static_box(centre, half);
    double t = (double)deg * 3.14159265358979323846 / 180.0;
    real c = real(std::cos(t)), s = real(std::sin(t));
    b.axes[0] = vec3(c, 0, -s);
    b.axes[1] = vec3(0, 1,  0);
    b.axes[2] = vec3(s, 0,  c);
    return b;
}
static phys_body ball(int scene_id, const vec3& pos, const vec3& vel, real r,
                      real m = real(1), real friction = real(0.5),
                      real restitution = real(0.7)) {
    phys_body b{ scene_id, pos, vel, r, vec3(), vec3() };
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
        bool hit = sphere_box_contact(vec3(1.4, 0, 0), real(0.5), bmin, bmax, n, pen);
        CHECK(hit && std::fabs((double)n[0] - 1.0) < 1e-6 &&
              std::fabs((double)pen - 0.1) < 1e-5,
              "sphere-vs-box FACE: axis normal, penetration = r - gap");
    }
    {
        // (b) CORNER: past the (+,+,+) vertex along the diagonal -> diagonal normal.
        real off = real(1) + real(0.5) / std::sqrt(3.0) - real(0.05);
        vec3 n; real pen;
        bool hit = sphere_box_contact(vec3(off, off, off), real(0.5), bmin, bmax, n, pen);
        vec3 diag = unit_vector(vec3(1,1,1));
        CHECK(hit && dot(n, diag) > real(0.999),
              "sphere-vs-box CORNER: closest point is the vertex, normal is the diagonal");
    }
    {
        // (c) INSIDE: centre inside the box, nearest the +x face -> ejected out +x.
        vec3 n; real pen;
        bool hit = sphere_box_contact(vec3(0.8, 0, 0), real(0.5), bmin, bmax, n, pen);
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

    // 8. CONVEX-CONVEX (B2): support functions + GJK/EPA, in src/gjk.h. These
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

        // 8d. The two paths must agree. contact_between takes the analytic
        //     sphere-box test; gjk_epa_contact takes the general one. Same
        //     configuration, same answer — that is what makes the fast path a
        //     cost decision rather than a second implementation to keep in sync.
        //
        //     The normal tolerance is 0.01, not 1e-5 like the depth: EPA's
        //     tolerance is on DISTANCE, which bounds the ANGLE only to
        //     sqrt(2*TOL/r) on a curved surface. Measured worst case here is
        //     0.0063 rad (0.36 degrees), identical in float and double. See the
        //     table at EPA_TOL in src/gjk.h.
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
                bool h1 = contact_between(S, B, n1, p1);      // analytic fast path
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
                if (contact_between(B, S, n3, p3)) {
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

        // 8e. The narrow phase emits box-box now. Before B2 contact_between
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
            bool oriented = C.size() == 1 &&
                            dot(C[0].n, b[C[0].a].pos - b[C[0].b].pos) > real(0) &&
                            std::fabs((double)std::fabs((double)C[0].n[1]) - 1) < 1e-3;
            CHECK(oriented,
                  "build_contacts emits a box-box contact, with n from b toward a");
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

    // 9. ROTATION (B3a): spheres carry angular velocity, friction spins them up,
    //    and a ball ROLLS instead of being scrubbed to a halt. Boxes stay
    //    rotation-free until B3c.
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
            vec3 got = inv_inertia_apply(s, vec3(1, 0, 0));
            CHECK(std::fabs((double)got[0] - 2.5 * 0.5 / 0.25) < 1e-6,
                  "inv_inertia: a solid sphere's is 2.5 * inv_mass / r^2");

            phys_body st = s; st.motion = STATIC;
            CHECK(inv_inertia_apply(st, vec3(1,0,0)).near_zero(),
                  "inv_inertia: an immovable body cannot be spun, whatever its mass");

            CHECK(inv_inertia_apply(static_box(vec3(0,0,0), vec3(1,1,1)), vec3(1,0,0)).near_zero() &&
                  inv_inertia_apply(dynamic_box(vec3(0,0,0), vec3(1,1,1)), vec3(1,0,0)).near_zero(),
                  "inv_inertia: a box is rotation-free until B3c, dynamic or not");
        }

        // 9b. A sphere's lever arm is PARALLEL to the contact normal, so the
        //     normal impulse exerts no torque and the normal effective mass is
        //     untouched. This is why bouncing is bit-for-bit what it was, and why
        //     only friction can spin a ball.
        {
            phys_body s = ball(0, vec3(0, real(0.5), 0), vec3(), real(0.5));
            vec3 up(0, 1, 0);
            vec3 r = contact_lever(s, up, true);
            phys_body none = ground_plane();
            vec3 rb = contact_lever(none, up, false);
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

        // 9e. A box takes no spin, however it is hit — the property that keeps
        //     B3c a deliberate change rather than a discovery.
        {
            std::vector<phys_body> b;
            b.push_back(ground_plane(real(0.5), real(0)));
            b.push_back(dynamic_box(vec3(0, real(0.5), 0), vec3(real(0.5), real(0.5), real(0.5)),
                                    real(0.5), real(0)));
            b[1].vel = vec3(2, 0, 0);
            for (int s = 0; s < 480; s++) physics_step(b, p, h);
            CHECK(b[1].omega.near_zero(),
                  "a box never acquires spin, so it cannot absorb impulse it could not express");
            CHECK(std::fabs((double)b[1].vel[0]) < 0.05,
                  "and so a sliding box still stops dead, as it did before B3a");
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

    printf(fails ? "PHYSICS TESTS FAILED (%d)\n" : "ALL PHYSICS TESTS PASSED\n", fails);
    return fails ? 1 : 0;
}
