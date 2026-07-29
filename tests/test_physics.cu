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

        // (a) A ball sliding on the ground decelerates at about mu*g and stops.
        //     mu = sqrt(0.5*0.5) = 0.5 -> a = 4.9 m/s^2, so 2 m/s stops in ~0.41 s.
        {
            std::vector<phys_body> b;
            b.push_back(ground_plane(real(0.5), real(0)));
            b.push_back(ball(0, vec3(0, real(0.5), 0), vec3(2,0,0), real(0.5), real(1), real(0.5), real(0)));
            for (int s = 0; s < 240; s++) physics_step(b, p, h);   // 1 s
            CHECK(std::fabs((double)b[1].vel[0]) < 0.05,
                  "friction: a sliding ball decelerates at ~mu*g and stops");
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
        //     sliding on TOP of a box stops too (the old height test missed this).
        {
            std::vector<phys_body> b;
            b.push_back(static_box(vec3(0, 0.5, 0), vec3(4, 0.5, 4), real(0.5), real(0)));  // wide slab, top y=1
            b.push_back(ball(0, vec3(0, real(1.5), 0), vec3(2,0,0), real(0.5), real(1), real(0.5), real(0)));
            for (int s = 0; s < 240; s++) physics_step(b, p, h);
            CHECK(std::fabs((double)b[1].vel[0]) < 0.05,
                  "friction: a ball sliding on a box stops too, not just on the ground");
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

    printf(fails ? "PHYSICS TESTS FAILED (%d)\n" : "ALL PHYSICS TESTS PASSED\n", fails);
    return fails ? 1 : 0;
}
