// Physics module tests — call the REAL src/physics.h (physics_step /
// resolve_sphere_pair), not a reimplementation. The whole point of extracting
// physics into a header was that tests exercise the shipping code directly.
//
//   nvcc tests/test_physics.cu -o build/test_physics -std=c++14 -arch=sm_86 -rdc=true -Isrc
//
// 1. resolve_sphere_pair: head-on collision has the analytic restitution result
//    and pushes the pair to just-touching.
// 2. physics_step: three clustered spheres dropped under gravity settle on the
//    ground, with no interpenetration (drives the exact viewer integrator).
#include <cstdio>
#include <cmath>
#include <vector>

#include "physics.h"

#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } \
                              else printf("ok: %s\n", msg); } while (0)

int main() {
    int fails = 0;

    // 1. Head-on: two unit spheres overlapping and approaching at +/-1 along x,
    //    e=0.7. Analytic: each bounces back at 0.7, centres end 2.0 apart.
    {
        phys_body a{0, vec3(-0.9, 0, 0), vec3( 1, 0, 0), real(1), vec3(), vec3()};
        phys_body b{1, vec3( 0.9, 0, 0), vec3(-1, 0, 0), real(1), vec3(), vec3()};
        resolve_sphere_pair(a, b, real(0.7));
        real sep = (a.pos - b.pos).length();
        CHECK(std::fabs((double)a.vel[0] + 0.7) < 1e-4 &&
              std::fabs((double)b.vel[0] - 0.7) < 1e-4 &&
              std::fabs((double)sep - 2.0)      < 1e-4,
              "head-on collision: restitution bounce + push to just-touching");
    }

    // 2. Three clustered spheres dropped under gravity settle on the ground with
    //    no interpenetration (exercises physics_step: integrate + ground + friction
    //    + pair collisions). Uses the viewer's own cluster/params.
    {
        const real h = real(1.0 / 240);
        phys_params p{ real(-9.8), real(0.7), real(0.99),
                       vec3(-1e30f, 0, -1e30f), vec3(1e30f, 0, 1e30f),  // no walls
                       false, vec3(), vec3() };                          // no box obstacle
        std::vector<phys_body> bodies;
        const int N = 3; const real drop = 3;
        for (int i = 0; i < N; i++) {
            real ox = real(0.7) * ((i & 1) ? real(1) : real(-1));
            real oz = real(0.4) * (real(i) - real(N - 1) * real(0.5));
            bodies.push_back({ i, vec3(ox, drop + real(1.8) * i, oz), vec3(0,0,0),
                               real(1), vec3(), vec3() });
        }
        real maxv = 0;
        for (int s = 0; s < 20000; s++) maxv = physics_step(bodies, p, h);

        bool on_ground = true, no_overlap = true;
        for (int i = 0; i < N; i++)
            if (bodies[i].pos[1] < bodies[i].radius - real(1e-2)) on_ground = false;
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++)
                if ((bodies[i].pos - bodies[j].pos).length()
                    < bodies[i].radius + bodies[j].radius - real(1e-2)) no_overlap = false;

        CHECK(maxv < real(0.05), "dropped spheres settle (max |v| -> 0)");
        CHECK(on_ground,         "all bodies rest on the ground plane");
        CHECK(no_overlap,        "no interpenetration at rest");
    }

    // 3. Sphere vs a static axis-aligned box [-1,1]^3, e=0.5.
    const vec3 bmin(-1, -1, -1), bmax(1, 1, 1);
    {
        // (a) FACE: sphere approaching the +x face head-on, overlapping. Closest
        //     point is on the face -> axis-aligned normal (+x); bounce back at 0.5.
        phys_body s{0, vec3(1.4, 0, 0), vec3(-1, 0, 0), real(0.5), vec3(), vec3()};
        resolve_sphere_box(s, bmin, bmax, real(0.5));
        CHECK(std::fabs((double)s.pos[0] - 1.5) < 1e-4 &&      // pushed to just touching (x = bmax+r)
              std::fabs((double)s.vel[0] - 0.5) < 1e-4 &&      // bounced +x at e*1
              std::fabs((double)s.vel[1]) < 1e-6,              // no tangential change
              "sphere-vs-box FACE: axis normal, restitution bounce");
    }
    {
        // (b) CORNER: sphere past the (+,+,+) corner along the diagonal. Closest
        //     point is the vertex -> diagonal normal; the whole velocity is normal.
        real off = real(1) + real(0.5) / std::sqrt(3.0) - real(0.05);  // overlapping the corner
        phys_body s{0, vec3(off, off, off), vec3(-1, -1, -1), real(0.5), vec3(), vec3()};
        resolve_sphere_box(s, bmin, bmax, real(0.5));
        vec3 d = s.pos - vec3(1, 1, 1);                        // vertex -> centre
        CHECK(std::fabs((double)d.length() - 0.5) < 1e-3,      // pushed to exactly radius from the corner
              "sphere-vs-box CORNER: closest point is the vertex, pushed to radius");
        CHECK(dot(s.vel, unit_vector(vec3(1,1,1))) > 0,        // now separating along the diagonal
              "sphere-vs-box CORNER: bounced along the corner diagonal");
    }
    {
        // (c) INSIDE: centre inside the box, nearest the +x face -> ejected out +x.
        phys_body s{0, vec3(0.8, 0, 0), vec3(0, 0, 0), real(0.5), vec3(), vec3()};
        resolve_sphere_box(s, bmin, bmax, real(0.5));
        CHECK(std::fabs((double)s.pos[0] - 1.5) < 1e-4,        // ejected to x = bmax + r
              "sphere-vs-box INSIDE: ejected along the nearest face");
    }

    printf(fails ? "PHYSICS TESTS FAILED (%d)\n" : "ALL PHYSICS TESTS PASSED\n", fails);
    return fails ? 1 : 0;
}
