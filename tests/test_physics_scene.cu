// Scene-level physics tests: do a viewer scene's AUTHORED bodies actually match
// the geometry it drew, and does the resulting world behave?
//
//   nvcc tests/test_physics_scene.cu -o build/test_physics_scene \
//        -std=c++14 -arch=sm_86 -rdc=true -Isrc
//
// tests/test_physics.cu covers src/physics.h over hand-built bodies — the solver
// maths in isolation. This file covers the layer above it: physics_utils.h's
// factories, the scenes that call them, and the derived colliders. Those are
// exactly what nothing else checks, because a scene builds its bodies from live
// hittables and so cannot be exercised without building the scene.
//
// It needs CUDA at run time (scenes live in unified memory) but no display and
// no SDL2/GLEW — it never opens a window, so it runs anywhere the offline
// renderer does.
#define RT_SKY 1
#include <cstdio>
#include <cmath>
#include <vector>

#include "camera.h"
#include "physics.h"
#include "scene.h"
#include "scenes/scene_utils.h"
#include "viewer/physics_utils.h"
#include "viewer/scenes/ball_pit.h"
#include "viewer/scenes/primitives.h"

#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } \
                              else printf("ok: %s\n", msg); } while (0)

static const real H        = real(1.0 / 240);   // the viewer's PHYS_DT
static const real BOX_HALF = real(1.3);         // tight pit: the crowded variant
static const int  SETTLE   = 4800;              // 20 s — long enough for the pile to sleep

// The viewer's index: scene id -> body index. Unlinked bodies never appear.
static std::vector<int> index_bodies(const std::vector<phys_body>& bodies, int n_objects) {
    std::vector<int> body_of_scene_id((size_t)n_objects, -1);
    for (int i = 0; i < (int)bodies.size(); i++)
        if (bodies[i].scene_id >= 0) body_of_scene_id[bodies[i].scene_id] = i;
    return body_of_scene_id;
}

int main() {
    int fails = 0;
    const phys_params p{ real(-9.8) };

    // ================= ball pit =================
    scene sc;
    viewer_scene vs = build_ball_pit(sc, BOX_HALF, real(0));   // the frictionless tight pit
    std::vector<phys_body> bodies = vs.bodies;
    std::vector<int> body_of_scene_id = index_bodies(bodies, (int)sc.objects.size());

    int boxes = 0, spheres = 0, unlinked = 0, box_i = -1, box_id = -1;
    for (int i = 0; i < (int)bodies.size(); i++) {
        if (bodies[i].scene_id < 0) unlinked++;
        if (bodies[i].shape == COLLIDER_SPHERE) spheres++;
        if (bodies[i].shape == COLLIDER_BOX)    boxes++;
        if (bodies[i].shape == COLLIDER_BOX && bodies[i].scene_id == 5)
            { box_i = i; box_id = bodies[i].scene_id; }   // the obstacle
    }
    // 4 walls + 1 obstacle as boxes; floor + 8 balls as spheres. EVERY body is
    // linked, and every collider is the shape its object actually draws as.
    CHECK(boxes == 5 && spheres == 9 && unlinked == 0,
          "ball pit authors 5 box + 9 sphere bodies, all linked to their objects");
    CHECK(box_id >= 0 && body_of_scene_id[box_id] == box_i,
          "the obstacle is LINKED, so it can be selected and dragged");

    // 1. The collider must sit where the scene DREW the box (x/z +/-0.5, y 0..0.9),
    //    within aabb::pad()'s ~5e-5-per-side widening of the box's planar faces.
    //    This is the check that catches a factory drifting from its geometry.
    {
        const phys_body& bx = bodies[box_i];
        CHECK(std::fabs((double)bx.pos[1]  - 0.45) < 1e-3 &&
              std::fabs((double)bx.half[0] - 0.50) < 1e-3 &&
              std::fabs((double)bx.half[1] - 0.45) < 1e-3 &&
              std::fabs((double)bx.half[2] - 0.50) < 1e-3,
              "box collider matches the drawn extents");
    }

    // 2. The pit runs: balls fall, cascade off the obstacle, and settle inside it
    //    without sinking, escaping, entering the box, or overlapping each other.
    {
        std::vector<phys_body> b = bodies;
        real maxv = 0;
        for (int s = 0; s < SETTLE; s++) maxv = physics_step(b, p, H);
        bool above_ground = true, inside_walls = true, clear_of_box = true, no_overlap = true;
        for (int i = 0; i < (int)b.size(); i++) {
            if (b[i].motion != DYNAMIC) continue;          // the balls
            vec3 gn; real gpen;                       // the ground is a sphere now, so
            if (contact_between(b[i], b[0], gn, gpen) && gpen > real(0.02))
                above_ground = false;                 // ask the narrow phase, not y > r
            if (std::fabs((double)b[i].pos[0]) > (double)BOX_HALF + 0.02 ||
                std::fabs((double)b[i].pos[2]) > (double)BOX_HALF + 0.02) inside_walls = false;
            vec3 n; real pen;
            if (contact_between(b[i], b[box_i], n, pen) && pen > real(0.02)) clear_of_box = false;
            for (int j = i + 1; j < (int)b.size(); j++) {
                if (b[j].motion != DYNAMIC) continue;
                if ((b[i].pos - b[j].pos).length() < b[i].radius + b[j].radius - real(0.02))
                    no_overlap = false;
            }
        }
        printf("  tight pit after 20 s: max |v| = %.4f\n", (double)maxv);
        CHECK(maxv < real(0.1), "tight ball pit settles");
        CHECK(above_ground,     "no ball sinks through the ground");
        CHECK(inside_walls,     "no ball escapes the walls");
        CHECK(clear_of_box,     "no ball penetrates the obstacle");
        CHECK(no_overlap,       "no ball-ball interpenetration at rest");
    }

    // 3. A4: the obstacle set KINEMATIC and dragged across the settled pile must
    //    hold exactly the pose the driver gives it, and shove balls aside.
    {
        std::vector<phys_body> b = bodies;
        for (int s = 0; s < SETTLE; s++) physics_step(b, p, H);
        std::vector<vec3> before; for (const phys_body& q : b) before.push_back(q.pos);

        phys_body& box = b[box_i];
        box.motion = KINEMATIC;
        vec3 from = box.pos, to = box.pos + vec3(real(0.9), 0, 0);
        for (int k = 0; k < 60; k++) {                    // 60 small drag steps
            box.pos = from + (to - from) * (real(k) / real(59));
            box.vel = vec3(0, 0, 0);                      // position-only, as a panel drag is
            for (int s = 0; s < 4; s++) physics_step(b, p, H);
        }
        int pushed = 0; bool clear = true;
        for (int i = 0; i < (int)b.size(); i++) {
            if (b[i].motion != DYNAMIC) continue;          // the balls
            if ((b[i].pos - before[i]).length() > real(0.15)) pushed++;
            vec3 n; real pen;
            if (contact_between(b[i], box, n, pen) && pen > real(0.05)) clear = false;
        }
        CHECK((box.pos - to).length() < real(1e-6),
              "a dragged KINEMATIC body holds exactly the pose it was given");
        CHECK(pushed >= 2, "dragging the obstacle shoves balls out of its path");
        CHECK(clear,       "no ball is left inside the dragged obstacle");
    }

    // 4. Oriented colliders through the REAL edit path: rewrite the transform the
    //    way the Object panel does, then re-derive as the viewer does. The
    //    collider must TURN, not grow — a 45-degree turn makes the enclosing
    //    world AABB 0.707 half-wide while the box is still 0.5.
    {
        transform* tr = static_cast<transform*>(sc.get(box_id)->object);
        new(tr) transform(tr->child, tr->translation, vec3(0, 45, 0), tr->scale);
        sc.refit();
        std::vector<phys_body> b = bodies;
        box_collider_of(tr, b[box_i].pos, b[box_i].half, b[box_i].axes);

        const real rt = real(std::sqrt(0.5));
        CHECK(std::fabs((double)b[box_i].half[0] - 0.5) < 1e-3 &&
              std::fabs((double)b[box_i].half[2] - 0.5) < 1e-3,
              "a turned box collider keeps its extents (0.5, not the AABB's 0.707)");
        CHECK(std::fabs((double)b[box_i].axes[0][0] - (double)rt) < 1e-4 &&
              std::fabs((double)b[box_i].axes[0][2] + (double)rt) < 1e-4 &&
              std::fabs((double)b[box_i].axes[2][0] - (double)rt) < 1e-4 &&
              std::fabs((double)b[box_i].axes[2][2] - (double)rt) < 1e-4,
              "collider axes follow the transform's turn");

        // A point inside the enclosing AABB but well clear of the turned box.
        phys_body probe = b.back();
        probe.pos = vec3(real(0.62), real(0.45), real(0.62));
        probe.radius = real(0.2);
        std::vector<phys_body> two; two.push_back(probe); two.push_back(b[box_i]);
        std::vector<contact> C; build_contacts(two, C);
        CHECK(C.size() == 0, "no contact in the AABB corner the turned box does not occupy");

        real maxv = 0;
        for (int s = 0; s < SETTLE; s++) maxv = physics_step(b, p, H);
        bool clear = true;
        for (int i = 0; i < (int)b.size(); i++) {
            if (b[i].motion != DYNAMIC) continue;          // the balls
            vec3 n; real pen;
            if (contact_between(b[i], b[box_i], n, pen) && pen > real(0.02)) clear = false;
        }
        printf("  turned obstacle after 20 s: max |v| = %.4f\n", (double)maxv);
        CHECK(maxv < real(0.1), "the pit settles with the obstacle turned 45 degrees");
        CHECK(clear,            "no ball penetrates the turned obstacle");
    }
    sc.release();

    // ================= rolling ball pit =================
    // The roomy pit at real friction (VIEWER_SCENE 3). Frictionless pits cannot
    // show this: with mu = 0 nothing ever torques a ball, so the whole angular
    // path is dead code in scenes 1 and 2. This is the scene-level check that
    // B3a is actually reached from authored geometry.
    {
        scene sc3;
        viewer_scene vs3 = build_ball_pit_rolling_scene(sc3);
        std::vector<phys_body> b = vs3.bodies;

        real maxv = 0; real peak_spin = 0;
        for (int s = 0; s < SETTLE; s++) {
            maxv = physics_step(b, p, H);
            for (const phys_body& q : b)
                if (inv_mass(q) > real(0) && q.omega.length() > peak_spin)
                    peak_spin = q.omega.length();
        }
        int spun = 0;
        for (const phys_body& q : b)
            if (inv_mass(q) > real(0) && q.omega.length() > real(0.5)) spun++;

        printf("  rolling pit after 20 s: max |v| = %.4f, peak spin %.2f rad/s\n",
               (double)maxv, (double)peak_spin);
        CHECK(peak_spin > real(1),
              "balls in the rolling pit actually spin up — friction reaches the angular path");
        CHECK(maxv < real(0.1),
              "and the rolling pit still settles: rolling resistance stops them");
        CHECK(spun == 0,
              "nothing is left spinning at rest");

        // Every ball still ends up inside the container and above the floor —
        // the same invariants the frictionless pit is held to, now with rolling
        // in play, which is what would carry a ball out through a wall.
        bool contained = true, above_ground = true;
        for (int i = 0; i < (int)b.size(); i++) {
            if (b[i].motion != DYNAMIC) continue;
            vec3 gn; real gpen;
            if (contact_between(b[i], b[0], gn, gpen) && gpen > real(0.02)) above_ground = false;
            if (std::fabs((double)b[i].pos[0]) > 1.5 + 0.02 ||
                std::fabs((double)b[i].pos[2]) > 1.5 + 0.02) contained = false;
        }
        CHECK(above_ground, "no rolling ball sinks through the ground");
        CHECK(contained,    "no rolling ball escapes the walls");
        sc3.release();
    }

    // ================= primitives showcase =================
    // The scene decides what is simulated, so a decorative object simply has no
    // body. This is the case the old derive-by-rule code got wrong: it gave the
    // triangle a box collider nobody asked for.
    {
        scene sc2;
        viewer_scene vs2 = build_primitives_scene(sc2);
        std::vector<int> idx = index_bodies(vs2.bodies, (int)sc2.objects.size());
        int linked = 0, unlinked = 0;
        for (const phys_body& b : vs2.bodies) (b.scene_id >= 0 ? linked : unlinked)++;
        int without_body = 0;
        for (int id = 0; id < (int)sc2.objects.size(); id++) if (idx[id] < 0) without_body++;

        CHECK(vs2.bodies.size() == 5 && linked == 5 && unlinked == 0,
              "showcase authors 5 bodies, all linked to their objects");
        CHECK((int)sc2.objects.size() == 6 && without_body == 1,
              "6 objects, 1 with no body: the decorative triangle");
        sc2.release();
    }

    printf(fails ? "SCENE PHYSICS TESTS FAILED (%d)\n" : "ALL SCENE PHYSICS TESTS PASSED\n", fails);
    return fails ? 1 : 0;
}
