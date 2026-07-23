#ifndef VIEWER_SCENES_BALL_PIT_H
#define VIEWER_SCENES_BALL_PIT_H

#include "scene.h"
#include "scenes/scene_utils.h"
#include "viewer/scenes/viewer_scene.h"

// Physics-scene container: floor (the ground plane) + 4 walls, open top. BOX_HALF
// is the half-width on x/z, BOX_H the wall height; the wall quads and the returned
// wall bounds are both derived from these, so geometry and collision stay in sync.
static constexpr float BOX_HALF = 1.3f;
static constexpr float BOX_H    = 3.0f;
static constexpr int   BALL_N   = 8;      // spheres in the physics scene
static constexpr float BALL_R   = 0.5f;

// BALL-PIT scene (VIEWER_SCENE 1): physics container: a container with BALL_N diffuse spheres to drop
// and collide. Each sphere is a UNIT prim scaled to BALL_R and transform-wrapped,
// so the physics-body scan (which selects transform-wrapped spheres) picks them
// up; the walls are plain static quads (not bodies, not editable).
inline viewer_scene build_ball_pit_scene(scene& sc) {
    sc.init();

    material* ground = new_lambertian(
        make_checker(0.6, color(.2, .3, .1), color(.9, .9, .9)), sc.allocs);
    material* wall   = new_lambertian(color(0.55, 0.55, 0.6), sc.allocs);

    sc.add(make_sphere(point3(0, -1000, 0), 1000, ground, sc.allocs));   // id 0: floor

    // 4 walls: quads spanning z (or x) horizontally and y=[0,BOX_H] vertically.
    const real W = real(BOX_HALF), H = real(BOX_H);
    sc.add(make_quad(point3(-W, 0, -W), vec3(0, 0, 2*W), vec3(0, H, 0), wall, sc.allocs));  // x = -W
    sc.add(make_quad(point3( W, 0, -W), vec3(0, 0, 2*W), vec3(0, H, 0), wall, sc.allocs));  // x = +W
    sc.add(make_quad(point3(-W, 0, -W), vec3(2*W, 0, 0), vec3(0, H, 0), wall, sc.allocs));  // z = -W
    sc.add(make_quad(point3(-W, 0,  W), vec3(2*W, 0, 0), vec3(0, H, 0), wall, sc.allocs));  // z = +W

    // BALL_N diffuse spheres, resting in a grid on the floor (Drop launches them).
    const int cols = 3;
    for (int i = 0; i < BALL_N; i++) {
        int r = i / cols, c = i % cols;
        real x = (real(c) - 1) * real(1.0);          // grid centred on the floor
        real z = (real(r) - 1) * real(1.0);
        color col(0.5 + 0.4 * ((i * 37) % 7) / 6.0,  // spread hues deterministically
                  0.5 + 0.4 * ((i * 53) % 5) / 4.0,
                  0.5 + 0.4 * ((i * 29) % 3) / 2.0);
        material* m = new_lambertian(col, sc.allocs);
        sc.add(new_transform(make_sphere(point3(0,0,0), BALL_R, m, sc.allocs),
                             vec3(x, BALL_R, z), vec3(0,0,0), vec3(1,1,1), sc.allocs));
    }

    sc.build();
    return { point3(4.5, 4.5, 4.5), point3(0, 0.5, 0), real(35),   // look down into the box
             vec3(real(-BOX_HALF), 0, real(-BOX_HALF)),
             vec3(real( BOX_HALF), 0, real( BOX_HALF)) };
}

#endif // VIEWER_SCENES_BALL_PIT_H
