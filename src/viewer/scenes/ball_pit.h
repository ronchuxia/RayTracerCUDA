#ifndef VIEWER_SCENES_BALL_PIT_H
#define VIEWER_SCENES_BALL_PIT_H

#include "scene.h"
#include "scenes/scene_utils.h"
#include "viewer/scenes/viewer_scene.h"

// Physics-scene container: floor (the ground plane) + 4 walls, open top. BOX_HALF
// is the half-width on x/z, BOX_H the wall height; the wall quads and the returned
// wall bounds are both derived from these, so geometry and collision stay in sync.
static constexpr float BOX_HALF = 1.5f;   // room to settle around the central obstacle
static constexpr float BOX_H    = 3.0f;
static constexpr int   BALL_N   = 8;      // spheres in the physics scene
static constexpr float BALL_R   = 0.5f;
// A static box obstacle in the centre: balls drop onto it and cascade off the
// edges (sphere-vs-box collision). Its extents are the collision AABB too.
static constexpr float OBS_HALF = 0.5f;   // x/z half-width
static constexpr float OBS_TOP  = 0.9f;   // height

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

    // Static box obstacle in the centre (a plain box, not a physics body).
    material* obs_mat = new_lambertian(color(0.7, 0.3, 0.2), sc.allocs);
    const real OH = real(OBS_HALF), OT = real(OBS_TOP);
    sc.add(new_box(point3(-OH, 0, -OH), point3(OH, OT, OH), obs_mat, sc.allocs, sc.list_dtors));

    // BALL_N diffuse spheres, resting in the 3x3 grid AROUND the obstacle. The
    // centre cell sits inside the box, so skip it (BALL_N == 8 == the 8 ring
    // cells). Drop launches them from above.
    int placed = 0;
    for (int dz = -1; dz <= 1 && placed < BALL_N; dz++)
        for (int dx = -1; dx <= 1 && placed < BALL_N; dx++) {
            if (dx == 0 && dz == 0) continue;             // centre cell is inside the obstacle
            color col(0.5 + 0.4 * ((placed * 37) % 7) / 6.0,  // spread hues deterministically
                      0.5 + 0.4 * ((placed * 53) % 5) / 4.0,
                      0.5 + 0.4 * ((placed * 29) % 3) / 2.0);
            material* m = new_lambertian(col, sc.allocs);
            sc.add(new_transform(make_sphere(point3(0,0,0), BALL_R, m, sc.allocs),
                                 vec3(real(dx), BALL_R, real(dz)), vec3(0,0,0), vec3(1,1,1), sc.allocs));
            placed++;
        }

    sc.build();
    return { point3(4.5, 4.5, 4.5), point3(0, 0.5, 0), real(35),   // look down into the box
             vec3(real(-BOX_HALF), 0, real(-BOX_HALF)),
             vec3(real( BOX_HALF), 0, real( BOX_HALF)),
             true,                                              // collide balls against the obstacle
             vec3(real(-OBS_HALF), 0,            real(-OBS_HALF)),
             vec3(real( OBS_HALF), real(OBS_TOP), real( OBS_HALF)) };
}

#endif // VIEWER_SCENES_BALL_PIT_H
