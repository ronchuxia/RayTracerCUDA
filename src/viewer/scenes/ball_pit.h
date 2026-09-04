#ifndef VIEWER_SCENES_BALL_PIT_H
#define VIEWER_SCENES_BALL_PIT_H

#include <cmath>

#include "viewer/scene.h"
#include "scenes/scene_utils.h"
#include "viewer/physics_utils.h"

inline void build_ball_pit(scene& sc, real box_half, real pit_mu) {
    sc.init();

    constexpr real BOX_H    = real(3.0);    // wall height
    constexpr int  BALL_N   = 8;
    constexpr real BALL_R   = real(0.5);
    constexpr real OBS_TOP     = real(0.9);          // obstacle height
    constexpr real OBS_HALF_XZ = real(0.5);          // obstacle half-extent in x/z
    constexpr real OBS_HALF_Y  = OBS_TOP * real(0.5);// obstacle half-extent in y
    constexpr real DROP_H   = real(3.0);    // height of the lowest ball

    constexpr real BALL_E  = real(0.7);
    const real PIT_MU = pit_mu;

    material* ground = new_lambertian(make_checker(0.6, color(.2, .3, .1), color(.9, .9, .9)), sc.allocs);
    material* wall   = new_lambertian(color(0.55, 0.55, 0.6), sc.allocs);

    sc.add(new_transform(make_sphere(point3(0, 0, 0), 1000, ground, sc.allocs),
                         vec3(0, -1000, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));  // id 0: floor

    // 4 walls
    const real W = box_half, H = BOX_H;
    const vec3 span_z(0, 0, 2*W), span_x(2*W, 0, 0), up(0, H, 0);
    const point3 corner_z(0, -H/2, -W), corner_x(-W, -H/2, 0);
    sc.add(new_transform(make_quad(corner_z, span_z, up, wall, sc.allocs),
                         vec3(-W, H/2, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));   // id 1: x = -W
    sc.add(new_transform(make_quad(corner_z, span_z, up, wall, sc.allocs),
                         vec3( W, H/2, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));   // id 2: x = +W
    sc.add(new_transform(make_quad(corner_x, span_x, up, wall, sc.allocs),
                         vec3(0, H/2, -W), vec3(0,0,0), vec3(1,1,1), sc.allocs));   // id 3: z = -W
    sc.add(new_transform(make_quad(corner_x, span_x, up, wall, sc.allocs),
                         vec3(0, H/2,  W), vec3(0,0,0), vec3(1,1,1), sc.allocs));   // id 4: z = +W

    // box obstacle in the center
    material* obs_mat = new_lambertian(color(0.7, 0.3, 0.2), sc.allocs);
    int obs_id = sc.add(new_transform(new_box(point3(-OBS_HALF_XZ, -OBS_HALF_Y, -OBS_HALF_XZ),
                                              point3( OBS_HALF_XZ,  OBS_HALF_Y,  OBS_HALF_XZ),
                                              obs_mat, sc.allocs, sc.list_dtors),
                                      vec3(0, OBS_HALF_Y, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));

    // physics
    sc.bodies.push_back(make_sphere_body(sc, 0, STATIC, real(1), PIT_MU));   // floor
    for (int wall_id = 1; wall_id <= 4; wall_id++)
        sc.bodies.push_back(make_box_body(sc, wall_id, STATIC, real(1), PIT_MU));  // 4 walls
    sc.bodies.push_back(make_box_body(sc, obs_id, STATIC, real(1), PIT_MU));   // box obstacle

    // BALL_N spheres in a spiral above the box
    for (int i = 0; i < BALL_N; i++) {
        real ang = real(2.4) * real(i);
        real rad = real(0.7) * BALL_R;
        real x = rad * std::cos(ang), z = rad * std::sin(ang);
        real y = DROP_H + real(2.4) * BALL_R * real(i);
        color col(0.5 + 0.4 * ((i * 37) % 7) / 6.0,
                  0.5 + 0.4 * ((i * 53) % 5) / 4.0,
                  0.5 + 0.4 * ((i * 29) % 3) / 2.0);
        material* m = new_lambertian(col, sc.allocs);
        int ball_id = sc.add(new_transform(make_sphere(point3(0,0,0), BALL_R, m, sc.allocs),
                                           vec3(x, y, z), vec3(0,0,0), vec3(1,1,1), sc.allocs));
        sc.bodies.push_back(make_sphere_body(sc, ball_id, DYNAMIC, real(1),
                                          PIT_MU, BALL_E));
    }

    sc.build();
    // camera
    sc.lookfrom = point3(3.5, 8, 3.5);
    sc.lookat   = point3(0, 0.5, 0);
    sc.vfov     = real(40);
}

// ROOMY pit
inline void build_ball_pit_scene(scene& sc) {
    build_ball_pit(sc, real(1.5), real(0.0));
}
// TIGHT pit
inline void build_ball_pit_tight_scene(scene& sc) {
    build_ball_pit(sc, real(1.3), real(0.0));
}
// ROLLING pit
inline void build_ball_pit_rolling_scene(scene& sc) {
    build_ball_pit(sc, real(1.5), real(0.5));
}

#endif // VIEWER_SCENES_BALL_PIT_H
