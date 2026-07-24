#ifndef VIEWER_SCENES_BALL_PIT_H
#define VIEWER_SCENES_BALL_PIT_H

#include <cmath>

#include "scene.h"
#include "scenes/scene_utils.h"
#include "viewer/scenes/viewer_scene.h"

// Ball pit (VIEWER_SCENE 1 = roomy, 2 = tight): a physics container (ground + 4
// walls, open top) with a static central box obstacle and BALL_N diffuse spheres.
// The balls' initial location IS their drop-start: a loose spiral column above
// the box (off-axis angles + staggered heights), so on Drop they fall one-by-one
// and cascade off the box. Each sphere is a UNIT prim scaled to BALL_R and
// transform-wrapped so the physics-body scan picks it up; Drop resets each body
// to this authored pose.
//
// The two variants differ ONLY in the container half-width (box_half), so they
// share build_ball_pit(): the ROOMY pit (1.5) gives the balls room to spread and
// settle under either solver; the TIGHT pit (1.3) crowds them into a pile that
// only the sequential solver settles.
inline viewer_scene build_ball_pit(scene& sc, real box_half) {
    sc.init();

    constexpr real BOX_H    = real(3.0);   // wall height
    constexpr int  BALL_N   = 8;           // spheres to drop
    constexpr real BALL_R   = real(0.5);
    constexpr real OBS_HALF = real(0.5);   // central obstacle x/z half-width
    constexpr real OBS_TOP  = real(0.9);   // central obstacle height
    constexpr real DROP_H   = real(3.0);   // spawn height of the lowest ball

    material* ground = new_lambertian(
        make_checker(0.6, color(.2, .3, .1), color(.9, .9, .9)), sc.allocs);
    material* wall   = new_lambertian(color(0.55, 0.55, 0.6), sc.allocs);

    sc.add(make_sphere(point3(0, -1000, 0), 1000, ground, sc.allocs));   // id 0: floor

    // 4 walls: quads spanning z (or x) horizontally and y=[0,BOX_H] vertically.
    const real W = box_half, H = BOX_H;
    sc.add(make_quad(point3(-W, 0, -W), vec3(0, 0, 2*W), vec3(0, H, 0), wall, sc.allocs));  // x = -W
    sc.add(make_quad(point3( W, 0, -W), vec3(0, 0, 2*W), vec3(0, H, 0), wall, sc.allocs));  // x = +W
    sc.add(make_quad(point3(-W, 0, -W), vec3(2*W, 0, 0), vec3(0, H, 0), wall, sc.allocs));  // z = -W
    sc.add(make_quad(point3(-W, 0,  W), vec3(2*W, 0, 0), vec3(0, H, 0), wall, sc.allocs));  // z = +W

    // Static box obstacle in the centre (a plain box, not a physics body); its
    // extents are the collision AABB too.
    material* obs_mat = new_lambertian(color(0.7, 0.3, 0.2), sc.allocs);
    sc.add(new_box(point3(-OBS_HALF, 0, -OBS_HALF), point3(OBS_HALF, OBS_TOP, OBS_HALF),
                   obs_mat, sc.allocs, sc.list_dtors));

    // BALL_N spheres in a loose spiral column above the box: this IS the drop pose
    // (off-axis angles + staggered heights > a diameter apart), so they fall one
    // by one and cascade off the box. Drop re-spawns each body at exactly this pose.
    for (int i = 0; i < BALL_N; i++) {
        real ang = real(2.4) * real(i);                  // loose spiral so they interleave
        real rad = real(0.7) * BALL_R;                   // horizontal offset < diameter
        real x = rad * std::cos(ang), z = rad * std::sin(ang);
        real y = DROP_H + real(2.4) * BALL_R * real(i);  // staggered > diameter
        color col(0.5 + 0.4 * ((i * 37) % 7) / 6.0,      // spread hues deterministically
                  0.5 + 0.4 * ((i * 53) % 5) / 4.0,
                  0.5 + 0.4 * ((i * 29) % 3) / 2.0);
        material* m = new_lambertian(col, sc.allocs);
        sc.add(new_transform(make_sphere(point3(0,0,0), BALL_R, m, sc.allocs),
                             vec3(x, y, z), vec3(0,0,0), vec3(1,1,1), sc.allocs));
    }

    sc.build();
    return { point3(3.5, 8, 3.5), point3(0, 0.5, 0), real(40),     // look down into the container
             vec3(-W, 0, -W), vec3(W, 0, W),                       // container walls (x/z)
             true, vec3(-OBS_HALF, 0, -OBS_HALF), vec3(OBS_HALF, OBS_TOP, OBS_HALF) };  // box obstacle
}

// ROOMY pit (1.5): the balls spread and settle; either solver handles it.
inline viewer_scene build_ball_pit_scene(scene& sc)       { return build_ball_pit(sc, real(1.5)); }
// TIGHT pit (1.3): crowded pile — only the sequential solver settles it.
inline viewer_scene build_ball_pit_tight_scene(scene& sc) { return build_ball_pit(sc, real(1.3)); }

#endif // VIEWER_SCENES_BALL_PIT_H
