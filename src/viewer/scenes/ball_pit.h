#ifndef VIEWER_SCENES_BALL_PIT_H
#define VIEWER_SCENES_BALL_PIT_H

#include <cmath>

#include "scene.h"
#include "scenes/scene_utils.h"
#include "viewer/physics_utils.h"
#include "viewer/scenes/viewer_scene.h"

// Ball pit (VIEWER_SCENE 1 = roomy, 2 = tight, 3 = rolling): a physics container
// (ground + 4 walls, open top) with a central box obstacle and BALL_N diffuse
// spheres. The balls' initial location IS their drop-start: a loose spiral column
// above the box (off-axis angles + staggered heights), so on Play they fall
// one-by-one and cascade off the box. Each sphere is a UNIT prim scaled to BALL_R
// and transform-wrapped, which is what lets the panel edit it; Stop restores
// every body to this authored pose.
//
// All three variants share build_ball_pit() and differ only in its two authored
// parameters — container half-width and surface friction:
//   ROOMY   (1.5, mu 0)    balls spread and settle; either solver handles it.
//   TIGHT   (1.3, mu 0)    crowded pile — only the sequential solver settles it.
//   ROLLING (1.5, mu 0.5)  the roomy pit with real friction, so the balls ROLL.
inline viewer_scene build_ball_pit(scene& sc, real box_half, real pit_mu) {
    sc.init();

    constexpr real BOX_H    = real(3.0);   // wall height
    constexpr int  BALL_N   = 8;           // spheres to drop
    constexpr real BALL_R   = real(0.5);
    constexpr real OBS_TOP     = real(0.9);          // obstacle height (its base sits at y = 0)
    constexpr real OBS_HALF_XZ = real(0.5);          // obstacle half-extent in x/z
    constexpr real OBS_HALF_Y  = OBS_TOP * real(0.5);// obstacle half-extent in y
    constexpr real DROP_H   = real(3.0);   // spawn height of the lowest ball

    // `pit_mu` is every surface in the container, authored rather than left at
    // the default, and it is what separates the ROLLING variant from the other
    // two. The frictionless variants are not a preference — they reproduce how
    // the pit behaved before per-contact friction existed, which is worth keeping
    // one of as a reference.
    //
    // FRICTION USED TO OVER-DAMP THIS SCENE, because a sphere with no angular
    // dynamics cannot convert sliding into rolling, so every bit of tangential
    // friction was pure loss. B3a fixed that, measured on the roomy pit as time
    // until the pile stops moving:
    //          mu 0     mu 0.3    mu 0.5
    //   before  5.8 s    3.8 s     4.5 s     friction made it die SOONER
    //   after   5.9 s   10.8 s     5.3 s     it no longer costs the pit anything
    // Rolling and spinning resistance (phys_body's defaults) are what eventually
    // stop a ball; at 0 it would turn forever. Single runs of a chaotic pile, so
    // read the direction rather than the second decimal.
    constexpr real BALL_E  = real(0.7);    // = the default, so every contact is 0.7
    const real PIT_MU = pit_mu;

    material* ground = new_lambertian(
        make_checker(0.6, color(.2, .3, .1), color(.9, .9, .9)), sc.allocs);
    material* wall   = new_lambertian(color(0.55, 0.55, 0.6), sc.allocs);

    // EVERY object here is transform-wrapped, container included: a body can only
    // be linked to an object that has a pose to follow, and the transform is
    // what supplies one. The prim is authored centred on its own origin and the
    // transform places it, so `translation` reads as the object's position — the
    // same convention the balls and the obstacle already use.
    sc.add(new_transform(make_sphere(point3(0, 0, 0), 1000, ground, sc.allocs),
                         vec3(0, -1000, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));  // id 0: floor

    // 4 walls: quads spanning z (or x) horizontally and y=[0,BOX_H] vertically,
    // authored about their own centre and translated into place.
    const real W = box_half, H = BOX_H;
    const vec3 span_z(0, 0, 2*W), span_x(2*W, 0, 0), up(0, H, 0);
    const point3 corner_z(0, -H/2, -W), corner_x(-W, -H/2, 0);   // centred prims
    sc.add(new_transform(make_quad(corner_z, span_z, up, wall, sc.allocs),
                         vec3(-W, H/2, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));   // id 1: x = -W
    sc.add(new_transform(make_quad(corner_z, span_z, up, wall, sc.allocs),
                         vec3( W, H/2, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));   // id 2: x = +W
    sc.add(new_transform(make_quad(corner_x, span_x, up, wall, sc.allocs),
                         vec3(0, H/2, -W), vec3(0,0,0), vec3(1,1,1), sc.allocs));   // id 3: z = -W
    sc.add(new_transform(make_quad(corner_x, span_x, up, wall, sc.allocs),
                         vec3(0, H/2,  W), vec3(0,0,0), vec3(1,1,1), sc.allocs));   // id 4: z = +W

    // Box obstacle in the centre. Transform-wrapped like every editable object,
    // which is what makes it a physics body (STATIC + an oriented box collider
    // derived from its pose) as well as selectable: set it KINEMATIC in the
    // Object panel and dragging it shoves the pile around, and rotating it turns
    // the collider with it. The prim is centred on the origin and the transform
    // lifts it, so it still spans y = [0, OBS_TOP].
    material* obs_mat = new_lambertian(color(0.7, 0.3, 0.2), sc.allocs);
    int obs_id = sc.add(new_transform(new_box(point3(-OBS_HALF_XZ, -OBS_HALF_Y, -OBS_HALF_XZ),
                                              point3( OBS_HALF_XZ,  OBS_HALF_Y,  OBS_HALF_XZ),
                                              obs_mat, sc.allocs, sc.list_dtors),
                                      vec3(0, OBS_HALF_Y, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));

    // ---- physics ----
    // EVERY collider is read from the object it belongs to, so nothing here can
    // disagree with what you see: the floor collides as the radius-1000 sphere it
    // draws as, and each wall as its own quad (2W x H x ~1e-4 thick, which is
    // bounded — a ball above the wall top now clears it instead of hitting a
    // barrier that is not drawn).
    std::vector<phys_body> bodies;
    bodies.push_back(make_sphere_body(sc, 0, STATIC, real(1), PIT_MU));   // floor
    for (int wall_id = 1; wall_id <= 4; wall_id++)
        bodies.push_back(make_box_body(sc, wall_id, STATIC, real(1), PIT_MU));
    // The obstacle: STATIC, but switch it to KINEMATIC in the Object panel and
    // dragging it shoves the whole pile around.
    bodies.push_back(make_box_body(sc, obs_id, STATIC, real(1), PIT_MU));

    // BALL_N spheres in a loose spiral column above the box: this IS the drop pose
    // (off-axis angles + staggered heights > a diameter apart), so they fall one
    // by one and cascade off the box. Stop re-spawns each body at exactly this pose.
    for (int i = 0; i < BALL_N; i++) {
        real ang = real(2.4) * real(i);                  // loose spiral so they interleave
        real rad = real(0.7) * BALL_R;                   // horizontal offset < diameter
        real x = rad * std::cos(ang), z = rad * std::sin(ang);
        real y = DROP_H + real(2.4) * BALL_R * real(i);  // staggered > diameter
        color col(0.5 + 0.4 * ((i * 37) % 7) / 6.0,      // spread hues deterministically
                  0.5 + 0.4 * ((i * 53) % 5) / 4.0,
                  0.5 + 0.4 * ((i * 29) % 3) / 2.0);
        material* m = new_lambertian(col, sc.allocs);
        int ball_id = sc.add(new_transform(make_sphere(point3(0,0,0), BALL_R, m, sc.allocs),
                                           vec3(x, y, z), vec3(0,0,0), vec3(1,1,1), sc.allocs));
        bodies.push_back(make_sphere_body(sc, ball_id, DYNAMIC, real(1),
                                          PIT_MU, BALL_E));   // DYNAMIC: the fallers
    }

    sc.build();
    return { point3(3.5, 8, 3.5), point3(0, 0.5, 0), real(40),   // look down into the container
             bodies };
}

// ROOMY pit: the balls spread and settle; either solver handles it.
inline viewer_scene build_ball_pit_scene(scene& sc) {
    return build_ball_pit(sc, real(1.5), real(0.0));
}
// TIGHT pit: crowded pile — only the sequential solver settles it.
inline viewer_scene build_ball_pit_tight_scene(scene& sc) {
    return build_ball_pit(sc, real(1.3), real(0.0));
}
// ROLLING pit: the roomy pit at phys_body's default friction, which is the one
// variant where the balls actually roll. Watch a ball that lands off-centre on
// the obstacle — it spins up as it slides off, then rolls across the floor and
// coasts to a stop under rolling resistance instead of stopping dead. The
// Object panel's spin read-out is the only other place that motion is visible,
// since a solid-colour sphere looks the same however it is turned.
inline viewer_scene build_ball_pit_rolling_scene(scene& sc) {
    return build_ball_pit(sc, real(1.5), real(0.5));
}

#endif // VIEWER_SCENES_BALL_PIT_H
