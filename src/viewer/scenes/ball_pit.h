#ifndef VIEWER_SCENES_BALL_PIT_H
#define VIEWER_SCENES_BALL_PIT_H

#include <cmath>

#include "scene.h"
#include "scenes/scene_utils.h"
#include "viewer/physics_utils.h"
#include "viewer/scenes/viewer_scene.h"

// Ball pit (VIEWER_SCENE 1 = roomy, 2 = tight): a physics container (ground + 4
// walls, open top) with a central box obstacle and BALL_N diffuse spheres.
// The balls' initial location IS their drop-start: a loose spiral column above
// the box (off-axis angles + staggered heights), so on Play they fall one-by-one
// and cascade off the box. Each sphere is a UNIT prim scaled to BALL_R and
// transform-wrapped, which is what lets the panel edit it; Stop restores every
// body to this authored pose.
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
    constexpr real OBS_TOP     = real(0.9);          // obstacle height (its base sits at y = 0)
    constexpr real OBS_HALF_XZ = real(0.5);          // obstacle half-extent in x/z
    constexpr real OBS_HALF_Y  = OBS_TOP * real(0.5);// obstacle half-extent in y
    constexpr real DROP_H   = real(3.0);   // spawn height of the lowest ball

    // Surface feel, authored rather than left at the defaults. FRICTIONLESS for
    // now, which reproduces how the pit behaved before per-contact friction
    // existed (the old height-gated damp never fired here — no ball in this
    // scene ever reaches the floor).
    //
    // THE REASON A REAL COEFFICIENT OVER-DAMPS is the missing rolling: with no
    // angular dynamics (B3), a sphere cannot convert sliding into rolling, so
    // every bit of tangential friction is pure loss where a real ball would keep
    // going. The roomy pit is quiet in 4.9 s at the default 0.5, 8.3 s at 0.
    // The cost of 0 is that a ball slides forever once it is moving. Revisit at
    // B3: with rolling, a physical 0.3-0.5 should read correctly.
    constexpr real PIT_MU  = real(0.0);    // every surface in the container
    constexpr real BALL_E  = real(0.7);    // = the default, so every contact is 0.7

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

// ROOMY pit (1.5): the balls spread and settle; either solver handles it.
inline viewer_scene build_ball_pit_scene(scene& sc)       { return build_ball_pit(sc, real(1.5)); }
// TIGHT pit (1.3): crowded pile — only the sequential solver settles it.
inline viewer_scene build_ball_pit_tight_scene(scene& sc) { return build_ball_pit(sc, real(1.3)); }

#endif // VIEWER_SCENES_BALL_PIT_H
