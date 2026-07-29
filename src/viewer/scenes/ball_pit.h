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

    sc.add(make_sphere(point3(0, -1000, 0), 1000, ground, sc.allocs));   // id 0: floor

    // 4 walls: quads spanning z (or x) horizontally and y=[0,BOX_H] vertically.
    const real W = box_half, H = BOX_H;
    sc.add(make_quad(point3(-W, 0, -W), vec3(0, 0, 2*W), vec3(0, H, 0), wall, sc.allocs));  // x = -W
    sc.add(make_quad(point3( W, 0, -W), vec3(0, 0, 2*W), vec3(0, H, 0), wall, sc.allocs));  // x = +W
    sc.add(make_quad(point3(-W, 0, -W), vec3(2*W, 0, 0), vec3(0, H, 0), wall, sc.allocs));  // z = -W
    sc.add(make_quad(point3(-W, 0,  W), vec3(2*W, 0, 0), vec3(0, H, 0), wall, sc.allocs));  // z = +W

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
    // The container's collision shapes, deliberately NOT the visual ones:
    // the floor renders as a huge sphere and the walls as thin quads, but both
    // collide as half-spaces (a plane cannot be tunnelled through the way a thin
    // box can, and it costs one dot product).
    std::vector<phys_body> bodies;
    bodies.push_back(make_plane_body(vec3(0, 1, 0), 0,   PIT_MU));  // ground y = 0
    bodies.push_back(make_plane_body(vec3( 1, 0, 0), -W, PIT_MU));  // x = -W, facing inward
    bodies.push_back(make_plane_body(vec3(-1, 0, 0), -W, PIT_MU));  // x = +W
    bodies.push_back(make_plane_body(vec3(0, 0,  1), -W, PIT_MU));  // z = -W
    bodies.push_back(make_plane_body(vec3(0, 0, -1), -W, PIT_MU));  // z = +W
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
