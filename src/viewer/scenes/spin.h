#ifndef VIEWER_SCENES_SPIN_H
#define VIEWER_SCENES_SPIN_H

#include "viewer/scene.h"
#include "scenes/scene_utils.h"
#include "loaders/image_loader.h"
#include "viewer/physics_utils.h"

#ifndef RT_EARTH_IMG
#define RT_EARTH_IMG "assets/earthmap.jpg"
#endif

inline void build_spin_scene(scene& sc) {
    sc.init();

    constexpr real R  = real(0.5);
    constexpr real MU = real(0.5);

    material* ground = new_lambertian(make_checker(0.6, color(.2, .3, .1), color(.9, .9, .9)), sc.allocs);
    material* earth  = new_lambertian(load_image_texture(RT_EARTH_IMG, sc.allocs), sc.allocs);

    sc.add(new_transform(make_sphere(point3(0, 0, 0), 1000, ground, sc.allocs),
                         vec3(0, -1000, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));  // id 0: floor
    sc.bodies.push_back(make_sphere_body(sc, 0, STATIC, real(1), MU));

    // ids 1-3
    for (int i = 0; i < 3; i++) {
        int id = sc.add(new_transform(make_sphere(point3(0, 0, 0), R, earth, sc.allocs),
                                      vec3(-3, R, real(-1.5) + real(1.5) * real(i)),
                                      vec3(0,0,0), vec3(1,1,1), sc.allocs));
        phys_body b = make_sphere_body(sc, id, DYNAMIC, real(1), MU);
        b.vel = vec3(real(3) + real(i), 0, 0);
        sc.bodies.push_back(b);
    }

    // id 4: spin about z
    {
        int id = sc.add(new_transform(make_sphere(point3(0, 0, 0), R, earth, sc.allocs),
                                      vec3(-3, R, 3), vec3(0,0,0), vec3(1,1,1), sc.allocs));
        phys_body b = make_sphere_body(sc, id, DYNAMIC, real(1), MU);
        b.omega = vec3(0, 0, -10);
        sc.bodies.push_back(b);
    }

    // id 5: spin about the normal
    {
        int id = sc.add(new_transform(make_sphere(point3(0, 0, 0), R, earth, sc.allocs),
                                      vec3(0, R, -3), vec3(0,0,0), vec3(1,1,1), sc.allocs));
        phys_body b = make_sphere_body(sc, id, DYNAMIC, real(1), MU);
        b.omega = vec3(0, 10, 0);
        sc.bodies.push_back(b);
    }

    sc.build();
    // camera
    sc.lookfrom = point3(0, 4, 9);
    sc.lookat   = point3(0, 0.5, 0);
    sc.vfov     = real(35);
}

#endif // VIEWER_SCENES_SPIN_H
