#ifndef VIEWER_SCENES_ROOM_H
#define VIEWER_SCENES_ROOM_H

#include "viewer/scene.h"
#include "scenes/scene_utils.h"
#include "loaders/image_loader.h"
#include "viewer/physics_utils.h"

#ifndef RT_EARTH_IMG
#define RT_EARTH_IMG "assets/earthmap.jpg"
#endif

inline void build_denoise_room_scene(scene& sc) {
    sc.init();

    const real HW = real(3);     // room half-width
    const real HT = real(5);     // room height

    material* white = new_lambertian(color(0.73, 0.73, 0.73), sc.allocs);
    material* red   = new_lambertian(color(0.65, 0.05, 0.05), sc.allocs);
    material* green = new_lambertian(color(0.12, 0.45, 0.15), sc.allocs);
    material* light = new_diffuse_light(color(15, 15, 15), sc.allocs);
    material* earth = new_lambertian(load_image_texture(RT_EARTH_IMG, sc.allocs), sc.allocs);
    material* glass = new_dielectric(1.5, sc.allocs);
    material* metal = new_metal(color(0.8, 0.7, 0.6), 0.3, sc.allocs);
    material* mirror = new_metal(color(0.9, 0.9, 0.9), 0.0, sc.allocs);
    material* blue  = new_lambertian(color(0.2, 0.4, 0.7), sc.allocs);
    material* gold  = new_lambertian(color(0.9, 0.75, 0.2), sc.allocs);

    const vec3 id3(1, 1, 1), no_rot(0, 0, 0);
    const vec3 span_x(2*HW, 0, 0), span_y(0, HT, 0), span_z(0, 0, 2*HW);

    // walls
    auto wall = [&](const point3& Q, const vec3& u, const vec3& v, material* m, const vec3& at) {
        return sc.add(make_instance(make_quad(Q, u, v, m, sc.allocs), at, no_rot, id3));
    };
    int floor_id = wall(point3(-HW, 0, -HW), span_x, span_z, white, vec3(0, 0, 0));       // id 0: floor y=0
    wall(point3(-HW, 0, -HW), span_x, span_z, white, vec3(0, HT, 0));                     // id 1: ceiling y=HT
    int back_id  = wall(point3(-HW, 0, 0), span_x, span_y, white, vec3(0, 0,  HW));       // id 2: back z=+HW
    int left_id  = wall(point3(0, 0, -HW), span_z, span_y, red,   vec3(-HW, 0, 0));       // id 3: left x=-HW
    int right_id = wall(point3(0, 0, -HW), span_z, span_y, green, vec3( HW, 0, 0));       // id 4: right x=+HW
    int front_id = wall(point3(-HW, 0, 0), span_x, span_y, white, vec3(0, 0, -HW));       // id 5: front z=-HW (behind camera)
    wall(point3(-0.6, 0, -0.6), vec3(1.2, 0, 0), vec3(0, 0, 1.2), light, vec3(0, HT - real(0.01), 0.5));  // id 6: light

    // objects
    int earth_id = sc.add(make_instance(make_sphere(point3(0, 0, 0), 0.7, earth, sc.allocs),
                                        vec3(-1.7, 1.8, 1.2), vec3(0, 0, 0), id3));         // id 7: right
    int glass_id = sc.add(make_instance(make_sphere(point3(0, 0, 0), 0.7, glass, sc.allocs),
                                        vec3(0.0, 2.2, 1.0), no_rot, id3));                 // id 8: centre
    int metal_id = sc.add(make_instance(make_sphere(point3(0, 0, 0), 0.7, metal, sc.allocs),
                                        vec3(1.9, 2.6, 1.6), no_rot, id3));                 // id 9: upper left
    int mirror_id = sc.add(make_instance(make_sphere(point3(0, 0, 0), 0.7, mirror, sc.allocs),
                                        vec3(-2.0, 3.4, 1.8), no_rot, id3));                // id 10: upper right
    int box_id   = sc.add(make_instance(new_box(point3(-0.5, -0.9, -0.5), point3(0.5, 0.9, 0.5), blue, sc.allocs, sc.mesh_dtors),
                                        vec3(1.4, 1.2, 2.2), vec3(15, 20, 0), id3));        // id 11: lower left
    sc.add(make_instance(make_triangle(point3(-0.8, -0.6, 0), point3(0.8, -0.6, 0), point3(0, 0.9, 0),
                         vec3(0, 0, -1), gold, sc.allocs),
                         vec3(-0.6, 3.4, 2.8), no_rot, id3));                               // id 12: on the back wall

    // physics
    for (int id : {floor_id, back_id, left_id, right_id, front_id})
        sc.bodies.push_back(make_box_body(sc, id, STATIC));
    sc.bodies.push_back(make_sphere_body(sc, earth_id));
    sc.bodies.push_back(make_sphere_body(sc, glass_id));
    sc.bodies.push_back(make_sphere_body(sc, metal_id));
    sc.bodies.push_back(make_sphere_body(sc, mirror_id));
    sc.bodies.push_back(make_box_body(sc, box_id, DYNAMIC));

    sc.build();
    // camera
    sc.lookfrom = point3(0, 2.4, -2.7);
    sc.lookat   = point3(0, 1.9, 1.0);
    sc.vfov     = real(65);
}

#endif // VIEWER_SCENES_ROOM_H
