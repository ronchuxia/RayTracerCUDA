#ifndef VIEWER_SCENES_PRIMITIVES_H
#define VIEWER_SCENES_PRIMITIVES_H

#include "viewer/scene.h"
#include "scenes/scene_utils.h"
#include "viewer/physics_utils.h"

inline void build_primitives_scene(scene& sc) {
    // scene objects
    sc.init();

    material* ground  = new_lambertian(make_checker(0.32, color(.2, .3, .1), color(.9, .9, .9)), sc.allocs);
    material* diffuse = new_lambertian(color(0.4, 0.2, 0.1), sc.allocs);
    material* glass   = new_dielectric(1.5, sc.allocs);
    material* metal_m = new_metal(color(0.7, 0.6, 0.5), 0.0, sc.allocs);
    material* box_mat = new_lambertian(color(0.2, 0.4, 0.7), sc.allocs);
    material* tri_mat = new_lambertian(color(0.9, 0.75, 0.2), sc.allocs);

    sc.add(make_instance(make_sphere(point3(0,0,0), 1000, ground, sc.allocs),
                         vec3(0, -1000, 0), vec3(0,0,0), vec3(1,1,1)));     // id 0: floor

    int s1 = sc.add(make_instance(make_sphere(point3(0,0,0), 1.0, diffuse, sc.allocs),
                         vec3(-4, 1, 0), vec3(0,0,0), vec3(1,1,1)));        // id 1: diffuse sphere
    int s2 = sc.add(make_instance(make_sphere(point3(0,0,0), 1.0, glass, sc.allocs),
                         vec3(0, 1, 0), vec3(0,0,0), vec3(1,1,1)));         // id 2: glass sphere
    int s3 = sc.add(make_instance(make_sphere(point3(0,0,0), 1.0, metal_m, sc.allocs),
                         vec3(4, 1, 0), vec3(0,0,0), vec3(1,1,1)));         // id 3: metal sphere
    int bx = sc.add(make_instance(new_box(point3(-0.6,-0.6,-0.6), point3(0.6,0.6,0.6), box_mat, sc.allocs, sc.mesh_dtors),
                         vec3(-2, 0.6, -3), vec3(0, 35, 0), vec3(1,1,1)));  // id 4: box
    sc.add(make_instance(make_triangle(point3(-0.8,-0.6,0), point3(0.8,-0.6,0), point3(0,0.9,0),
                                       vec3(0,0,1), tri_mat, sc.allocs),
                         vec3(2, 1.3, -3), vec3(0,0,0), vec3(1,1,1)));      // id 5: triangle

    // physics bodies
    sc.bodies.push_back(make_sphere_body(sc, 0, STATIC));
    sc.bodies.push_back(make_sphere_body(sc, s1));
    sc.bodies.push_back(make_sphere_body(sc, s2));
    sc.bodies.push_back(make_sphere_body(sc, s3));
    sc.bodies.push_back(make_box_body(sc, bx));

    sc.build();
    // camera
    sc.lookfrom = point3(13, 2, 3);
    sc.lookat   = point3(0, 0, 0);
    sc.vfov     = real(20);
}

#endif // VIEWER_SCENES_PRIMITIVES_H
