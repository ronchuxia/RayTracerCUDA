#ifndef VIEWER_SCENES_PRIMITIVES_H
#define VIEWER_SCENES_PRIMITIVES_H

#include "scene.h"
#include "scenes/scene_utils.h"
#include "viewer/physics_utils.h"
#include "viewer/scenes/viewer_scene.h"

// PRIMITIVES scene (VIEWER_SCENE 0): one of each primitive type,: checker ground + editable objects, sky-lit.
// Every editable object (spheres, a box, a triangle — one of each prim type) is
// registered as a transform(prim), so B5 can drive full T/R/S on any of them
// uniformly. The underlying prims are UNIT and centred at the origin; the
// transform supplies position and size. The ground stays a plain sphere — it's
// the floor, not something you manipulate. Ids are registration order. The three
// spheres rest ON the ground (centre y = radius = 1), so the viewer's shared Drop
// control is a no-op here — this scene is for viewing/editing, not simulation.
inline viewer_scene build_primitives_scene(scene& sc) {
    sc.init();

    material* ground  = new_lambertian(
        make_checker(0.32, color(.2, .3, .1), color(.9, .9, .9)), sc.allocs);
    material* diffuse = new_lambertian(color(0.4, 0.2, 0.1), sc.allocs);
    material* glass   = new_dielectric(1.5, sc.allocs);
    material* metal_m = new_metal(color(0.7, 0.6, 0.5), 0.0, sc.allocs);
    material* box_mat = new_lambertian(color(0.2, 0.4, 0.7), sc.allocs);
    material* tri_mat = new_lambertian(color(0.9, 0.75, 0.2), sc.allocs);

    // Transform-wrapped like everything else: a body can only be linked to an
    // object that has a pose to follow, and the transform is what supplies one.
    sc.add(new_transform(make_sphere(point3(0,0,0), 1000, ground, sc.allocs),
                         vec3(0, -1000, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));  // id 0: floor

    // Editable objects: unit prim at origin + transform(T, R°, S).
    int s1 = sc.add(new_transform(make_sphere(point3(0,0,0), 1.0, diffuse, sc.allocs),
                         vec3(-4, 1, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));   // id 1: diffuse sphere
    int s2 = sc.add(new_transform(make_sphere(point3(0,0,0), 1.0, glass, sc.allocs),
                         vec3(0, 1, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));    // id 2: glass sphere
    int s3 = sc.add(new_transform(make_sphere(point3(0,0,0), 1.0, metal_m, sc.allocs),
                         vec3(4, 1, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));    // id 3: metal sphere
    int bx = sc.add(new_transform(new_box(point3(-0.6,-0.6,-0.6), point3(0.6,0.6,0.6), box_mat, sc.allocs, sc.list_dtors),
                         vec3(-2, 0.6, -3), vec3(0, 35, 0), vec3(1,1,1), sc.allocs));  // id 4: box (composite)
    sc.add(new_transform(make_triangle(point3(-0.8,-0.6,0), point3(0.8,-0.6,0), point3(0,0.9,0),
                                       vec3(0,0,1), tri_mat, sc.allocs),
                         vec3(2, 1.3, -3), vec3(0,0,0), vec3(1,1,1), sc.allocs));      // id 5: triangle

    // ---- physics ----
    // A ground plane and the three spheres, which start resting on it, so Play is
    // a no-op until you drag one up and let it fall. No walls: this scene is for
    // viewing and editing, not for containing anything.
    //
    // The TRIANGLE is deliberately NOT a body. It is decoration, and the only
    // collider we could give it today is its enclosing box — a shape that does
    // not match what you see. Authoring bodies rather than deriving them from
    // geometry is exactly what makes "no collider" expressible.
    std::vector<phys_body> bodies;
    bodies.push_back(make_sphere_body(sc, 0, STATIC));   // floor: collides as the sphere it draws as
    bodies.push_back(make_sphere_body(sc, s1));
    bodies.push_back(make_sphere_body(sc, s2));
    bodies.push_back(make_sphere_body(sc, s3));
    bodies.push_back(make_box_body(sc, bx));          // static obstacle to roll a sphere into

    sc.build();
    return { point3(13, 2, 3), point3(0, 0, 0), real(20), bodies };   // camera + physics
}

#endif // VIEWER_SCENES_PRIMITIVES_H
