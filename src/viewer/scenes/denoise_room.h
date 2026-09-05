#ifndef VIEWER_SCENES_DENOISE_ROOM_H
#define VIEWER_SCENES_DENOISE_ROOM_H

#include "viewer/scene.h"
#include "scenes/scene_utils.h"
#include "loaders/image_loader.h"
#include "viewer/physics_utils.h"

#ifndef RT_EARTH_IMG
#define RT_EARTH_IMG "assets/earthmap.jpg"
#endif

// Denoiser evaluation room (VIEWER_SCENE 5). A closed Cornell-style room —
// the camera never sees the sky, so the only light is one small ceiling quad
// and everything else is bounce-lit: mottled low-spp noise on the walls, a
// caustic under the glass sphere, glossy noise on the fuzzy metal, a sharp
// reflection on the mirror ball. Materials
// exercise each guide: the earth texture (albedo), the box and triangle
// (normal edges). Objects start in the air so Play drops them — denoising
// under motion and accumulation resets. Small coordinates on purpose:
// docs/issues/quad-pad-below-float-ulp.md.
inline void build_denoise_room_scene(scene& sc) {
    sc.init();

    const real HW = real(3);     // room half-width (x, z)
    const real HT = real(5);     // room height (y from 0 to HT)

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
    // walls: quads authored around the origin, placed by the transform
    auto wall = [&](const point3& Q, const vec3& u, const vec3& v, material* m, const vec3& at) {
        return sc.add(new_transform(make_quad(Q, u, v, m, sc.allocs), at, no_rot, id3, sc.allocs));
    };
    int floor_id = wall(point3(-HW, 0, -HW), span_x, span_z, white, vec3(0, 0, 0));      // id 0: floor y=0
    wall(point3(-HW, 0, -HW), span_x, span_z, white, vec3(0, HT, 0));                      // id 1: ceiling y=HT
    int back_id  = wall(point3(-HW, 0, 0), span_x, span_y, white, vec3(0, 0,  HW));       // id 2: back z=+HW
    int left_id  = wall(point3(0, 0, -HW), span_z, span_y, red,   vec3(-HW, 0, 0));       // id 3: left x=-HW
    int right_id = wall(point3(0, 0, -HW), span_z, span_y, green, vec3( HW, 0, 0));       // id 4: right x=+HW
    int front_id = wall(point3(-HW, 0, 0), span_x, span_y, white, vec3(0, 0, -HW));       // id 5: front z=-HW (behind camera)
    wall(point3(-0.6, 0, -0.6), vec3(1.2, 0, 0), vec3(0, 0, 1.2), light, vec3(0, HT - real(0.01), 0.5));  // id 6: light

    // objects, in the air so Play drops them. Image-left is +x for this camera.
    int earth_id = sc.add(new_transform(make_sphere(point3(0, 0, 0), 0.7, earth, sc.allocs),
                                        vec3(-1.7, 1.8, 1.2), vec3(0, 0, 0), id3, sc.allocs));         // id 7: right
    int glass_id = sc.add(new_transform(make_sphere(point3(0, 0, 0), 0.7, glass, sc.allocs),
                                        vec3(0.0, 2.2, 1.0), no_rot, id3, sc.allocs));                 // id 8: centre
    int metal_id = sc.add(new_transform(make_sphere(point3(0, 0, 0), 0.7, metal, sc.allocs),
                                        vec3(1.9, 2.6, 1.6), no_rot, id3, sc.allocs));                 // id 9: upper left
    int mirror_id = sc.add(new_transform(make_sphere(point3(0, 0, 0), 0.7, mirror, sc.allocs),
                                        vec3(-2.0, 3.4, 1.8), no_rot, id3, sc.allocs));                // id 10: upper right
    int box_id   = sc.add(new_transform(new_box(point3(-0.5, -0.9, -0.5), point3(0.5, 0.9, 0.5), blue, sc.allocs, sc.list_dtors),
                                        vec3(1.4, 1.2, 2.2), vec3(15, 20, 0), id3, sc.allocs));         // id 11: lower left
    sc.add(new_transform(make_triangle(point3(-0.8, -0.6, 0), point3(0.8, -0.6, 0), point3(0, 0.9, 0),
                                       vec3(0, 0, -1), gold, sc.allocs),
                         vec3(-0.6, 3.4, 2.8), no_rot, id3, sc.allocs));                               // id 12: on the back wall, no body

    // physics: room faces are STATIC box bodies (padded quad bboxes, as in the ball pit)
    for (int id : {floor_id, back_id, left_id, right_id, front_id})
        sc.bodies.push_back(make_box_body(sc, id, STATIC));
    sc.bodies.push_back(make_sphere_body(sc, earth_id));
    sc.bodies.push_back(make_sphere_body(sc, glass_id));
    sc.bodies.push_back(make_sphere_body(sc, metal_id));
    sc.bodies.push_back(make_sphere_body(sc, mirror_id));
    sc.bodies.push_back(make_box_body(sc, box_id, DYNAMIC));

    sc.build();
    // camera: inside the room, just in front of the front wall, facing the back wall head-on
    sc.lookfrom = point3(0, 2.4, -2.7);
    sc.lookat   = point3(0, 1.9, 1.0);
    sc.vfov     = real(65);
}

#endif // VIEWER_SCENES_DENOISE_ROOM_H
