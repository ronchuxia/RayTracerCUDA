#ifndef VIEWER_SCENES_HULL_PIT_H
#define VIEWER_SCENES_HULL_PIT_H

#include <cmath>
#include <vector>

#include "viewer/scene.h"
#include "scenes/scene_utils.h"
#include "viewer/physics_utils.h"

// build a convex mesh object from a vector of points
inline mesh* new_convex(const std::vector<vec3>& pts, material* mat,
                        std::vector<void*>& allocs, std::vector<mesh*>& mesh_dtors) {
    // build a convex hull
    hull_shape h;
    build_hull(pts, h);

    // add triangles on the convex hull's faces to the mesh
    mesh* faces = new_mesh(allocs, mesh_dtors);
    for (const hull_shape::face& f : h.faces)
        for (size_t k = 1; k + 1 < f.loop.size(); k++)
            mesh_add_triangle(faces, h.verts[f.loop[0]], h.verts[f.loop[k]], h.verts[f.loop[k + 1]], f.normal, mat, allocs);
    faces->build();
    return faces;
}

inline std::vector<vec3> prism_points(int sides, real r, real h) {
    std::vector<vec3> p;
    for (int i = 0; i < sides; i++) {
        const real t = real(2 * M_PI) * i / sides;
        p.push_back(vec3(r * std::cos(t), -h, r * std::sin(t)));
        p.push_back(vec3(r * std::cos(t),  h, r * std::sin(t)));
    }
    return p;
}

inline std::vector<vec3> frustum_points(real r_bottom, real r_top, real h) {
    std::vector<vec3> p;
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sz = -1; sz <= 1; sz += 2) {
            p.push_back(vec3(sx * r_bottom, -h, sz * r_bottom));
            p.push_back(vec3(sx * r_top,     h, sz * r_top));
        }
    return p;
}

inline std::vector<vec3> octahedron_points(real r) {
    std::vector<vec3> p;
    for (int a = 0; a < 3; a++)
        for (int s = -1; s <= 1; s += 2) p.push_back(vec3(a == 0 ? s * r : 0, a == 1 ? s * r : 0, a == 2 ? s * r : 0));
    return p;
}

inline std::vector<vec3> icosahedron_points(real r) {
    const real phi = real((1 + std::sqrt(5.0)) / 2), s = r / std::sqrt(real(1) + phi * phi);
    std::vector<vec3> p;
    for (int a = 0; a < 3; a++)
        for (int i = -1; i <= 1; i += 2)
            for (int j = -1; j <= 1; j += 2) {
                vec3 v(0, 0, 0);
                v[(a + 1) % 3] = i * s; v[(a + 2) % 3] = j * phi * s;
                p.push_back(v);
            }
    return p;
}

inline mesh* new_spiky_ball(real r_core, real r_tip, material* mat,
                            std::vector<void*>& allocs, std::vector<mesh*>& mesh_dtors) {
    mesh* faces = new_mesh(allocs, mesh_dtors);
    
    hull_shape core;
    build_hull(icosahedron_points(r_core), core);
    for (const hull_shape::face& f : core.faces) {
        const vec3 tip = f.normal * r_tip;
        for (int k = 0; k < 3; k++) {
            const vec3& a = core.verts[f.loop[k]];
            const vec3& b = core.verts[f.loop[(k + 1) % 3]];
            mesh_add_triangle(faces, a, b, tip, unit_vector(cross(b - a, tip - a)), mat, allocs);
        }
    }
    faces->build();
    return faces;
}

inline std::vector<vec3> dodecahedron_points(real r) {
    const real phi = real((1 + std::sqrt(5.0)) / 2), s = r / std::sqrt(real(3));
    std::vector<vec3> p;
    for (int i = -1; i <= 1; i += 2)
        for (int j = -1; j <= 1; j += 2)
            for (int k = -1; k <= 1; k += 2) p.push_back(vec3(i * s, j * s, k * s));
    for (int a = 0; a < 3; a++)
        for (int i = -1; i <= 1; i += 2)
            for (int j = -1; j <= 1; j += 2) {
                vec3 v(0, 0, 0);
                v[(a + 1) % 3] = i * s / phi; v[(a + 2) % 3] = j * phi * s;
                p.push_back(v);
            }
    return p;
}

inline void build_hull_pit_scene(scene& sc) {
    sc.init();
    constexpr real W = real(1.5), H = real(1.5), MU = real(0.5), DROP_H = real(3.0), E = real(0.3);

    // quad floor (id 0) and four walls (ids 1..4)
    material* ground = new_lambertian(color(0.7, 0.7, 0.7), sc.allocs);
    material* wall   = new_lambertian(color(0.55, 0.55, 0.6), sc.allocs);
    sc.add(make_instance(make_quad(point3(-1000, 0, -1000), vec3(2000, 0, 0), vec3(0, 0, 2000), ground, sc.allocs),
                         vec3(0, 0, 0), vec3(0,0,0), vec3(1,1,1)));   // id 0: floor
    const vec3 span_z(0, 0, 2*W), span_x(2*W, 0, 0), up(0, H, 0);
    const point3 corner_z(0, -H/2, -W), corner_x(-W, -H/2, 0);
    sc.add(make_instance(make_quad(corner_z, span_z, up, wall, sc.allocs),
                         vec3(-W, H/2, 0), vec3(0,0,0), vec3(1,1,1)));   // id 1: x = -W
    sc.add(make_instance(make_quad(corner_z, span_z, up, wall, sc.allocs),
                         vec3( W, H/2, 0), vec3(0,0,0), vec3(1,1,1)));   // id 2: x = +W
    sc.add(make_instance(make_quad(corner_x, span_x, up, wall, sc.allocs),
                         vec3(0, H/2, -W), vec3(0,0,0), vec3(1,1,1)));   // id 3: z = -W
    sc.add(make_instance(make_quad(corner_x, span_x, up, wall, sc.allocs),
                         vec3(0, H/2,  W), vec3(0,0,0), vec3(1,1,1)));   // id 4: z = +W
    for (int id = 0; id <= 4; id++) sc.bodies.push_back(make_box_body(sc, id, STATIC, real(1), MU));

    struct solid { std::vector<vec3> pts; color col; vec3 rot; };
    const solid solids[] = {
        { prism_points(6, real(0.5), real(0.35)),                 color(0.8, 0.3, 0.2), vec3(0, 0, 0) },    // hexagonal prism
        { prism_points(12, real(0.35), real(0.6)),                color(0.2, 0.5, 0.8), vec3(0, 0, 90) },   // 12-gon prism on its side
        { frustum_points(real(0.5), real(0.25), real(0.3)),       color(0.8, 0.7, 0.2), vec3(0, 30, 0) },   // square frustum
        { prism_points(3, real(0.55), real(0.3)),                 color(0.3, 0.7, 0.3), vec3(0, 60, 0) },   // triangular wedge
        { octahedron_points(real(0.5)),                           color(0.7, 0.3, 0.7), vec3(0, 0, 0) },    // octahedron
        { icosahedron_points(real(0.45)),                         color(0.9, 0.9, 0.9), vec3(0, 0, 0) },    // icosahedron (metal)
        { dodecahedron_points(real(0.45)),                        color(0.9, 0.5, 0.2), vec3(0, 0, 0) },    // dodecahedron
    };
    const int n_solids = (int)(sizeof(solids) / sizeof(solids[0]));

    for (int i = 0; i < n_solids + 3; i++) {
        const real ang = real(2.4) * real(i), rad = real(0.4);
        const real x = rad * std::cos(ang), z = rad * std::sin(ang);
        const real y = DROP_H + real(1.4) * real(i);
        if (i < n_solids) {
            const solid& s = solids[i];
            material* m = i == 5 ? new_metal(s.col, 0.05, sc.allocs) : new_lambertian(s.col, sc.allocs);
            int id = sc.add(make_instance(new_convex(s.pts, m, sc.allocs, sc.mesh_dtors),
                                          vec3(x, y, z), s.rot, vec3(1, 1, 1)));
            sc.bodies.push_back(make_hull_body(sc, id, DYNAMIC, real(1), MU, E));
        } else if (i == n_solids) {
            material* m = new_lambertian(color(0.9, 0.8, 0.3), sc.allocs);
            int id = sc.add(make_instance(new_spiky_ball(real(0.3), real(0.55), m, sc.allocs, sc.mesh_dtors),
                                          vec3(x, y, z), vec3(0, 0, 0), vec3(1, 1, 1)));
            sc.bodies.push_back(make_hull_body(sc, id, DYNAMIC, real(1), MU, E));
        } else {
            material* m = new_lambertian(color(0.5, 0.6, 0.9), sc.allocs);
            int id = sc.add(make_instance(make_sphere(point3(0, 0, 0), real(0.4), m, sc.allocs),
                                          vec3(x, y, z), vec3(0, 0, 0), vec3(1, 1, 1)));
            sc.bodies.push_back(make_sphere_body(sc, id, DYNAMIC, real(1), MU, E));
        }
    }

    sc.build();
    // camera
    sc.lookfrom = point3(3.5, 8, 3.5);
    sc.lookat   = point3(0, 0.5, 0);
    sc.vfov     = real(40);
}

#endif // VIEWER_SCENES_HULL_PIT_H
