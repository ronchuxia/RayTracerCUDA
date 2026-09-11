#ifndef SCENE_UTILS_H
#define SCENE_UTILS_H

#include <vector>

#include "cuda_helper.h"
#include "hittable.h"
#include "material.h"
#include "vec3.h"

// Host-only scene-construction helpers

// --- materials -------------------------------------------------------------

inline material* new_lambertian(const color& albedo, std::vector<void*>& allocs) {
    material* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(material)));
    m->type = LAMBERTIAN;
    m->lam = lambertian(albedo);
    allocs.push_back(m);
    return m;
}

inline material* new_lambertian(const texture& tex, std::vector<void*>& allocs) {
    material* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(material)));
    m->type = LAMBERTIAN;
    m->lam = lambertian(tex);
    allocs.push_back(m);
    return m;
}

inline material* new_metal(const color& albedo, double fuzz, std::vector<void*>& allocs) {
    material* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(material)));
    m->type = METAL;
    m->met = metal(albedo, fuzz);
    allocs.push_back(m);
    return m;
}

inline material* new_dielectric(double ir, std::vector<void*>& allocs) {
    material* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(material)));
    m->type = DIELECTRIC;
    m->die = dielectric(ir);
    allocs.push_back(m);
    return m;
}

inline material* new_tinted_glass(double ir, const color& absorption, std::vector<void*>& allocs) {
    material* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(material)));
    m->type = DIELECTRIC;
    m->die = dielectric(ir, absorption);
    allocs.push_back(m);
    return m;
}

inline material* new_diffuse_light(const color& emit, std::vector<void*>& allocs) {
    material* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(material)));
    m->type = DIFFUSE_LIGHT;
    m->light = diffuse_light(emit);
    allocs.push_back(m);
    return m;
}

inline material* new_isotropic(const texture& albedo, std::vector<void*>& allocs) {
    material* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(material)));
    m->type = ISOTROPIC;
    m->iso = isotropic(albedo);
    allocs.push_back(m);
    return m;
}

// --- primitives --------------------------------------------------------------

inline primitive* make_sphere(const point3& center, double radius,
                              material* mat, std::vector<void*>& allocs) {
    sphere* s;
    checkCudaErrors(cudaMallocManaged((void**)&s, sizeof(sphere)));
    new(s) sphere(center, radius, mat);

    primitive* p;
    checkCudaErrors(cudaMallocManaged((void**)&p, sizeof(primitive)));
    p->type = SPHERE;
    p->object = s;

    allocs.push_back(s);
    allocs.push_back(p);
    return p;
}

inline primitive* make_quad(const point3& Q, const vec3& u, const vec3& v,
                            material* mat, std::vector<void*>& allocs) {
    quad* q;
    checkCudaErrors(cudaMallocManaged((void**)&q, sizeof(quad)));
    new(q) quad(Q, u, v, mat);

    primitive* p;
    checkCudaErrors(cudaMallocManaged((void**)&p, sizeof(primitive)));
    p->type = QUAD;
    p->object = q;

    allocs.push_back(q);
    allocs.push_back(p);
    return p;
}

inline primitive* make_triangle(const point3& v0, const point3& v1,
                                const point3& v2, const vec3& n, material* mat,
                                std::vector<void*>& allocs) {
    triangle* t;
    checkCudaErrors(cudaMallocManaged((void**)&t, sizeof(triangle)));
    new(t) triangle(v0, v1, v2, n, mat);

    primitive* p;
    checkCudaErrors(cudaMallocManaged((void**)&p, sizeof(primitive)));
    p->type = TRIANGLE;
    p->object = t;

    allocs.push_back(t);
    allocs.push_back(p);
    return p;
}

// --- instances ---------------------------------------------------------------

inline instance make_instance(const primitive* p, const vec3& translation,
                              const vec3& rotation, const vec3& scale) {
    return instance(p, translation, rotation, scale);
}

inline instance make_instance(const mesh* m, const vec3& translation,
                              const vec3& rotation, const vec3& scale) {
    return instance(m, translation, rotation, scale);
}

inline instance make_medium(const mesh* boundary, const vec3& translation,
                            const vec3& rotation, const vec3& scale, double density,
                            const texture& albedo, std::vector<void*>& allocs) {
    instance in = make_instance(boundary, translation, rotation, scale);
    in.set_medium(density, new_isotropic(albedo, allocs));
    return in;
}

// add_* helpers place a bare shape in the world with the identity transform

inline void add_sphere(world* w, const point3& center, double radius,
                       material* mat, std::vector<void*>& allocs) {
    w->add(make_instance(make_sphere(center, radius, mat, allocs), vec3(0,0,0), vec3(0,0,0), vec3(1,1,1)));
}

inline void add_quad(world* w, const point3& Q, const vec3& u, const vec3& v,
                     material* mat, std::vector<void*>& allocs) {
    w->add(make_instance(make_quad(Q, u, v, mat, allocs), vec3(0,0,0), vec3(0,0,0), vec3(1,1,1)));
}

inline void add_triangle(world* w, const point3& v0, const point3& v1,
                         const point3& v2, const vec3& n, material* mat,
                         std::vector<void*>& allocs) {
    w->add(make_instance(make_triangle(v0, v1, v2, n, mat, allocs), vec3(0,0,0), vec3(0,0,0), vec3(1,1,1)));
}

inline void add_medium(world* w, const mesh* boundary, double density,
                       const texture& albedo, std::vector<void*>& allocs) {
    w->add(make_medium(boundary, vec3(0,0,0), vec3(0,0,0), vec3(1,1,1), density, albedo, allocs));
}

// --- meshes ------------------------------------------------------------------

inline mesh* new_mesh(std::vector<void*>& allocs, std::vector<mesh*>& mesh_dtors) {
    mesh* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(mesh)));
    new(m) mesh();
    allocs.push_back(m);
    mesh_dtors.push_back(m);
    return m;
}

inline void mesh_add_quad(mesh* m, const point3& Q, const vec3& u, const vec3& v,
                          material* mat, std::vector<void*>& allocs) {
    quad* q;
    checkCudaErrors(cudaMallocManaged((void**)&q, sizeof(quad)));
    new(q) quad(Q, u, v, mat);
    allocs.push_back(q);
    primitive p; p.type = QUAD; p.object = q;
    m->add(p);
}

inline void mesh_add_triangle(mesh* m, const point3& v0, const point3& v1,
                              const point3& v2, const vec3& n, material* mat,
                              std::vector<void*>& allocs) {
    triangle* t;
    checkCudaErrors(cudaMallocManaged((void**)&t, sizeof(triangle)));
    new(t) triangle(v0, v1, v2, n, mat);
    allocs.push_back(t);
    primitive p; p.type = TRIANGLE; p.object = t;
    m->add(p);
}

inline mesh* new_box(const point3& a, const point3& b, material* mat,
                     std::vector<void*>& allocs,
                     std::vector<mesh*>& mesh_dtors) {
    mesh* sides = new_mesh(allocs, mesh_dtors);

    auto min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
    auto max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

    auto dx = vec3(max.x() - min.x(), 0, 0);
    auto dy = vec3(0, max.y() - min.y(), 0);
    auto dz = vec3(0, 0, max.z() - min.z());

    mesh_add_quad(sides, point3(min.x(), min.y(), max.z()),  dx,  dy, mat, allocs);  // front
    mesh_add_quad(sides, point3(max.x(), min.y(), max.z()), -dz,  dy, mat, allocs);  // right
    mesh_add_quad(sides, point3(max.x(), min.y(), min.z()), -dx,  dy, mat, allocs);  // back
    mesh_add_quad(sides, point3(min.x(), min.y(), min.z()),  dz,  dy, mat, allocs);  // left
    mesh_add_quad(sides, point3(min.x(), max.y(), max.z()),  dx, -dz, mat, allocs);  // top
    mesh_add_quad(sides, point3(min.x(), min.y(), min.z()),  dx,  dz, mat, allocs);  // bottom

    sides->build();
    return sides;
}

#endif // SCENE_UTILS_H
