#ifndef SCENE_UTILS_H
#define SCENE_UTILS_H

#include <vector>

#include "cuda_helper.h"
#include "hittable.h"
#include "material.h"
#include "vec3.h"

// Host-only scene-construction helpers. Each follows the repo's three-step
// pattern (material / concrete shape / hittable wrapper — see CLAUDE.md) but
// records every cudaMallocManaged allocation in `allocs` so a scene can tear
// down with one `for (void* p : allocs) cudaFree(p);` loop instead of naming
// each free. All are `inline` (header-defined, possibly unused per scene).

// --- materials -------------------------------------------------------------

inline material* new_lambertian(const color& albedo, std::vector<void*>& allocs) {
    material* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(material)));
    m->type = LAMBERTIAN;
    m->lam = lambertian(albedo);
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

inline material* new_diffuse_light(const color& emit, std::vector<void*>& allocs) {
    material* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(material)));
    m->type = DIFFUSE_LIGHT;
    m->light = diffuse_light(emit);
    allocs.push_back(m);
    return m;
}

// --- shapes ----------------------------------------------------------------

inline void add_sphere(hittable_list* world, const point3& center, double radius,
                       material* mat, std::vector<void*>& allocs) {
    sphere* s;
    checkCudaErrors(cudaMallocManaged((void**)&s, sizeof(sphere)));
    new(s) sphere(center, radius, mat);

    hittable* h;
    checkCudaErrors(cudaMallocManaged((void**)&h, sizeof(hittable)));
    h->type = SPHERE;
    h->object = s;

    world->add(h);
    allocs.push_back(s);
    allocs.push_back(h);
}

inline void add_quad(hittable_list* world, const point3& Q, const vec3& u, const vec3& v,
                     material* mat, std::vector<void*>& allocs) {
    quad* q;
    checkCudaErrors(cudaMallocManaged((void**)&q, sizeof(quad)));
    new(q) quad(Q, u, v, mat);

    hittable* h;
    checkCudaErrors(cudaMallocManaged((void**)&h, sizeof(hittable)));
    h->type = QUAD;
    h->object = q;

    world->add(h);
    allocs.push_back(q);
    allocs.push_back(h);
}

inline void add_triangle(hittable_list* world, const point3& v0, const point3& v1,
                         const point3& v2, const vec3& n, material* mat,
                         std::vector<void*>& allocs) {
    triangle* t;
    checkCudaErrors(cudaMallocManaged((void**)&t, sizeof(triangle)));
    new(t) triangle(v0, v1, v2, n, mat);

    hittable* h;
    checkCudaErrors(cudaMallocManaged((void**)&h, sizeof(hittable)));
    h->type = TRIANGLE;
    h->object = t;

    world->add(h);
    allocs.push_back(t);
    allocs.push_back(h);
}

// Appends the six quads of the axis-aligned box spanning opposite vertices
// a & b. The quads are added individually (not as a nested sub-list) so the
// BVH can split them and device traversal stays recursion-free.
inline void box(const point3& a, const point3& b, material* mat,
                hittable_list* world, std::vector<void*>& allocs) {
    auto min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
    auto max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

    auto dx = vec3(max.x() - min.x(), 0, 0);
    auto dy = vec3(0, max.y() - min.y(), 0);
    auto dz = vec3(0, 0, max.z() - min.z());

    add_quad(world, point3(min.x(), min.y(), max.z()),  dx,  dy, mat, allocs);  // front
    add_quad(world, point3(max.x(), min.y(), max.z()), -dz,  dy, mat, allocs);  // right
    add_quad(world, point3(max.x(), min.y(), min.z()), -dx,  dy, mat, allocs);  // back
    add_quad(world, point3(min.x(), min.y(), min.z()),  dz,  dy, mat, allocs);  // left
    add_quad(world, point3(min.x(), max.y(), max.z()),  dx, -dz, mat, allocs);  // top
    add_quad(world, point3(min.x(), min.y(), min.z()),  dx,  dz, mat, allocs);  // bottom
}

#endif // SCENE_UTILS_H
