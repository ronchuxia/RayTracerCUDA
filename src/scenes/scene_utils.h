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

// --- composites & transforms -------------------------------------------------

// Builds the six-quad axis-aligned box spanning opposite vertices a & b as a
// hittable_list wrapped in a hittable, so it can be wrapped in transforms
// (new_rotate_y / new_translate) or added to the world as a unit. The inner
// list is also recorded in `sublists`: its destructor must run at teardown
// (it frees the list's internal objects array), before the allocs free loop.
inline hittable* new_box(const point3& a, const point3& b, material* mat,
                         std::vector<void*>& allocs,
                         std::vector<hittable_list*>& sublists) {
    hittable_list* sides;
    checkCudaErrors(cudaMallocManaged((void**)&sides, sizeof(hittable_list)));
    new(sides) hittable_list();

    hittable* h;
    checkCudaErrors(cudaMallocManaged((void**)&h, sizeof(hittable)));
    h->type = HITTABLE_LIST;
    h->object = sides;

    allocs.push_back(sides);
    allocs.push_back(h);
    sublists.push_back(sides);

    auto min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
    auto max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

    auto dx = vec3(max.x() - min.x(), 0, 0);
    auto dy = vec3(0, max.y() - min.y(), 0);
    auto dz = vec3(0, 0, max.z() - min.z());

    add_quad(sides, point3(min.x(), min.y(), max.z()),  dx,  dy, mat, allocs);  // front
    add_quad(sides, point3(max.x(), min.y(), max.z()), -dz,  dy, mat, allocs);  // right
    add_quad(sides, point3(max.x(), min.y(), min.z()), -dx,  dy, mat, allocs);  // back
    add_quad(sides, point3(min.x(), min.y(), min.z()),  dz,  dy, mat, allocs);  // left
    add_quad(sides, point3(min.x(), max.y(), max.z()),  dx, -dz, mat, allocs);  // top
    add_quad(sides, point3(min.x(), min.y(), min.z()),  dx,  dz, mat, allocs);  // bottom

    return h;
}

// Volume wrapper: turns a boundary hittable (a box, sphere, or transform
// chain) into a constant-density medium with an isotropic phase function.
inline hittable* new_constant_medium(hittable* boundary, double density,
                                     const texture& albedo,
                                     std::vector<void*>& allocs) {
    material* phase = new_isotropic(albedo, allocs);

    constant_medium* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(constant_medium)));
    new(m) constant_medium(boundary, density, phase);

    hittable* h;
    checkCudaErrors(cudaMallocManaged((void**)&h, sizeof(hittable)));
    h->type = CONSTANT_MEDIUM;
    h->object = m;

    allocs.push_back(m);
    allocs.push_back(h);
    return h;
}

// Instance-transform wrappers. Each wraps a child hittable* and returns the
// wrapper hittable*, so they chain: scale first, then rotate, then translate,
// e.g. new_translate(new_rotate_y(new_box(...), 15), vec3(265,0,295), ...).

inline hittable* new_translate(hittable* child, const vec3& offset,
                               std::vector<void*>& allocs) {
    translate* t;
    checkCudaErrors(cudaMallocManaged((void**)&t, sizeof(translate)));
    new(t) translate(child, offset);

    hittable* h;
    checkCudaErrors(cudaMallocManaged((void**)&h, sizeof(hittable)));
    h->type = TRANSLATE;
    h->object = t;

    allocs.push_back(t);
    allocs.push_back(h);
    return h;
}

inline hittable* new_rotate_y(hittable* child, double angle_degrees,
                              std::vector<void*>& allocs) {
    rotate_y* rot;
    checkCudaErrors(cudaMallocManaged((void**)&rot, sizeof(rotate_y)));
    new(rot) rotate_y(child, angle_degrees);

    hittable* h;
    checkCudaErrors(cudaMallocManaged((void**)&h, sizeof(hittable)));
    h->type = ROTATE_Y;
    h->object = rot;

    allocs.push_back(rot);
    allocs.push_back(h);
    return h;
}

inline hittable* new_uniform_scale(hittable* child, double scale,
                                   std::vector<void*>& allocs) {
    uniform_scale* s;
    checkCudaErrors(cudaMallocManaged((void**)&s, sizeof(uniform_scale)));
    new(s) uniform_scale(child, scale);

    hittable* h;
    checkCudaErrors(cudaMallocManaged((void**)&h, sizeof(hittable)));
    h->type = UNIFORM_SCALE;
    h->object = s;

    allocs.push_back(s);
    allocs.push_back(h);
    return h;
}

#endif // SCENE_UTILS_H
