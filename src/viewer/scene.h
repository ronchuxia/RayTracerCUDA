#ifndef VIEWER_SCENE_H
#define VIEWER_SCENE_H

#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda_helper.h"
#include "hittable.h"
#include "physics/body.h"

struct scene {
    world* w = nullptr;

    std::vector<void*> allocs;      // tracker for cudaMallocManaged
    std::vector<mesh*> mesh_dtors;  // tracker for meshes

    // initial camera
    point3 lookfrom, lookat;
    real   vfov = real(20);
    // initial physics bodies
    std::vector<phys_body> bodies;

    void init() {
        checkCudaErrors(cudaMallocManaged((void**)&w, sizeof(world)));
        new(w) world();
    }

    // register a scene object
    int add(instance in) {
        in.id = w->item_count;
        w->add(in);
        return in.id;
    }

    instance* get(int id) const {
        return (id >= 0 && id < w->item_count) ? &w->items[id] : nullptr;
    }

    int size() const { return w->item_count; }

    // build the BVH after scene construction
    void build() { w->build(); }

    // refit the BVH after scene objects moved
    void refit() { w->refit(); }

    world& root() const { return *w; }

    // teardown
    void release() {
        for (mesh* m : mesh_dtors) m->~mesh();
        for (void* p : allocs) cudaFree(p);
        if (w) { w->~world(); cudaFree(w); }
        *this = scene();
    }
};

#endif // VIEWER_SCENE_H
