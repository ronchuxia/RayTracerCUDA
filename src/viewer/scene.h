#ifndef VIEWER_SCENE_H
#define VIEWER_SCENE_H

#include <vector>

#include "cuda_helper.h"
#include "hittable.h"
#include "physics/body.h"

struct scene {
    hittable_list* world = nullptr;          // list of objects
    hittable*      world_hittable = nullptr; // world wrapped for camera
    bvh*     world_bvh = nullptr;            // BVH of objects
    hittable*      world_bvh_hittable = nullptr;   // world_bvh wrapped for camera

    std::vector<void*>          allocs;      // destructor that tracks cudaMallocManaged for materials, shapes, transforms, and hittable wrappers
    std::vector<hittable_list*> list_dtors;  // destructor for internal hittable_lists (e.g. new_box)
    std::vector<bvh*>     bvh_dtors;         // destructor for internal bvhs
    std::vector<hittable*>      objects;

    // initial camera
    point3 lookfrom, lookat;
    real   vfov = real(20);
    // initial physics bodies
    std::vector<phys_body> bodies;

    void init() {
        checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(hittable_list)));
        new(world) hittable_list();

        checkCudaErrors(cudaMallocManaged((void**)&world_hittable, sizeof(hittable)));
        world_hittable->type = HITTABLE_LIST;
        world_hittable->id = -1;
        world_hittable->object = world;

        checkCudaErrors(cudaMallocManaged((void**)&world_bvh, sizeof(bvh)));
        new(world_bvh) bvh();

        checkCudaErrors(cudaMallocManaged((void**)&world_bvh_hittable, sizeof(hittable)));
        world_bvh_hittable->type = BVH;
        world_bvh_hittable->id = -1;
        world_bvh_hittable->object = world_bvh;
    }

    // register a scene object
    int add(hittable* h) {
        int id = (int)objects.size();
        h->id = id; // assign scene object id
        world->add(h);
        objects.push_back(h);
        return id;
    }

    hittable* get(int id) const {
        return (id >= 0 && id < (int)objects.size()) ? objects[id] : nullptr;
    }

    // build the BVH after scene construction
    void build() {
        world_bvh->prim_count = 0;
        for (int i = 0; i < world->size; i++)
            world_bvh->add(*world->objects[i]);
        world_bvh->build();
    }

    // refit the BVH after scene objects moved
    void refit() { world_bvh->refit(); }

    hittable& root() const { return *world_bvh_hittable; }

    // teardown
    void release() {
        for (bvh* m : bvh_dtors) m->~bvh();
        for (hittable_list* l : list_dtors) l->~hittable_list();
        for (void* p : allocs) cudaFree(p);
        if (world_bvh) { world_bvh->~bvh(); cudaFree(world_bvh); cudaFree(world_bvh_hittable); }
        if (world) { world->~hittable_list(); cudaFree(world); cudaFree(world_hittable); }
        *this = scene();
    }
};

#endif // VIEWER_SCENE_H
