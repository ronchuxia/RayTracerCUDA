#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "cuda_helper.h"
#include "hittable.h"

// Host-side scene registry (the "mutable scene + stable ids" foundation).
//
// Owns a persistent, addressable world: every object registered via add()
// gets a stable integer id (its index in `objects`), stamped on the OUTERMOST
// wrapper of the object — a bare sphere wrapper, or the top of a transform
// chain (translate(rotate(box))), or a constant_medium. hittable::hit()
// writes that id into hit_record.id on a hit, so a single ray cast through a
// cursor pixel identifies the scene object (B4 picking), and get(id) maps the
// id back to the wrapper for inspection or mutation.
//
// Nothing is freed until release() is called explicitly — the scene outlives
// any number of renders (the interactive viewer keeps one alive for its whole
// run; dynamic scenes will mutate it between frames).
//
// Mutation protocol (workstream C): objects live in managed memory, so the
// host can rewrite their parameters directly between renders. Anything that
// changes geometry must then restore two invariants:
//   1. the object's own cached bbox (e.g. re-placement-new the shape, or
//      recompute the transform's bbox), and
//   2. the acceleration structure: call refit() (topology unchanged) or
//      build() (objects added/removed).
// The flat hittable_list caches only the UNION of bboxes at add() time and
// cannot shrink or follow moves — mutation is only valid on the BVH path.
struct scene {
    hittable_list* world = nullptr;          // flat container (build/traverse source)
    hittable*      world_hittable = nullptr; // world wrapped for the camera (id -1)
    bvh*     world_bvh = nullptr;            // built over world's objects
    hittable*      world_bvh_hittable = nullptr;   // world_bvh wrapped for the camera (id -1)

    std::vector<void*>          allocs;      // every cudaMallocManaged block: bare cudaFree
    std::vector<hittable_list*> list_dtors;  // interior lists (boxes): dtor before the free loop
    std::vector<bvh*>     bvh_dtors;   // per-mesh BVHs (STL): dtor before the free loop
    std::vector<hittable*>      objects;     // id -> outermost wrapper

    // Allocate the (empty) containers: the world list, the BVH, and both
    // camera-facing wrappers. After init() every field is valid; add() fills
    // the world and build() indexes it.
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

    // Register a scene object: assign the next stable id to its outermost
    // wrapper, add it to the world, and remember it for get().
    int add(hittable* h) {
        int id = (int)objects.size();
        h->id = id;
        world->add(h);
        objects.push_back(h);
        return id;
    }

    hittable* get(int id) const {
        return (id >= 0 && id < (int)objects.size()) ? objects[id] : nullptr;
    }

    // (Re)build the BVH over the world's current objects: re-copy the wrappers
    // (membership/ids/params may have changed since the last build), then
    // reconstruct the tree.
    void build() {
        world_bvh->prim_count = 0;
        for (int i = 0; i < world->size; i++)
            world_bvh->add(*world->objects[i]);
        world_bvh->build();
    }

    // Recompute BVH node boxes bottom-up after objects moved (same topology).
    void refit() { world_bvh->refit(); }

    // What the camera renders: always the BVH root. Before build() the tree is
    // empty and renders nothing — an obvious failure, unlike silently handing
    // out the flat world, whose grow-only bbox is invalid under mutation. The
    // flat path stays available EXPLICITLY (world_hittable) for static
    // flat-vs-BVH verification only.
    hittable& root() const { return *world_bvh_hittable; }

    // Explicit teardown — the scene is persistent until this is called.
    void release() {
        for (bvh* m : bvh_dtors) m->~bvh();
        for (hittable_list* l : list_dtors) l->~hittable_list();
        for (void* p : allocs) cudaFree(p);
        if (world_bvh) { world_bvh->~bvh(); cudaFree(world_bvh); cudaFree(world_bvh_hittable); }
        if (world) { world->~hittable_list(); cudaFree(world); cudaFree(world_hittable); }
        *this = scene();
    }
};

#endif // SCENE_H
