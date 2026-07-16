#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "aabb.h"
#include "cuda_helper.h"
#include "hit_record.h"
#include "hittable.h"
#include "interval.h"
#include "ray.h"

// Flat scene container: a growable unified-memory array of hittable*, hit by
// looping over every object (per-object AABB short-circuit, no tree). The BVH
// (bvh.h) is the accelerated alternative.

struct hittable_list {
    aabb bbox;
    hittable** objects;
    int size;
    int capacity;

    hittable_list() {
        size = 0;
        capacity = 16;
        bbox = aabb();
        checkCudaErrors(cudaMallocManaged((void**)&objects, capacity * sizeof(hittable*)));
    }

    hittable_list(hittable* object) {
        size = 0;
        capacity = 16;
        bbox = aabb();
        checkCudaErrors(cudaMallocManaged((void**)&objects, capacity * sizeof(hittable*)));
        add(object);
    }

    ~hittable_list() {
        cudaFree(objects);
    }

    __host__ void add(hittable* object);

    __host__ __device__ aabb bounding_box() const {return bbox;}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const;
};

__host__ void hittable_list::add(hittable* object) {
    if (size >= capacity) {
        capacity *= 2;
        hittable** new_objects;
        checkCudaErrors(cudaMallocManaged((void**)&new_objects, capacity * sizeof(hittable*)));
        for (int i = 0; i < size; i++) {
            new_objects[i] = objects[i];
        }
        cudaFree(objects);
        objects = new_objects;
    }
    objects[size] = object;
    size++;
    bbox = aabb(bbox, object->bounding_box());
}

__device__ bool hittable_list::hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
    if (!bbox.hit(r, ray_t)) return false;

    hit_record temp_rec;
    auto hit_anything = false;
    auto closest_so_far = ray_t.max;

    // Add bounds checking to prevent corruption issues
    // if (size < 0 || size > capacity || objects == nullptr) return false;

    for (int i = 0; i < size; i++) {
        if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec, state)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

// Dispatch shims declared in hittable.h (before hittable_list is complete) and
// defined here, so hittable's switches can route to the list.
__device__ bool hittable_list_hit(const hittable_list* list, const ray& r, interval ray_t, hit_record& rec, curandState* state) {
    return list->hit(r, ray_t, rec, state);
}

__host__ __device__ aabb hittable_list_bounding_box(const hittable_list* list) {
    return list->bounding_box();
}

#endif // HITTABLE_LIST_H
