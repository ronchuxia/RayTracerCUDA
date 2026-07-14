#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"
#include "cuda_helper.h"
#include "hit_record.h"
#include "interval.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"

struct material;

enum HittableType {
  HITTABLE_LIST,
  SPHERE
};

struct hittable {
  HittableType type;
  void* object;

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const;

  __host__ __device__ aabb bounding_box() const;
};

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

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const;
};

__device__ bool hittable::hit(const ray& r, interval ray_t, hit_record& rec) const {
    switch (type) {
        case HITTABLE_LIST:
        return static_cast<hittable_list*>(object)->hit(r, ray_t, rec);
        case SPHERE:
        return static_cast<sphere*>(object)->hit(r, ray_t, rec);
        default:
        return false;
    }
}

__host__ __device__ aabb hittable::bounding_box() const {
    switch (type) {
        case HITTABLE_LIST:
        return static_cast<hittable_list*>(object)->bounding_box();
        case SPHERE:
        return static_cast<sphere*>(object)->bounding_box();
        default:
        return aabb();
    }
}

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

__device__ bool hittable_list::hit(const ray& r, interval ray_t, hit_record& rec) const {
    if (!bbox.hit(r, ray_t)) return false;

    hit_record temp_rec;
    auto hit_anything = false;
    auto closest_so_far = ray_t.max;

    // Add bounds checking to prevent corruption issues
    // if (size < 0 || size > capacity || objects == nullptr) return false;

    for (int i = 0; i < size; i++) {
        if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif