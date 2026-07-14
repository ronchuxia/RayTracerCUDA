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
  SPHERE,
  BVH
};

// hittable_list and bvh_scene live in hittable_list.h / bvh.h (included at the
// bottom of this file, after hittable is fully defined — both reference
// hittable, so they can't be included up top like sphere.h). These shims let
// the dispatch switches below route to them while they are still incomplete
// types.
struct hittable_list;
__device__ bool hittable_list_hit(const hittable_list* list, const ray& r, interval ray_t, hit_record& rec);
__host__ __device__ aabb hittable_list_bounding_box(const hittable_list* list);

struct bvh_scene;
__device__ bool bvh_scene_hit(const bvh_scene* bvh, const ray& r, interval ray_t, hit_record& rec);
__host__ __device__ aabb bvh_scene_bounding_box(const bvh_scene* bvh);

struct hittable {
  HittableType type;
  void* object;

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const;

  __host__ __device__ aabb bounding_box() const;
};

__device__ bool hittable::hit(const ray& r, interval ray_t, hit_record& rec) const {
    switch (type) {
        case HITTABLE_LIST:
        return hittable_list_hit(static_cast<hittable_list*>(object), r, ray_t, rec);
        case SPHERE:
        return static_cast<sphere*>(object)->hit(r, ray_t, rec);
        case BVH:
        return bvh_scene_hit(static_cast<bvh_scene*>(object), r, ray_t, rec);
        default:
        return false;
    }
}

__host__ __device__ aabb hittable::bounding_box() const {
    switch (type) {
        case HITTABLE_LIST:
        return hittable_list_bounding_box(static_cast<hittable_list*>(object));
        case SPHERE:
        return static_cast<sphere*>(object)->bounding_box();
        case BVH:
        return bvh_scene_bounding_box(static_cast<bvh_scene*>(object));
        default:
        return aabb();
    }
}

// Define hittable_list / bvh_scene and the shims declared above. Included last
// so that including hittable.h alone is enough to compile the switch cases
// (the include guards make any include order — hittable.h first or either
// concrete header first — resolve correctly).
#include "hittable_list.h"
#include "bvh.h"

#endif