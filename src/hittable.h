#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"
#include "cuda_helper.h"
#include "hit_record.h"
#include "interval.h"
#include "ray.h"
#include "hittables/sphere.h"
#include "hittables/quad.h"
#include "hittables/triangle.h"
#include "vec3.h"

struct material;

enum HittableType {
  HITTABLE_LIST,
  SPHERE,
  QUAD,
  TRIANGLE,
  TRANSLATE,
  ROTATE_Y,
  UNIFORM_SCALE,
  BVH
};

// The composite types — hittable_list, bvh_scene, and the instance transforms
// (translate / rotate_y / uniform_scale) — live in hittables/hittable_list.h /
// bvh.h / transforms.h, included at the bottom of this file after hittable is
// fully defined: they all reference hittable, so they can't be included up top
// like the leaf shapes. These shims let the dispatch switches below route to
// them while they are still incomplete types.
struct hittable_list;
__device__ bool hittable_list_hit(const hittable_list* list, const ray& r, interval ray_t, hit_record& rec);
__host__ __device__ aabb hittable_list_bounding_box(const hittable_list* list);

struct bvh_scene;
__device__ bool bvh_scene_hit(const bvh_scene* bvh, const ray& r, interval ray_t, hit_record& rec);
__host__ __device__ aabb bvh_scene_bounding_box(const bvh_scene* bvh);

struct translate;
__device__ bool translate_hit(const translate* t, const ray& r, interval ray_t, hit_record& rec);
__host__ __device__ aabb translate_bounding_box(const translate* t);

struct rotate_y;
__device__ bool rotate_y_hit(const rotate_y* rot, const ray& r, interval ray_t, hit_record& rec);
__host__ __device__ aabb rotate_y_bounding_box(const rotate_y* rot);

struct uniform_scale;
__device__ bool uniform_scale_hit(const uniform_scale* s, const ray& r, interval ray_t, hit_record& rec);
__host__ __device__ aabb uniform_scale_bounding_box(const uniform_scale* s);

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
        case QUAD:
        return static_cast<quad*>(object)->hit(r, ray_t, rec);
        case TRIANGLE:
        return static_cast<triangle*>(object)->hit(r, ray_t, rec);
        case TRANSLATE:
        return translate_hit(static_cast<translate*>(object), r, ray_t, rec);
        case ROTATE_Y:
        return rotate_y_hit(static_cast<rotate_y*>(object), r, ray_t, rec);
        case UNIFORM_SCALE:
        return uniform_scale_hit(static_cast<uniform_scale*>(object), r, ray_t, rec);
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
        case QUAD:
        return static_cast<quad*>(object)->bounding_box();
        case TRIANGLE:
        return static_cast<triangle*>(object)->bounding_box();
        case TRANSLATE:
        return translate_bounding_box(static_cast<translate*>(object));
        case ROTATE_Y:
        return rotate_y_bounding_box(static_cast<rotate_y*>(object));
        case UNIFORM_SCALE:
        return uniform_scale_bounding_box(static_cast<uniform_scale*>(object));
        case BVH:
        return bvh_scene_bounding_box(static_cast<bvh_scene*>(object));
        default:
        return aabb();
    }
}

// Define hittable_list / bvh_scene / the transforms and the shims declared
// above. Included last so that including hittable.h alone is enough to compile
// the switch cases (the include guards make any include order — hittable.h
// first or any concrete header first — resolve correctly).
#include "hittables/hittable_list.h"
#include "hittables/bvh.h"
#include "hittables/transforms.h"

#endif