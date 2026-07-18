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
  CONSTANT_MEDIUM,
  BVH
};

// The composite types — hittable_list, bvh_scene, and the instance transforms
// (translate / rotate_y / uniform_scale) — live in hittables/hittable_list.h /
// bvh.h / transforms.h, included at the bottom of this file after hittable is
// fully defined: they all reference hittable, so they can't be included up top
// like the leaf shapes. These shims let the dispatch switches below route to
// them while they are still incomplete types.
struct hittable_list;
__device__ bool hittable_list_hit(const hittable_list* list, const ray& r, interval ray_t, hit_record& rec, curandState* state);
__host__ __device__ aabb hittable_list_bounding_box(const hittable_list* list);

struct bvh_scene;
__device__ bool bvh_scene_hit(const bvh_scene* bvh, const ray& r, interval ray_t, hit_record& rec, curandState* state);
__host__ __device__ aabb bvh_scene_bounding_box(const bvh_scene* bvh);

struct translate;
__device__ bool translate_hit(const translate* t, const ray& r, interval ray_t, hit_record& rec, curandState* state);
__host__ __device__ aabb translate_bounding_box(const translate* t);

struct rotate_y;
__device__ bool rotate_y_hit(const rotate_y* rot, const ray& r, interval ray_t, hit_record& rec, curandState* state);
__host__ __device__ aabb rotate_y_bounding_box(const rotate_y* rot);

struct uniform_scale;
__device__ bool uniform_scale_hit(const uniform_scale* s, const ray& r, interval ray_t, hit_record& rec, curandState* state);
__host__ __device__ aabb uniform_scale_bounding_box(const uniform_scale* s);

struct constant_medium;
__device__ bool constant_medium_hit(const constant_medium* m, const ray& r, interval ray_t, hit_record& rec, curandState* state);
__host__ __device__ aabb constant_medium_bounding_box(const constant_medium* m);

struct hittable {
  HittableType type;
  void* object;
  int id = -1;   // stable scene-object id; -1 = untagged (containers, sub-parts)

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const;

  __host__ __device__ aabb bounding_box() const;
};

__device__ bool hittable::hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
    bool is_hit;
    switch (type) {
        case HITTABLE_LIST:
        is_hit = hittable_list_hit(static_cast<hittable_list*>(object), r, ray_t, rec, state); break;
        case SPHERE:
        is_hit = static_cast<sphere*>(object)->hit(r, ray_t, rec, state); break;
        case QUAD:
        is_hit = static_cast<quad*>(object)->hit(r, ray_t, rec, state); break;
        case TRIANGLE:
        is_hit = static_cast<triangle*>(object)->hit(r, ray_t, rec, state); break;
        case TRANSLATE:
        is_hit = translate_hit(static_cast<translate*>(object), r, ray_t, rec, state); break;
        case ROTATE_Y:
        is_hit = rotate_y_hit(static_cast<rotate_y*>(object), r, ray_t, rec, state); break;
        case UNIFORM_SCALE:
        is_hit = uniform_scale_hit(static_cast<uniform_scale*>(object), r, ray_t, rec, state); break;
        case CONSTANT_MEDIUM:
        is_hit = constant_medium_hit(static_cast<constant_medium*>(object), r, ray_t, rec, state); break;
        case BVH:
        is_hit = bvh_scene_hit(static_cast<bvh_scene*>(object), r, ray_t, rec, state); break;
        default:
        is_hit = false;
    }
    // Stamp the hit with this wrapper's scene-object id. Dispatch nests
    // outside-in and returns inside-out, so the OUTERMOST tagged wrapper's
    // stamp lands last: a hit on a box face or mesh triangle reports the id
    // of the transform-chain wrapper registered with the scene, not the
    // internal part. Untagged wrappers (id < 0: the world list/BVH,
    // box-interior quads, mesh triangles) never overwrite.
    if (is_hit && id >= 0) rec.id = id;
    return is_hit;
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
        case CONSTANT_MEDIUM:
        return constant_medium_bounding_box(static_cast<constant_medium*>(object));
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
#include "hittables/constant_medium.h"

#endif