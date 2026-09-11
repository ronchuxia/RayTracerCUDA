#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include <curand_kernel.h>

#include "aabb.h"
#include "hit_record.h"
#include "hittables/bvh.h"
#include "hittables/quad.h"
#include "hittables/sphere.h"
#include "hittables/triangle.h"
#include "interval.h"
#include "ray.h"

enum PrimitiveType {
  SPHERE,
  QUAD,
  TRIANGLE
};

struct primitive {
  PrimitiveType type;
  void* object; // sphere* / quad* / triangle*

  __host__ __device__ aabb bounding_box() const {
    switch (type) {
      case SPHERE:   return static_cast<const sphere*>(object)->bounding_box();
      case QUAD:     return static_cast<const quad*>(object)->bounding_box();
      case TRIANGLE: return static_cast<const triangle*>(object)->bounding_box();
      default:       return aabb();
    }
  }

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
    switch (type) {
      case SPHERE:   return static_cast<const sphere*>(object)->hit(r, ray_t, rec, state);
      case QUAD:     return static_cast<const quad*>(object)->hit(r, ray_t, rec, state);
      case TRIANGLE: return static_cast<const triangle*>(object)->hit(r, ray_t, rec, state);
      default:       return false;
    }
  }
};

__device__ inline bool hit_item(const primitive& p, const ray& r, interval ray_t, hit_record& rec, curandState* state) {
  return p.hit(r, ray_t, rec, state);
}

// BLAS
using mesh = bvh<primitive>;

#endif // PRIMITIVE_H
