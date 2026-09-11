#ifndef INSTANCE_H
#define INSTANCE_H

#include <curand_kernel.h>

#include "aabb.h"
#include "hit_record.h"
#include "hittables/bvh.h"
#include "hittables/primitive.h"
#include "hittables/transforms.h"
#include "interval.h"
#include "ray.h"
#include "util.h"
#include "vec3.h"

struct material;

enum LeafType {
  LEAF_PRIMITIVE,
  LEAF_MESH
};

struct instance {
  transform xf;
  aabb      bbox;
  LeafType  leaf_type;
  const void* leaf;            // primitive* / mesh*
  int       id;                // scene id
  bool      identity;          // xf is identity
  material* phase;
  real      neg_inv_density;   // constant medium: -1 / density

  instance()
  : xf(), leaf_type(LEAF_PRIMITIVE), leaf(nullptr), id(-1), identity(true), phase(nullptr), neg_inv_density(0) {}

  instance(const primitive* p, const vec3& t, const vec3& rot_deg, const vec3& s)
  : leaf_type(LEAF_PRIMITIVE), leaf(p), id(-1), phase(nullptr), neg_inv_density(0) {
    set_transform(t, rot_deg, s);
  }

  instance(const mesh* m, const vec3& t, const vec3& rot_deg, const vec3& s)
  : leaf_type(LEAF_MESH), leaf(m), id(-1), phase(nullptr), neg_inv_density(0) {
    set_transform(t, rot_deg, s);
  }

  __host__ __device__ const primitive* prim_ptr() const { return static_cast<const primitive*>(leaf); }
  __host__ __device__ const mesh* mesh_ptr() const { return static_cast<const mesh*>(leaf); }

  // the leaf's object-space aabb
  __host__ aabb leaf_bounding_box() const {
    return leaf_type == LEAF_PRIMITIVE ? prim_ptr()->bounding_box() : mesh_ptr()->bounding_box();
  }

  // set the transform of the instance
  __host__ void set_transform(const vec3& t, const vec3& rot_deg, const vec3& s) {
    xf = transform(t, rot_deg, s);
    identity = xf.is_identity();
    bbox = xf.bbox_to_world(leaf_bounding_box());
  }

  // turn the instance into a constant-density medium bounded by its leaf
  __host__ void set_medium(real density, material* phase_function) {
    phase = phase_function;
    neg_inv_density = real(-1) / density;
  }

  __host__ __device__ aabb bounding_box() const { return bbox; }
};

__device__ inline bool leaf_hit(const instance& in, const ray& r, interval ray_t, hit_record& rec, curandState* state) {
  return in.leaf_type == LEAF_PRIMITIVE ? in.prim_ptr()->hit(r, ray_t, rec, state)
                                        : in.mesh_ptr()->hit(r, ray_t, rec, state);
}

__device__ inline bool medium_hit(const instance& in, const ray& obj, interval ray_t, hit_record& rec, curandState* state, real world_dir_length) {
  hit_record rec1, rec2;
  const real inf = infinity;

  if (!leaf_hit(in, obj, interval(-inf, inf), rec1, state))
    return false;

  if (!leaf_hit(in, obj, interval(rec1.t + real(0.0001), inf), rec2, state))
    return false;

  if (rec1.t < ray_t.min) rec1.t = ray_t.min;
  if (rec2.t > ray_t.max) rec2.t = ray_t.max;

  if (rec1.t >= rec2.t)
    return false;

  if (rec1.t < 0)
    rec1.t = 0;

  auto distance_inside_boundary = (rec2.t - rec1.t) * world_dir_length;
  auto hit_distance = in.neg_inv_density * log(random_real(state));

  if (hit_distance > distance_inside_boundary)
    return false;

  rec.t = rec1.t + hit_distance / world_dir_length;
  rec.p = obj.at(rec.t);

  rec.normal = vec3(1, 0, 0);   // arbitrary, ignored by isotropic material
  rec.front_face = true;        // arbitrary, ignored by isotropic material
  rec.u = 0;
  rec.v = 0;
  rec.mat = in.phase;

  return true;
}

__device__ inline bool instance_hit(const instance& in, const ray& r, interval ray_t, hit_record& rec, curandState* state) {
  if (!in.bbox.hit(r, ray_t)) return false;

  if (in.identity) {
    bool hit = in.phase ? medium_hit(in, r, ray_t, rec, state, r.direction().length())
                        : leaf_hit(in, r, ray_t, rec, state);
    if (!hit) return false;
    rec.id = in.id;
    return true;
  }

  ray obj = in.xf.ray_to_object(r);
  bool hit = in.phase ? medium_hit(in, obj, ray_t, rec, state, r.direction().length())
                      : leaf_hit(in, obj, ray_t, rec, state);
  if (!hit) return false;
  rec.p = in.xf.point_to_world(rec.p);
  rec.normal = in.xf.normal_to_world(rec.normal);
  rec.id = in.id;
  return true;
}

__device__ inline bool hit_item(const instance& in, const ray& r, interval ray_t, hit_record& rec, curandState* state) {
  return instance_hit(in, r, ray_t, rec, state);
}

// TLAS
using world = bvh<instance>;

#endif // INSTANCE_H
