#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include <limits>

#include "aabb.h"
#include "ray.h"
#include "util.h"
#include "vec3.h"

struct transform {
  vec3 translation;
  vec3 rotation;    // Euler angles in degrees
  vec3 scale;
  vec3 inv_scale;
  vec3 m[3];        // rows of the combined rotation matrix R

  // R * v
  __host__ __device__ vec3 apply_R(const vec3& v) const {
    return vec3(dot(m[0], v), dot(m[1], v), dot(m[2], v));
  }

  // R^T * v
  __host__ __device__ vec3 apply_Rt(const vec3& v) const {
    return v.x() * m[0] + v.y() * m[1] + v.z() * m[2];
  }

  transform() : transform(vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 1)) {}

  transform(const vec3& t, const vec3& rot_deg, const vec3& s)
  : translation(t), rotation(rot_deg), scale(s) {
    inv_scale = vec3(real(1) / s.x(), real(1) / s.y(), real(1) / s.z());

    real a = degrees_to_radians(rot_deg.x());
    real b = degrees_to_radians(rot_deg.y());
    real g = degrees_to_radians(rot_deg.z());
    real ca = cos(a), sa = sin(a), cb = cos(b), sb = sin(b), cg = cos(g), sg = sin(g);

    // R = Rz(g) * Ry(b) * Rx(a)
    m[0] = vec3(cg*cb, cg*sb*sa - sg*ca, cg*sb*ca + sg*sa);
    m[1] = vec3(sg*cb, sg*sb*sa + cg*ca, sg*sb*ca - cg*sa);
    m[2] = vec3(  -sb,            cb*sa,            cb*ca);
  }

  __host__ bool is_identity() const {
    return translation.x() == 0 && translation.y() == 0 && translation.z() == 0 &&
           rotation.x() == 0 && rotation.y() == 0 && rotation.z() == 0 &&
           scale.x() == 1 && scale.y() == 1 && scale.z() == 1;
  }

  // Transform a ray from world space to object space. Direction is left UNNORMALIZED so the leaf's hit t carries back to world space unchanged.
  __host__ __device__ ray ray_to_object(const ray& r) const {
    vec3 o = apply_Rt(r.origin() - translation) * inv_scale;
    vec3 d = apply_Rt(r.direction()) * inv_scale;
    return ray(o, d);
  }

  __host__ __device__ point3 point_to_world(const point3& p) const {
    return apply_R(p * scale) + translation;
  }

  __host__ __device__ vec3 normal_to_world(const vec3& n) const {
    return unit_vector(apply_R(n * inv_scale));
  }

  __host__ aabb bbox_to_world(const aabb& c) const {
    auto inf = std::numeric_limits<real>::infinity();
    point3 lo( inf,  inf,  inf), hi(-inf, -inf, -inf);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) {
          vec3 corner(i ? c.x.max : c.x.min, j ? c.y.max : c.y.min, k ? c.z.max : c.z.min);
          vec3 w = point_to_world(corner);
          for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], w[d]); hi[d] = fmax(hi[d], w[d]); }
        }
    return aabb(lo, hi);
  }
};

#endif // TRANSFORMS_H
