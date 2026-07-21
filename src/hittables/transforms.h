#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include <limits>

#include "aabb.h"
#include "hit_record.h"
#include "hittable.h"
#include "interval.h"
#include "ray.h"
#include "util.h"
#include "vec3.h"

// Instance transforms: hittables that wrap a child hittable* and reposition it
// without moving the geometry — hit() transforms the ray into the child's
// object space, delegates, and transforms the hit point/normal back out.
// Composite shapes like hittable_list/bvh: hittable.h forward-declares
// their dispatch shims and includes this header at the bottom. Transforms
// compose (e.g. translate(rotate_y(box))); each level delegates through
// hittable::hit, a device-recursive chain bounded by the construction depth.

struct translate {
  hittable* child;
  vec3 offset;
  aabb bbox;

  translate(hittable* _child, const vec3& _offset)
  : child(_child), offset(_offset) {
    bbox = child->bounding_box() + offset;
  }

  __host__ __device__ aabb bounding_box() const {return bbox;}

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
    // Move the ray backwards by the offset
    ray offset_r(r.origin() - offset, r.direction());

    // Determine whether an intersection exists along the offset ray (and if so, where)
    if (!child->hit(offset_r, ray_t, rec, state))
        return false;

    // Move the intersection point forwards by the offset
    rec.p += offset;

    return true;
  }
};

struct rotate_y {
  hittable* child;
  real sin_theta;
  real cos_theta;
  aabb bbox;

  rotate_y(hittable* _child, real angle) : child(_child) {
    auto radians = degrees_to_radians(angle);
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    bbox = child->bounding_box();

    // Rotate all 8 corners of the child's box and re-union them.
    auto inf = std::numeric_limits<real>::infinity();
    point3 min( inf,  inf,  inf);
    point3 max(-inf, -inf, -inf);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                auto x = i*bbox.x.max + (1-i)*bbox.x.min;
                auto y = j*bbox.y.max + (1-j)*bbox.y.min;
                auto z = k*bbox.z.max + (1-k)*bbox.z.min;

                auto newx =  cos_theta*x + sin_theta*z;
                auto newz = -sin_theta*x + cos_theta*z;

                vec3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    bbox = aabb(min, max);
  }

  __host__ __device__ aabb bounding_box() const {return bbox;}

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
    // Change the ray from world space to object space
    auto origin = r.origin();
    auto direction = r.direction();

    origin[0] = cos_theta*r.origin()[0] - sin_theta*r.origin()[2];
    origin[2] = sin_theta*r.origin()[0] + cos_theta*r.origin()[2];

    direction[0] = cos_theta*r.direction()[0] - sin_theta*r.direction()[2];
    direction[2] = sin_theta*r.direction()[0] + cos_theta*r.direction()[2];

    ray rotated_r(origin, direction);

    // Determine whether an intersection exists in object space (and if so, where)
    if (!child->hit(rotated_r, ray_t, rec, state))
        return false;

    // Change the intersection point from object space to world space
    auto p = rec.p;
    p[0] =  cos_theta*rec.p[0] + sin_theta*rec.p[2];
    p[2] = -sin_theta*rec.p[0] + cos_theta*rec.p[2];

    // Change the normal from object space to world space (rotation preserves
    // dot products, so the front_face flag set in the child stays valid)
    auto normal = rec.normal;
    normal[0] =  cos_theta*rec.normal[0] + sin_theta*rec.normal[2];
    normal[2] = -sin_theta*rec.normal[0] + cos_theta*rec.normal[2];

    rec.p = p;
    rec.normal = normal;

    return true;
  }
};

struct uniform_scale {
  hittable* child;
  real scale;   // assumed positive
  aabb bbox;

  uniform_scale(hittable* _child, real _scale)
  : child(_child), scale(_scale) {
    bbox = child->bounding_box() * scale;
  }

  __host__ __device__ aabb bounding_box() const {return bbox;}

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
    // Scale the ray into object space. Origin AND direction are both scaled,
    // so the hit parameter t carries over to world space unchanged; the
    // normal's direction is preserved by a uniform scale.
    ray scaled_r(r.origin() * (1/scale), r.direction() * (1/scale));

    if (!child->hit(scaled_r, ray_t, rec, state))
        return false;

    // Scale the intersection point back to world space
    rec.p *= scale;

    return true;
  }
};

// Full TRS instance transform: translation + Euler rotation (degrees) +
// per-axis scale, in one node (vs. chaining translate/rotate_y/uniform_scale).
// Rotation order is fixed R = Rz * Ry * Rx (apply X, then Y, then Z). The
// forward map (object -> world) of a point is  p_world = T + R (S . p_obj),
// where "." is per-component scale. The rotation matrix rows are cached in m[];
// scale's reciprocal in inv_scale.
struct transform {
  hittable* child;
  vec3 translation;
  vec3 rotation;    // Euler angles in DEGREES (kept for the editor to read back)
  vec3 scale;
  vec3 inv_scale;   // 1 / scale, per component
  vec3 m[3];        // rows of the combined rotation matrix R
  aabb bbox;

  // R * v  (m holds R's rows)
  __host__ __device__ vec3 apply_R(const vec3& v) const {
    return vec3(dot(m[0], v), dot(m[1], v), dot(m[2], v));
  }
  // R^T * v  (= R^-1 v, since R is orthonormal): columns of R are v-weighted rows
  __host__ __device__ vec3 apply_Rt(const vec3& v) const {
    return v.x() * m[0] + v.y() * m[1] + v.z() * m[2];
  }

  transform(hittable* _child, const vec3& t, const vec3& rot_deg, const vec3& s)
  : child(_child), translation(t), rotation(rot_deg), scale(s) {
    inv_scale = vec3(real(1) / s.x(), real(1) / s.y(), real(1) / s.z());

    real a = degrees_to_radians(rot_deg.x());
    real b = degrees_to_radians(rot_deg.y());
    real g = degrees_to_radians(rot_deg.z());
    real ca = cos(a), sa = sin(a), cb = cos(b), sb = sin(b), cg = cos(g), sg = sin(g);

    // R = Rz(g) * Ry(b) * Rx(a), stored as rows (standard Tait-Bryan ZYX).
    m[0] = vec3(cg*cb, cg*sb*sa - sg*ca, cg*sb*ca + sg*sa);
    m[1] = vec3(sg*cb, sg*sb*sa + cg*ca, sg*sb*ca - cg*sa);
    m[2] = vec3(  -sb,            cb*sa,            cb*ca);

    // bbox of the transformed child. A SPHERE is the one leaf whose AABB (a
    // cube) is looser than its geometry, so rotating that cube's 8 corners would
    // spuriously inflate the box — even though a sphere's AABB is rotation-
    // invariant. Bound the exact transformed sphere (an ellipsoid) instead:
    // center via the forward map, half-extent along world axis k =
    // radius * || row_k(R · diag(scale)) ||. Every OTHER leaf (quad/triangle/box)
    // equals its AABB corners, so transforming those corners is already tight.
    auto inf = std::numeric_limits<real>::infinity();
    point3 lo( inf,  inf,  inf), hi(-inf, -inf, -inf);
    if (child->type == SPHERE) {
      const sphere* sp = static_cast<const sphere*>(child->object);
      vec3 center = apply_R(sp->center * scale) + translation;   // ellipsoid center
      vec3 ext;
      for (int k = 0; k < 3; k++) {
        vec3 row(m[k].x() * scale.x(), m[k].y() * scale.y(), m[k].z() * scale.z());
        ext[k] = sp->radius * row.length();
      }
      lo = center - ext;  hi = center + ext;
    } else {
      aabb c = child->bounding_box();
      for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
          for (int k = 0; k < 2; k++) {
            vec3 corner(i ? c.x.max : c.x.min, j ? c.y.max : c.y.min, k ? c.z.max : c.z.min);
            vec3 w = apply_R(corner * scale) + translation;   // object -> world
            for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], w[d]); hi[d] = fmax(hi[d], w[d]); }
          }
    }
    bbox = aabb(lo, hi);
  }

  __host__ __device__ aabb bounding_box() const { return bbox; }

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
    // World -> object: o' = S^-1 R^T (o - T), d' = S^-1 R^T d. Direction is left
    // UNNORMALIZED so the child's hit t carries back to world space unchanged.
    vec3 o = apply_Rt(r.origin() - translation) * inv_scale;
    vec3 d = apply_Rt(r.direction()) * inv_scale;

    if (!child->hit(ray(o, d), ray_t, rec, state))
        return false;

    // Object -> world: point via the forward map; normal via the
    // inverse-transpose (R * S^-1), renormalized (non-uniform scale changes its
    // length). The sign of dot(dir, normal) is preserved by this pair, so the
    // child's front_face flag stays valid.
    rec.p = apply_R(rec.p * scale) + translation;
    rec.normal = unit_vector(apply_R(rec.normal * inv_scale));
    return true;
  }
};

// Dispatch shims declared in hittable.h (before the transforms are complete)
// and defined here, so hittable's switches can route to them.
__device__ bool translate_hit(const translate* t, const ray& r, interval ray_t, hit_record& rec, curandState* state) {
    return t->hit(r, ray_t, rec, state);
}

__host__ __device__ aabb translate_bounding_box(const translate* t) {
    return t->bounding_box();
}

__device__ bool rotate_y_hit(const rotate_y* rot, const ray& r, interval ray_t, hit_record& rec, curandState* state) {
    return rot->hit(r, ray_t, rec, state);
}

__host__ __device__ aabb rotate_y_bounding_box(const rotate_y* rot) {
    return rot->bounding_box();
}

__device__ bool uniform_scale_hit(const uniform_scale* s, const ray& r, interval ray_t, hit_record& rec, curandState* state) {
    return s->hit(r, ray_t, rec, state);
}

__host__ __device__ aabb uniform_scale_bounding_box(const uniform_scale* s) {
    return s->bounding_box();
}

__device__ bool transform_hit(const transform* t, const ray& r, interval ray_t, hit_record& rec, curandState* state) {
    return t->hit(r, ray_t, rec, state);
}

__host__ __device__ aabb transform_bounding_box(const transform* t) {
    return t->bounding_box();
}

#endif // TRANSFORMS_H
