#ifndef QUAD_H
#define QUAD_H

#include "aabb.h"
#include "hit_record.h"
#include "ray.h"
#include "vec3.h"

struct material;

// Parallelogram defined by a corner Q and two edge vectors u, v.
struct quad {
  point3 Q;
  vec3 u, v;
  material* mat;
  vec3 w;       // n / dot(n,n), for planar hit-point coordinates
  vec3 normal;
  real D;     // plane equation: dot(normal, p) = D
  aabb bbox;

  quad(const point3& _Q, const vec3& _u, const vec3& _v, material* _material)
  : Q(_Q), u(_u), v(_v), mat(_material) {
    auto n = cross(u, v);
    normal = unit_vector(n);
    D = dot(normal, Q);
    w = n / dot(n, n);
    bbox = aabb(Q, Q + u + v).pad();  // pad: axis-aligned quads have a zero-thickness box
  }

  __host__ __device__ aabb bounding_box() const {return bbox;}

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
    if (!bbox.hit(r, ray_t)) return false;

    auto denom = dot(normal, r.direction());

    // No hit if the ray is parallel to the plane.
    if (fabs(denom) < real(1e-8)) return false;

    // Return false if the hit point parameter t is outside the ray interval.
    auto t = (D - dot(normal, r.origin())) / denom;
    if (!ray_t.contains(t)) return false;

    // Determine if the hit point lies within the quad using its plane coordinates.
    auto intersection = r.at(t);
    vec3 planar_hitpt_vector = intersection - Q;
    auto alpha = dot(w, cross(planar_hitpt_vector, v));
    auto beta = dot(w, cross(u, planar_hitpt_vector));

    if (alpha < 0 || 1 < alpha || beta < 0 || 1 < beta) return false;

    rec.t = t;
    rec.p = intersection;
    rec.u = alpha;   // planar coordinates real as the quad's UV
    rec.v = beta;
    rec.mat = mat;
    rec.set_face_normal(r, normal);

    return true;
  }
};

#endif
