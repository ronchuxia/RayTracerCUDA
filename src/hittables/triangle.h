#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "aabb.h"
#include "hit_record.h"
#include "ray.h"
#include "vec3.h"

struct material;

// Triangle with an explicit outward normal (assumed unit length — STL files
// and hand-built scenes supply it; pass unit_vector(cross(v1-v0, v2-v0)) if
// deriving it from the vertices).
struct triangle {
  point3 v0, v1, v2;
  vec3 n;
  material* mat;
  aabb bbox;

  triangle(point3 _v0, point3 _v1, point3 _v2, vec3 _n, material* _material)
  : v0(_v0), v1(_v1), v2(_v2), n(_n), mat(_material) {
    auto xmin = fmin(fmin(v0.x(), v1.x()), v2.x());
    auto xmax = fmax(fmax(v0.x(), v1.x()), v2.x());
    auto ymin = fmin(fmin(v0.y(), v1.y()), v2.y());
    auto ymax = fmax(fmax(v0.y(), v1.y()), v2.y());
    auto zmin = fmin(fmin(v0.z(), v1.z()), v2.z());
    auto zmax = fmax(fmax(v0.z(), v1.z()), v2.z());
    bbox = aabb(point3(xmin, ymin, zmin), point3(xmax, ymax, zmax)).pad();
  }

  __host__ __device__ aabb bounding_box() const {return bbox;}

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const {
    if (!bbox.hit(r, ray_t)) return false;

    // Möller–Trumbore intersection.
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec3 s = r.origin() - v0;
    vec3 s1 = cross(r.direction(), e2);
    vec3 s2 = cross(s, e1);

    auto scale = dot(s1, e1);

    auto t = dot(s2, e2) / scale;
    auto b1 = dot(s1, s) / scale;
    auto b2 = dot(s2, r.direction()) / scale;

    if (!ray_t.surrounds(t)) return false;
    if (b1 < 0 || b1 > 1) return false;
    if (b2 < 0 || b2 > 1) return false;
    if (1 - b1 - b2 < 0) return false;

    rec.t = t;
    rec.p = r.at(rec.t);
    rec.set_face_normal(r, n);
    rec.mat = mat;

    return true;
  }
};

#endif
