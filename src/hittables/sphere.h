#ifndef SPHERE_H
#define SPHERE_H

#include "aabb.h"
#include "hit_record.h"
#include "ray.h"
#include "vec3.h"

struct material;

struct sphere {
  point3 center;
  double radius;
  material* mat;
  aabb bbox;

  sphere(point3 _center, double _radius, material* _material)
  : center(_center), radius(_radius), mat(_material) {
    auto rvec = vec3(radius, radius, radius);
    bbox = aabb(center - rvec, center + rvec);
  }

  __host__ __device__ aabb bounding_box() const {return bbox;}

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
    if (!bbox.hit(r, ray_t)) return false;

    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = half_b*half_b - a*c;
    if (discriminant < 0)
        return false;

    // Find the nearest root that lies in the acceptable range.
    auto sqrtd = sqrt(discriminant);
    auto root = (-half_b - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (-half_b + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat = mat;

    return true;
  }

  // Spherical (u,v) of a point on the unit sphere: u from the angle around the
  // Y axis (from x=-1), v from the angle from y=-1 up to y=+1.
  __device__ static void get_sphere_uv(const point3& p, double& u, double& v) {
    const double pi = 3.1415926535897932385;
    auto theta = acos(-p.y());
    auto phi = atan2(-p.z(), p.x()) + pi;

    u = phi / (2*pi);
    v = theta / pi;
  }
};

#endif