#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "aabb.h"
#include "hit_record.h"
#include "hittable.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

// Volume of constant density: wraps a boundary hittable* (any convex shape —
// box, sphere, transform-wrapped composite) and probabilistically scatters
// the ray INSIDE it. A ray entering the boundary samples an exponential
// free-flight distance (Beer–Lambert); if that distance falls inside the
// boundary span, the "hit" is at that interior point with the medium's phase
// function (an ISOTROPIC material). This is the one stochastic hit(): it is
// why the whole hit() chain carries a curandState*.
//
// NOTE: because hit() consumes RNG, flat-list and BVH renders of a scene with
// media are NOT byte-identical — the traversal orders consume the per-pixel
// RNG stream differently. Both are equally correct Monte Carlo estimates.
struct constant_medium {
  hittable* boundary;
  double neg_inv_density;
  material* phase_function;   // an ISOTROPIC material, owned by the scene

  constant_medium(hittable* _boundary, double density, material* _phase_function)
  : boundary(_boundary), neg_inv_density(-1/density), phase_function(_phase_function) {}

  __host__ __device__ aabb bounding_box() const {return boundary->bounding_box();}

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
    hit_record rec1, rec2;
    const double inf = 1.0/0.0;

    // Entry and exit points of the ray through the boundary, over the whole
    // line (not just ray_t), so a ray starting inside still finds its exit.
    if (!boundary->hit(r, interval(-inf, inf), rec1, state))
        return false;

    if (!boundary->hit(r, interval(rec1.t + 0.0001, inf), rec2, state))
        return false;

    if (rec1.t < ray_t.min) rec1.t = ray_t.min;
    if (rec2.t > ray_t.max) rec2.t = ray_t.max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0)
        rec1.t = 0;

    auto ray_length = r.direction().length();
    auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    // curand_uniform_double is in (0, 1], so log() is finite.
    auto hit_distance = neg_inv_density * log(curand_uniform_double(state));

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    rec.normal = vec3(1, 0, 0);   // arbitrary: isotropic scatter ignores it
    rec.front_face = true;        // also arbitrary
    rec.u = 0;                    // volumes have no surface parameterization
    rec.v = 0;
    rec.mat = phase_function;

    return true;
  }
};

// Dispatch shims declared in hittable.h (before constant_medium is complete)
// and defined here, so hittable's switches can route to it.
__device__ bool constant_medium_hit(const constant_medium* m, const ray& r, interval ray_t, hit_record& rec, curandState* state) {
    return m->hit(r, ray_t, rec, state);
}

__host__ __device__ aabb constant_medium_bounding_box(const constant_medium* m) {
    return m->bounding_box();
}

#endif // CONSTANT_MEDIUM_H
