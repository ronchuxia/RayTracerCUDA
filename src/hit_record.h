#ifndef HIT_RECORD_H
#define HIT_RECORD_H

#include "vec3.h"
#include "ray.h"

struct material;

struct hit_record {
    point3 p;
    vec3 normal;
    material* mat;
    double t;
    double u, v;   // surface coordinates of the hit, set by each primitive's hit()
    bool front_face;

    __host__ __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

#endif