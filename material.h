#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "vec3.h"
#include "ray.h"
#include "color.h"
#include "curand_kernel.h"

enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT
};

struct lambertian {
  color albedo;

  lambertian(const color& a) : albedo(a) {}

  __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const {
    auto scatter_direction = rec.normal + random_unit_vector(state);

    // Catch degenerate scatter direction
    if (scatter_direction.near_zero())
        scatter_direction = rec.normal;

    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;
  }
};

struct metal{
  color albedo;
  double fuzz;

  metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

  __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(state));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }
};

struct dielectric {
  double ir; // Index of Refraction

  dielectric(double index_of_refraction) : ir(index_of_refraction) {}

  __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const {
    attenuation = color(1.0, 1.0, 1.0);
    double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

    vec3 unit_direction = unit_vector(r_in.direction());
    double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    vec3 direction;

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform_double(state))
        direction = reflect(unit_direction, rec.normal);
    else
        direction = refract(unit_direction, rec.normal, refraction_ratio);

    scattered = ray(rec.p, direction);
    return true;
  }

  __host__ __device__ static double reflectance(double cosine, double ref_idx) {
      // Use Schlick's approximation for reflectance.
      // Reflectance measures the probability that the ray reflects instead of refracting.
      auto r0 = (1-ref_idx) / (1+ref_idx);
      r0 = r0*r0;
      return r0 + (1-r0)*pow((1 - cosine),5);
  }
};

struct diffuse_light {
  color emit;

  diffuse_light(const color & _emit) : emit(_emit) {}

  __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const {
    return false;
  }

  __host__ __device__ color emitted() const {
    return emit;
  }
};

struct material {
    MaterialType type;
    union {
      lambertian lam;
      metal met;
      dielectric die;
      diffuse_light light;
    };

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const {
      switch (type) {
          case LAMBERTIAN:
              return lam.scatter(r_in, rec, attenuation, scattered, state);
          case METAL:
              return met.scatter(r_in, rec, attenuation, scattered, state);
          case DIELECTRIC:
              return die.scatter(r_in, rec, attenuation, scattered, state);
          case DIFFUSE_LIGHT:
              return light.scatter(r_in, rec, attenuation, scattered, state);
          default:
              return false;
      }
    }

    __host__ __device__ color emitted() const {
      if (type == DIFFUSE_LIGHT) {
          return light.emitted();
      }
      return color(0, 0, 0);
    }
};

#endif // MATERIAL_H