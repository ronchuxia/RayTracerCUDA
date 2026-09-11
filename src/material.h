#ifndef MATERIAL_H
#define MATERIAL_H

#include "hit_record.h"
#include "vec3.h"
#include "ray.h"
#include "color.h"
#include "texture.h"
#include "curand_kernel.h"

enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT,
    ISOTROPIC
};

struct lambertian {
  texture albedo;

  lambertian(const color& a) : albedo(a) {}
  lambertian(const texture& t) : albedo(t) {}

  __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const {
    auto scatter_direction = rec.normal + random_unit_vector(state);

    // Catch degenerate scatter direction
    if (scatter_direction.near_zero())
        scatter_direction = rec.normal;

    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo.value(rec.u, rec.v, rec.p);
    return true;
  }
};

struct metal{
  color albedo;
  real fuzz;

  metal(const color& a, real f) : albedo(a), fuzz(f < 1 ? f : 1) {}

  __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(state));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }
};

struct dielectric {
  real   ir;          // Index of refraction
  color  absorption;  // Beer-Lambert coefficient per RGB channel

  dielectric(real index_of_refraction)
    : ir(index_of_refraction), absorption(0, 0, 0) {}
  dielectric(real index_of_refraction, const color& absorb)
    : ir(index_of_refraction), absorption(absorb) {}

  __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const {
    if (!rec.front_face)
        attenuation = color(exp(-absorption.x() * rec.t),
                            exp(-absorption.y() * rec.t),
                            exp(-absorption.z() * rec.t));
    else
        attenuation = color(1.0, 1.0, 1.0);
    real refraction_ratio = rec.front_face ? (real(1.0)/ir) : ir;

    vec3 unit_direction = unit_vector(r_in.direction());
    real cos_theta = fmin(dot(-unit_direction, rec.normal), real(1.0));
    real sin_theta = sqrt(real(1.0) - cos_theta*cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > real(1.0);
    vec3 direction;

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_real(state))
        direction = reflect(unit_direction, rec.normal);
    else
        direction = refract(unit_direction, rec.normal, refraction_ratio);

    scattered = ray(rec.p, direction);
    return true;
  }

  __host__ __device__ static real reflectance(real cosine, real ref_idx) {
      // Use Schlick's approximation for reflectance.
      auto r0 = (1-ref_idx) / (1+ref_idx);
      r0 = r0*r0;
      return r0 + (1-r0)*pow((real(1.0) - cosine), real(5.0));
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

struct isotropic {
  texture albedo;

  isotropic(const color& a) : albedo(a) {}
  isotropic(const texture& t) : albedo(t) {}

  __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const {
    scattered = ray(rec.p, random_unit_vector(state));
    attenuation = albedo.value(rec.u, rec.v, rec.p);
    return true;
  }
};

struct material {
    MaterialType type;
    union {
      lambertian lam;
      metal met;
      dielectric die;
      diffuse_light light;
      isotropic iso;
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
          case ISOTROPIC:
              return iso.scatter(r_in, rec, attenuation, scattered, state);
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

    __device__ color diffuse_albedo(const hit_record& rec) const {
      switch (type) {
          case LAMBERTIAN: return lam.albedo.value(rec.u, rec.v, rec.p);
          case ISOTROPIC:  return iso.albedo.value(rec.u, rec.v, rec.p);
          default:         return color(0, 0, 0);
      }
    }

    __host__ __device__ color specular_f0() const {
      switch (type) {
          case METAL:      return met.albedo;
          case DIELECTRIC: { real r0 = (1 - die.ir) / (1 + die.ir); return color(r0*r0, r0*r0, r0*r0); }
          default:         return color(0, 0, 0);
      }
    }
    
    __host__ __device__ real roughness() const {
      switch (type) {
          case METAL:      return met.fuzz;
          case DIELECTRIC: return 0;
          default:         return 1;
      }
    }
};

#endif // MATERIAL_H