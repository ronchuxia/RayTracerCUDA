#ifndef VIEWER_PRIMARY_HITS_H
#define VIEWER_PRIMARY_HITS_H

#include "cuda_helper.h"
#include "vec3.h"
#include "color.h"
#include "camera.h"

struct primary_hits {
    point3* p = nullptr;          // virtual hit point of the delta chain
    point3* hit = nullptr;        // non-delta surface point of the delta chain
    int*    id = nullptr;         // scene_id, -1 on miss
    vec3*   normal = nullptr;
    color*  diffuse = nullptr;    // diffuse albedo
    color*  f0 = nullptr;         // specular f0
    real*   roughness = nullptr;
    real*   depth = nullptr;      // axial distance
    vec3*   xform = nullptr;      // transform of the delta chain
    real*   spec_dist = nullptr;  // specular distance

    size_t  pixels = 0;

    void allocate(int w, int h) {
        release();
        pixels = (size_t)w * h;
        checkCudaErrors(cudaMalloc(&p,         pixels * sizeof(point3)));
        checkCudaErrors(cudaMalloc(&id,        pixels * sizeof(int)));
        checkCudaErrors(cudaMalloc(&normal,    pixels * sizeof(vec3)));
        checkCudaErrors(cudaMalloc(&diffuse,   pixels * sizeof(color)));
        checkCudaErrors(cudaMalloc(&f0,        pixels * sizeof(color)));
        checkCudaErrors(cudaMalloc(&roughness, pixels * sizeof(real)));
        checkCudaErrors(cudaMalloc(&depth,     pixels * sizeof(real)));
        checkCudaErrors(cudaMalloc(&spec_dist, pixels * sizeof(real)));
        checkCudaErrors(cudaMalloc(&hit,       pixels * sizeof(point3)));
        checkCudaErrors(cudaMalloc(&xform,     pixels * 3 * sizeof(vec3)));
    }
    void release() {
        cudaFree(p); cudaFree(id); cudaFree(normal); cudaFree(diffuse); cudaFree(f0); cudaFree(roughness); cudaFree(depth); cudaFree(spec_dist); cudaFree(hit); cudaFree(xform);
        p = nullptr; id = nullptr; normal = nullptr; diffuse = nullptr; f0 = nullptr; roughness = nullptr; depth = nullptr; spec_dist = nullptr; hit = nullptr; xform = nullptr; pixels = 0;
    }
};

static constexpr real delta_roughness = real(0.08);

__device__ inline void hit_through_pixel(const camera& cam, int i, int j, const world& w, curandState* state, bool psr,
                                         primary_hits ph, int idx) {
    ray r = cam.get_ray_through_pixel(i, j);
    point3 origin = r.origin();
    vec3 dir0 = unit_vector(r.direction());

    vec3 X[3] = { vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1) };    // accumulated transform of the delta chain
    real len = 0;                                                   // accumulated path length of the delta chain
    int k = 0;                                                      // accumulated number of hits of the delta chain

    hit_record rec;
    bool hit;
    for (;; k++) {
        hit = w.hit(r, interval(real(0.001), infinity), rec, state);
        if (!hit) break;

        len += (rec.p - r.origin()).length();
        
        const material& m = *rec.mat;
        bool delta = m.type == DIELECTRIC || (m.type == METAL && m.roughness() < delta_roughness);

        if (!psr || !delta || k + 1 >= cam.max_depth) break;

        vec3 unit_direction = unit_vector(r.direction());
        vec3 direction;
        bool refracted;
        if (m.type == METAL) {
            direction = reflect(unit_direction, rec.normal);
            refracted = false;
        }
        else {
            real refraction_ratio = rec.front_face ? (real(1.0)/m.die.ir) : m.die.ir;

            real cos_theta = fmin(dot(-unit_direction, rec.normal), real(1.0));
            real sin_theta = sqrt(real(1.0) - cos_theta*cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > real(1.0);

            if (cannot_refract || dielectric::reflectance(cos_theta, refraction_ratio) > real(0.5)) {
                direction = reflect(unit_direction, rec.normal);
                refracted = false;
            }
            else {
                direction = refract(unit_direction, rec.normal, refraction_ratio);
                refracted = true;
            }
        }

        if (!refracted) 
            for (vec3& x : X) x -= real(2) * dot(x, rec.normal) * rec.normal;
        else {
            vec3 axis = cross(direction, unit_direction); 
            real s = axis.length();
            real c = dot(direction, unit_direction);
            if (s > 0) { 
                axis /= s; 
                for (vec3& x : X) x = x * c - cross(axis, x) * s + axis * dot(axis, x) * (real(1) - c);
            }
        }

        r = ray(rec.p, direction);
    }

    if (hit) {
        ph.p[idx]         = k > 0 ? origin + dir0 * len : rec.p;
        ph.hit[idx]       = rec.p;
        ph.id[idx]        = rec.id;
        ph.normal[idx]    = vec3(dot(X[0], rec.normal), dot(X[1], rec.normal), dot(X[2], rec.normal));
        ph.diffuse[idx]   = rec.mat->diffuse_albedo(rec);
        ph.f0[idx]        = rec.mat->specular_f0();
        ph.roughness[idx] = rec.mat->roughness();
        ph.depth[idx]     = dot(ph.p[idx] - cam.center, -cam.w);
        // specular distance requires a second hit
        if (psr && k == 0 && (rec.mat->type == METAL || rec.mat->type == DIELECTRIC)) {
            r = ray(rec.p, reflect(unit_vector(r.direction()), rec.normal));
            hit_record rec1;
            ph.spec_dist[idx] = w.hit(r, interval(real(0.001), infinity), rec1, state) ? (rec1.p - rec.p).length() : real(1e4);
        } else{
            ph.spec_dist[idx] = 0;
        }
    } else {
        ph.p[idx]         = origin + dir0 * real(1e4);
        ph.hit[idx]       = origin + dir0 * real(1e4);
        ph.id[idx]        = -1;
        ph.normal[idx]    = vec3(0, 0, 0);
        ph.diffuse[idx]   = color(0, 0, 0);
        ph.f0[idx]        = color(0, 0, 0);
        ph.roughness[idx] = 1;
        ph.depth[idx]     = infinity;
        ph.spec_dist[idx] = 0;
    }
    for (int row = 0; row < 3; row++) ph.xform[3 * idx + row] = X[row];
}

#endif // VIEWER_PRIMARY_HITS_H
