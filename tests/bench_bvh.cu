// Benchmark: flat hittable_list vs. flattened BVH on a dense sphere grid.
//
// Builds one scene (ground + BENCH_GRID×BENCH_GRID jittered spheres + light),
// renders it twice with the same fixed seed — once through the flat list, once
// through the BVH — and reports both render times on stderr. The PPM images go
// to stdout; redirect it away:
//
//   nvcc tests/bench_bvh.cu -o build/bench_bvh -std=c++14 -arch=sm_86 -I.
//   ./build/bench_bvh > /dev/null
//
// Knobs: -DBENCH_GRID=24 -DBENCH_WIDTH=480 -DBENCH_SPP=16

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "camera.h"
#include "hittable.h"   // also pulls in hittables/bvh.h
#include "hittables/sphere.h"
#include "material.h"
#include "cuda_helper.h"

#ifndef BENCH_GRID
#define BENCH_GRID 24    // BENCH_GRID^2 small spheres
#endif
#ifndef BENCH_WIDTH
#define BENCH_WIDTH 480
#endif
#ifndef BENCH_SPP
#define BENCH_SPP 16
#endif

material* make_lambertian(color albedo) {
    material* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(material)));
    m->type = LAMBERTIAN;
    m->lam = lambertian(albedo);
    return m;
}

hittable* make_sphere_hittable(point3 c, double r, material* m,
                               std::vector<void*>& allocations) {
    sphere* s;
    checkCudaErrors(cudaMallocManaged((void**)&s, sizeof(sphere)));
    new(s) sphere(c, r, m);
    hittable* h;
    checkCudaErrors(cudaMallocManaged((void**)&h, sizeof(hittable)));
    h->type = SPHERE;
    h->object = s;
    allocations.push_back(s);
    allocations.push_back(h);
    return h;
}

long render_ms(camera* cam, const hittable& root) {
    auto t0 = std::chrono::steady_clock::now();
    cam->render(root);
    auto t1 = std::chrono::steady_clock::now();
    return (long)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

int main() {
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 2048));

    std::vector<void*> allocations;  // everything cudaMallocManaged'd below

    // world: ground + grid of jittered spheres + one light
    hittable_list* world;
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(hittable_list)));
    new(world) hittable_list();

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> jitter(-0.3, 0.3);
    std::uniform_real_distribution<double> chan(0.1, 0.9);

    material* ground_mat = make_lambertian(color(0.5, 0.5, 0.5));
    allocations.push_back(ground_mat);
    world->add(make_sphere_hittable(point3(0, -1000, 0), 1000, ground_mat, allocations));

    for (int a = 0; a < BENCH_GRID; a++) {
        for (int b = 0; b < BENCH_GRID; b++) {
            material* m = make_lambertian(color(chan(rng), chan(rng), chan(rng)));
            allocations.push_back(m);
            point3 c(a - BENCH_GRID / 2 + jitter(rng), 0.25, b - BENCH_GRID / 2 + jitter(rng));
            world->add(make_sphere_hittable(c, 0.25, m, allocations));
        }
    }

    material* light_mat;
    checkCudaErrors(cudaMallocManaged((void**)&light_mat, sizeof(material)));
    light_mat->type = DIFFUSE_LIGHT;
    light_mat->light = diffuse_light(color(10, 10, 10));
    allocations.push_back(light_mat);
    world->add(make_sphere_hittable(point3(0, 40, 0), 15.0, light_mat, allocations));

    hittable* flat_root;
    checkCudaErrors(cudaMallocManaged((void**)&flat_root, sizeof(hittable)));
    flat_root->type = HITTABLE_LIST;
    flat_root->object = world;

    // bvh over the same objects
    bvh_scene* bvh;
    checkCudaErrors(cudaMallocManaged((void**)&bvh, sizeof(bvh_scene)));
    new(bvh) bvh_scene();
    for (int i = 0; i < world->size; i++)
        bvh->add(*world->objects[i]);

    auto build_t0 = std::chrono::steady_clock::now();
    bvh->build();
    auto build_t1 = std::chrono::steady_clock::now();
    long build_us = (long)std::chrono::duration_cast<std::chrono::microseconds>(build_t1 - build_t0).count();

    hittable* bvh_root;
    checkCudaErrors(cudaMallocManaged((void**)&bvh_root, sizeof(hittable)));
    bvh_root->type = BVH;
    bvh_root->object = bvh;

    // camera
    camera* cam;
    checkCudaErrors(cudaMallocManaged((void**)&cam, sizeof(camera)));
    new(cam) camera();
    cam->aspect_ratio      = 16.0 / 9.0;
    cam->image_width       = BENCH_WIDTH;
    cam->samples_per_pixel = BENCH_SPP;
    cam->max_depth         = 8;
    cam->vfov     = 30;
    cam->lookfrom = point3(18, 12, 18);
    cam->lookat   = point3(0, 0, 0);
    cam->vup      = vec3(0, 1, 0);
    cam->defocus_angle = 0;
    cam->focus_dist    = 30.0;
    cam->seed = 42;  // identical sampling for both renders

    int n_objects = world->size;
    std::fprintf(stderr, "bench: %d objects, %dx%d, %d spp, BVH nodes %d, build %ld us\n",
                 n_objects, cam->image_width,
                 (int)(cam->image_width / cam->aspect_ratio), BENCH_SPP,
                 bvh->node_count, build_us);

    // warm-up (context/clock ramp), then timed renders
    cam->samples_per_pixel = 1;
    cam->render(*bvh_root);
    cam->samples_per_pixel = BENCH_SPP;

    long flat_ms = render_ms(cam, *flat_root);
    long bvh_ms  = render_ms(cam, *bvh_root);

    std::fprintf(stderr, "flat list: %ld ms\n", flat_ms);
    std::fprintf(stderr, "bvh:       %ld ms\n", bvh_ms);
    std::fprintf(stderr, "speedup:   %.2fx\n", bvh_ms > 0 ? (double)flat_ms / bvh_ms : 0.0);

    // clean up
    bvh->~bvh_scene();
    cudaFree(bvh);
    cudaFree(bvh_root);
    world->~hittable_list();
    cudaFree(world);
    cudaFree(flat_root);
    cudaFree(cam);
    for (void* p : allocations) cudaFree(p);
    return 0;
}
