#ifndef SCENE_BADGE_H
#define SCENE_BADGE_H

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "camera.h"
#include "scenes/scene_utils.h"
#include "loaders/stl_loader.h"

// Badge scene (Phase 4 parity milestone): the fork's sphere scene — ground,
// a field of random small spheres, three big spheres (glass / diffuse /
// mirror), an emissive sphere light — plus the school-badge STL mesh, loaded
// into a contiguous triangle array, BVH'd, and placed with transforms exactly
// like the fork (uniform_scale 0.033, translate(-3, 0, -5)).

// Badge-scene knobs (the shared RT_* knobs live in main.cu):
#ifndef RT_BADGE_STL
#define RT_BADGE_STL "references/RayTracing/src/InOneWeekend/school_badge2.STL"
#endif
#ifndef RT_BADGE_FIELD
#define RT_BADGE_FIELD 1   // 1: include the random sphere field; 0: skip it
                           // (tests use 0 so the flat-list render path stays fast)
#endif

inline void badge() {
    auto start = std::chrono::system_clock::now();
    std::clog << "Creating Scene.\n" << std::flush;

    // Increase CUDA stack size to prevent stack overflow. The badge is a
    // nested BVH behind transforms (world BVH → translate → scale → mesh BVH
    // → triangle), the deepest dispatch chain of the three scenes.
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

    // world
    hittable_list* world;
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(hittable_list)));
    new(world) hittable_list();

    hittable* world_hittable;
    checkCudaErrors(cudaMallocManaged((void**)&world_hittable, sizeof(hittable)));
    world_hittable->type = HITTABLE_LIST;
    world_hittable->object = world;

    std::vector<void*> allocs;
    std::vector<bvh_scene*> mesh_bvhs;   // mesh BVHs; dtors run at teardown

    // ground
    material* ground = new_lambertian(color(0.5, 0.5, 0.5), allocs);
    add_sphere(world, point3(0, -1000, 0), 1000, ground, allocs);

#if RT_BADGE_FIELD
    // Random sphere field, following the fork's loop. The fork uses the book's
    // rand()-based random_double(), so the exact layout differs; a fixed-seed
    // mt19937 keeps OUR layout deterministic (byte-identical test renders).
    {
        std::mt19937 rng(20240713);
        std::uniform_real_distribution<double> rd(0.0, 1.0);

        for (int a = -51; a < 11; a++) {
            for (int b = -51; b < 11; b++) {
                auto choose_mat = rd(rng);
                point3 center(a + 0.9*rd(rng), 0.2, b + 0.9*rd(rng));

                if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = color(rd(rng)*rd(rng), rd(rng)*rd(rng), rd(rng)*rd(rng));
                        add_sphere(world, center, 0.2, new_lambertian(albedo, allocs), allocs);
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = color(0.5 + 0.5*rd(rng), 0.5 + 0.5*rd(rng), 0.5 + 0.5*rd(rng));
                        auto fuzz = 0.5*rd(rng);
                        add_sphere(world, center, 0.2, new_metal(albedo, fuzz, allocs), allocs);
                    } else {
                        // glass
                        add_sphere(world, center, 0.2, new_dielectric(1.5, allocs), allocs);
                    }
                }
            }
        }
        std::clog << "Number of spheres: " << world->size << "\n" << std::flush;
    }
#endif

    // the three big spheres
    add_sphere(world, point3(0, 1, 0),  1.0, new_dielectric(1.5, allocs), allocs);
    add_sphere(world, point3(-4, 1, 0), 1.0, new_lambertian(color(0.4, 0.2, 0.1), allocs), allocs);
    add_sphere(world, point3(4, 1, 0),  1.0, new_metal(color(0.7, 0.6, 0.5), 0.0, allocs), allocs);

    // school-badge STL mesh: contiguous triangles + dedicated BVH, placed with
    // transforms (fork values: scale 0.033, translate(-3, 0, -5))
    material* badge_mat = new_metal(color(0.7, 0.6, 0.5), 0.5, allocs);
    hittable* badge_mesh = load_stl(RT_BADGE_STL, badge_mat, allocs, mesh_bvhs);
    badge_mesh = new_uniform_scale(badge_mesh, 0.033, allocs);
    badge_mesh = new_translate(badge_mesh, vec3(-3, 0, -5), allocs);
    world->add(badge_mesh);

    // diffuse_light
    material* light = new_diffuse_light(color(10, 10, 10), allocs);
    add_sphere(world, point3(0, 30, -30), 10.0, light, allocs);

    // bvh over the same objects
    bvh_scene* bvh;
    checkCudaErrors(cudaMallocManaged((void**)&bvh, sizeof(bvh_scene)));
    new(bvh) bvh_scene();
    for (int i = 0; i < world->size; i++)
        bvh->add(*world->objects[i]);
    bvh->build();

    hittable* bvh_hittable;
    checkCudaErrors(cudaMallocManaged((void**)&bvh_hittable, sizeof(hittable)));
    bvh_hittable->type = BVH;
    bvh_hittable->object = bvh;

    // camera (same as the fork's badge scene)
    camera* cam;
    checkCudaErrors(cudaMallocManaged((void**)&cam, sizeof(camera)));
    new(cam) camera();

    cam->aspect_ratio      = 16.0 / 9.0;
    cam->image_width       = RT_IMAGE_WIDTH;
    cam->samples_per_pixel = RT_SAMPLES;
    cam->max_depth         = RT_MAX_DEPTH;
#ifdef RT_SEED
    cam->seed = RT_SEED;
#endif

    cam->vfov     = 20;
    cam->lookfrom = point3(13, 10, 20);
    cam->lookat   = point3(-12, -5, -20);
    cam->vup      = vec3(0, 1, 0);

    cam->defocus_angle = 0;
    cam->focus_dist    = 100.0;

    std::clog << "Rendering.\n" << std::flush;

    auto render_start = std::chrono::system_clock::now();
#if USE_BVH
    cam->render(*bvh_hittable);
#else
    cam->render(*world_hittable);
#endif

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::clog << "Completed. Total time: " << duration.count() << "ms.\n" << std::flush;

    auto render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - render_start);
    std::clog << "Render time: " << render_duration.count() << "ms.\n" << std::flush;

    // clean up
    for (bvh_scene* m : mesh_bvhs)
        m->~bvh_scene();       // frees each mesh BVH's internal buffers
    for (void* p : allocs)
        cudaFree(p);
    bvh->~bvh_scene();         // frees its nodes/prim_index/prims buffers
    cudaFree(bvh);
    cudaFree(bvh_hittable);
    world->~hittable_list();   // frees its objects array
    cudaFree(world);
    cudaFree(world_hittable);
    cudaFree(cam);
}

#endif // SCENE_BADGE_H
