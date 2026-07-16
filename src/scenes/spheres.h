#ifndef SCENE_SPHERES_H
#define SCENE_SPHERES_H

#include <chrono>
#include <iostream>
#include <vector>

#include "camera.h"
#include "scenes/scene_utils.h"
#include "loaders/image_loader.h"

// Default scene: a checkered ground, one diffuse sphere, an earth-textured
// sphere (A1 texture validation), and an emissive sphere lighting them.
// Compile-time knobs (RT_IMAGE_WIDTH, RT_SAMPLES, RT_SEED, USE_BVH) come from
// main.cu, included before this header.

#ifndef RT_EARTH_IMG
#define RT_EARTH_IMG "references/RayTracing/images/earthmap.jpg"   // root-relative
#endif
inline void spheres() {
    auto start = std::chrono::system_clock::now();
    std::clog << "Creating Scene.\n" << std::flush;

    // Increase CUDA stack size to prevent stack overflow
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 2048));

    // world
    hittable_list* world;
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(hittable_list)));
    new(world) hittable_list();

    hittable* world_hittable;
    checkCudaErrors(cudaMallocManaged((void**)&world_hittable, sizeof(hittable)));
    world_hittable->type = HITTABLE_LIST;
    world_hittable->object = world;

    std::vector<void*> allocs;

    material* ground = new_lambertian(
        make_checker(0.32, color(.2, .3, .1), color(.9, .9, .9)), allocs);
    material* red    = new_lambertian(color(0.7, 0.3, 0.3), allocs);
    material* earth  = new_lambertian(load_image_texture(RT_EARTH_IMG, allocs), allocs);
    material* light  = new_diffuse_light(color(10, 10, 10), allocs);

    add_sphere(world, point3(0, -1000, 0), 1000, ground, allocs);
    add_sphere(world, point3(0, 1, 0),     1.0,  red,    allocs);
    add_sphere(world, point3(-2.5, 1, 1),  1.0,  earth,  allocs);
    add_sphere(world, point3(0, 30, -30),  10.0, light,  allocs);

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

    // camera
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

#endif // SCENE_SPHERES_H
