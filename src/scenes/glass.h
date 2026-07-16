#ifndef SCENE_GLASS_H
#define SCENE_GLASS_H

#include <chrono>
#include <iostream>
#include <vector>

#include "camera.h"
#include "scenes/scene_utils.h"

// Tinted-glass demo (beyond the fork): a row of four glass spheres over the
// checker ground — clear, red, green, blue — lit spheres-style by one emissive
// sphere (looks even nicer with -DRT_SKY=1). The colored three use the
// Beer-Lambert absorbing dielectric, so each sphere's deep core (long chord =>
// more absorption) is more saturated than its thin rim: thickness -> saturation,
// visible within a single sphere. A couple of diffuse spheres sit behind the
// row to give the glass something vivid to tint. Knobs come from main.cu.
inline void glass() {
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

    // Four glass spheres (radius 1). absorption is the per-channel coefficient of
    // what's *removed*, so the visible tint is its complement (green absorbs R+B).
    material* clear = new_dielectric(1.5, allocs);
    material* red   = new_tinted_glass(1.5, color(0.1, 2.0, 2.0), allocs);
    material* green = new_tinted_glass(1.5, color(2.0, 0.2, 2.0), allocs);
    material* blue  = new_tinted_glass(1.5, color(2.0, 2.0, 0.1), allocs);

    // Backdrop diffuse spheres — colorful things for the glass to tint.
    material* back_a = new_lambertian(color(0.9, 0.7, 0.2), allocs);
    material* back_b = new_lambertian(color(0.7, 0.2, 0.6), allocs);

    material* light = new_diffuse_light(color(10, 10, 10), allocs);

    add_sphere(world, point3(0, -1000, 0), 1000, ground, allocs);

    add_sphere(world, point3(-4.5, 1, 0), 1.0, clear, allocs);
    add_sphere(world, point3(-1.5, 1, 0), 1.0, red,   allocs);
    add_sphere(world, point3( 1.5, 1, 0), 1.0, green, allocs);
    add_sphere(world, point3( 4.5, 1, 0), 1.0, blue,  allocs);

    add_sphere(world, point3(-2.5, 1, -4), 1.0, back_a, allocs);
    add_sphere(world, point3( 2.5, 1, -4), 1.0, back_b, allocs);

    add_sphere(world, point3(0, 40, -10), 12.0, light, allocs);

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

    cam->vfov     = 30;
    cam->lookfrom = point3(0, 3, 16);
    cam->lookat   = point3(0, 1, 0);
    cam->vup      = vec3(0, 1, 0);

    cam->defocus_angle = 0;
    cam->focus_dist    = 16.0;

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

#endif // SCENE_GLASS_H
