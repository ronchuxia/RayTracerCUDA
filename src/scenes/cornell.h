#ifndef SCENE_CORNELL_H
#define SCENE_CORNELL_H

#include <chrono>
#include <iostream>
#include <vector>

#include "camera.h"
#include "scenes/scene_utils.h"

// Cornell box (Phase 2 validation for quads, triangle, and the box() helper):
// colored walls, a ceiling area light, two axis-aligned boxes, and a triangle
// on the back wall. Compile-time knobs come from main.cu, included first.
inline void cornell_box() {
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

    material* red   = new_lambertian(color(.65, .05, .05), allocs);
    material* white = new_lambertian(color(.73, .73, .73), allocs);
    material* green = new_lambertian(color(.12, .45, .15), allocs);
    material* blue  = new_lambertian(color(.20, .30, .70), allocs);
    material* light = new_diffuse_light(color(15, 15, 15), allocs);

    // walls + light (quad positions from the reference cornell_box)
    add_quad(world, point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green, allocs);
    add_quad(world, point3(0,0,0),   vec3(0,555,0), vec3(0,0,555), red,   allocs);
    add_quad(world, point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), light, allocs);
    add_quad(world, point3(0,0,0),       vec3(555,0,0),  vec3(0,0,555),  white, allocs);
    add_quad(world, point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), white, allocs);
    add_quad(world, point3(0,0,555),     vec3(555,0,0),  vec3(0,555,0),  white, allocs);

    // two boxes (axis-aligned until Phase 3 adds rotation)
    box(point3(265,0,295), point3(430,330,460), white, world, allocs);  // tall
    box(point3(130,0,65),  point3(295,165,230), white, world, allocs);  // short

    // a triangle on the back wall, exercising the TRIANGLE dispatch path
    add_triangle(world, point3(160,340,554), point3(395,340,554), point3(277.5,470,554),
                 vec3(0,0,-1), blue, allocs);

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

    cam->aspect_ratio      = 1.0;
    cam->image_width       = RT_IMAGE_WIDTH;
    cam->samples_per_pixel = RT_SAMPLES;
    cam->max_depth         = 10;
#ifdef RT_SEED
    cam->seed = RT_SEED;
#endif

    cam->vfov     = 40;
    cam->lookfrom = point3(278, 278, -800);
    cam->lookat   = point3(278, 278, 0);
    cam->vup      = vec3(0,1,0);

    cam->defocus_angle = 0;
    cam->focus_dist    = 10.0;

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

#endif // SCENE_CORNELL_H
