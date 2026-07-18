#ifndef SCENE_CORNELL_H
#define SCENE_CORNELL_H

#include <chrono>
#include <iostream>
#include <vector>

#include "camera.h"
#include "scenes/scene_utils.h"

// Cornell box: colored walls, a ceiling area light, two boxes placed with
// instance transforms (Phase 3), and a triangle on the back wall. Compile-time
// knobs come from main.cu, included first.
inline void cornell_box() {
    auto start = std::chrono::system_clock::now();
    std::clog << "Creating Scene.\n" << std::flush;

    // Increase CUDA stack size to prevent stack overflow. Transform chains
    // deepen the dispatch recursion (translate → rotate → list → quad), so
    // this scene uses a larger margin than the sphere scene's 2048.
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

    // world
    hittable_list* world;
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(hittable_list)));
    new(world) hittable_list();

    hittable* world_hittable;
    checkCudaErrors(cudaMallocManaged((void**)&world_hittable, sizeof(hittable)));
    world_hittable->type = HITTABLE_LIST;
    world_hittable->id = -1;
    world_hittable->object = world;

    std::vector<void*> allocs;
    std::vector<hittable_list*> list_dtors;   // inner box lists; dtors run at teardown

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

    // two boxes, placed with instance transforms (reference cornell_box values):
    // built at the origin, then rotated about Y, then translated into place.
    hittable* box1 = new_box(point3(0,0,0), point3(165,330,165), white, allocs, list_dtors);
    box1 = new_rotate_y(box1, 15, allocs);
    box1 = new_translate(box1, vec3(265,0,295), allocs);
    world->add(box1);

    // The short box is built double-size and uniform-scaled by 0.5 — the net
    // geometry is identical to the reference's 165^3 box, but the chain
    // exercises all three transforms (scale, then rotate, then translate).
    hittable* box2 = new_box(point3(0,0,0), point3(330,330,330), white, allocs, list_dtors);
    box2 = new_uniform_scale(box2, 0.5, allocs);
    box2 = new_rotate_y(box2, -18, allocs);
    box2 = new_translate(box2, vec3(130,0,65), allocs);
    world->add(box2);

    // a triangle on the back wall, exercising the TRIANGLE dispatch path
    add_triangle(world, point3(160,340,554), point3(395,340,554), point3(277.5,470,554),
                 vec3(0,0,-1), blue, allocs);

    // world_bvh over the same objects
    bvh* world_bvh;
    checkCudaErrors(cudaMallocManaged((void**)&world_bvh, sizeof(bvh)));
    new(world_bvh) bvh();
    for (int i = 0; i < world->size; i++)
        world_bvh->add(*world->objects[i]);
    world_bvh->build();

    hittable* world_bvh_hittable;
    checkCudaErrors(cudaMallocManaged((void**)&world_bvh_hittable, sizeof(hittable)));
    world_bvh_hittable->type = BVH;
    world_bvh_hittable->id = -1;
    world_bvh_hittable->object = world_bvh;

    // camera
    camera* cam;
    checkCudaErrors(cudaMallocManaged((void**)&cam, sizeof(camera)));
    new(cam) camera();

    cam->aspect_ratio      = 1.0;
    cam->image_width       = RT_IMAGE_WIDTH;
    cam->samples_per_pixel = RT_SAMPLES;
    cam->max_depth         = RT_MAX_DEPTH;
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
    cam->render(*world_bvh_hittable);
#else
    cam->render(*world_hittable);
#endif

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::clog << "Completed. Total time: " << duration.count() << "ms.\n" << std::flush;

    auto render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - render_start);
    std::clog << "Render time: " << render_duration.count() << "ms.\n" << std::flush;

    // clean up
    for (hittable_list* l : list_dtors)
        l->~hittable_list();   // frees each inner box list's objects array
    for (void* p : allocs)
        cudaFree(p);
    world_bvh->~bvh();         // frees its nodes/prim_index/prims buffers
    cudaFree(world_bvh);
    cudaFree(world_bvh_hittable);
    world->~hittable_list();   // frees its objects array
    cudaFree(world);
    cudaFree(world_hittable);
    cudaFree(cam);
}

#endif // SCENE_CORNELL_H
