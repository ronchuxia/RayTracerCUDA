#ifndef SCENE_CORNELL_SMOKE_H
#define SCENE_CORNELL_SMOKE_H

#include <chrono>
#include <iostream>
#include <vector>

#include "camera.h"
#include "scenes/scene_utils.h"

// Cornell smoke (A2 validation, the book's cornell_smoke): the Cornell walls
// with a larger, dimmer area light, and the two boxes replaced by constant-
// density media — a black smoke box and a white fog box. Direct translation
// of the reference scene (its light/ceiling values differ from cornell.h).
inline void cornell_smoke() {
    auto start = std::chrono::system_clock::now();
    std::clog << "Creating Scene.\n" << std::flush;

    // Increase CUDA stack size to prevent stack overflow. The media wrap
    // transform chains (medium → translate → rotate → list → quad) and call
    // boundary->hit twice per query.
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
    material* light = new_diffuse_light(color(7, 7, 7), allocs);

    // walls + light (quad positions from the reference cornell_smoke)
    add_quad(world, point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green, allocs);
    add_quad(world, point3(0,0,0),   vec3(0,555,0), vec3(0,0,555), red,   allocs);
    add_quad(world, point3(113,554,127), vec3(330,0,0), vec3(0,0,305), light, allocs);
    add_quad(world, point3(0,555,0),     vec3(555,0,0), vec3(0,0,555), white, allocs);
    add_quad(world, point3(0,0,0),       vec3(555,0,0), vec3(0,0,555), white, allocs);
    add_quad(world, point3(0,0,555),     vec3(555,0,0), vec3(0,555,0), white, allocs);

    // the two boxes, transform-placed as in the reference, then wrapped in
    // constant-density media: dark smoke (tall) and white fog (short)
    hittable* box1 = new_box(point3(0,0,0), point3(165,330,165), white, allocs, list_dtors);
    box1 = new_rotate_y(box1, 15, allocs);
    box1 = new_translate(box1, vec3(265,0,295), allocs);
    world->add(new_constant_medium(box1, 0.01, make_solid_color(color(0,0,0)), allocs));

    hittable* box2 = new_box(point3(0,0,0), point3(165,165,165), white, allocs, list_dtors);
    box2 = new_rotate_y(box2, -18, allocs);
    box2 = new_translate(box2, vec3(130,0,65), allocs);
    world->add(new_constant_medium(box2, 0.01, make_solid_color(color(1,1,1)), allocs));

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

    // camera (same as the reference cornell_smoke)
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

#endif // SCENE_CORNELL_SMOKE_H
