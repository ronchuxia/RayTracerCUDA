#ifndef SCENE_GLASS_H
#define SCENE_GLASS_H

#include <chrono>
#include <iostream>
#include <vector>

#include "camera.h"
#include "scenes/scene_utils.h"

inline void glass() {
    auto start = std::chrono::system_clock::now();
    std::clog << "Creating Scene.\n" << std::flush;

    // world
    world* w;
    checkCudaErrors(cudaMallocManaged((void**)&w, sizeof(world)));
    new(w) world();

    std::vector<void*> allocs;

    material* ground = new_lambertian(
        make_checker(0.32, color(.2, .3, .1), color(.9, .9, .9)), allocs);

    // four glass spheres
    material* clear = new_dielectric(1.5, allocs);
    material* red   = new_tinted_glass(1.5, color(0.1, 2.0, 2.0), allocs);
    material* green = new_tinted_glass(1.5, color(2.0, 0.2, 2.0), allocs);
    material* blue  = new_tinted_glass(1.5, color(2.0, 2.0, 0.1), allocs);

    material* back_a = new_lambertian(color(0.9, 0.7, 0.2), allocs);
    material* back_b = new_lambertian(color(0.7, 0.2, 0.6), allocs);

    material* light = new_diffuse_light(color(10, 10, 10), allocs);

    add_sphere(w, point3(0, -1000, 0), 1000, ground, allocs);

    add_sphere(w, point3(-4.5, 1, 0), 1.0, clear, allocs);
    add_sphere(w, point3(-1.5, 1, 0), 1.0, red,   allocs);
    add_sphere(w, point3( 1.5, 1, 0), 1.0, green, allocs);
    add_sphere(w, point3( 4.5, 1, 0), 1.0, blue,  allocs);

    add_sphere(w, point3(-2.5, 1, -4), 1.0, back_a, allocs);
    add_sphere(w, point3( 2.5, 1, -4), 1.0, back_b, allocs);

    add_sphere(w, point3(0, 40, -10), 12.0, light, allocs);

    // the BVH over the instances
    w->build();

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
    cam->render(*w);

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::clog << "Completed. Total time: " << duration.count() << "ms.\n" << std::flush;

    auto render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - render_start);
    std::clog << "Render time: " << render_duration.count() << "ms.\n" << std::flush;

    // clean up
    for (void* p : allocs)
        cudaFree(p);
    w->~world();
    cudaFree(w);
    cudaFree(cam);
}

#endif // SCENE_GLASS_H
