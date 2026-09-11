#ifndef SCENE_SPHERES_H
#define SCENE_SPHERES_H

#include <chrono>
#include <iostream>
#include <vector>

#include "camera.h"
#include "scenes/scene_utils.h"
#include "loaders/image_loader.h"

#ifndef RT_EARTH_IMG
#define RT_EARTH_IMG "assets/earthmap.jpg"   // root-relative
#endif

inline void spheres() {
    auto start = std::chrono::system_clock::now();
    std::clog << "Creating Scene.\n" << std::flush;

    // world
    world* w;
    checkCudaErrors(cudaMallocManaged((void**)&w, sizeof(world)));
    new(w) world();

    std::vector<void*> allocs;

    material* ground = new_lambertian(make_checker(0.32, color(.2, .3, .1), color(.9, .9, .9)), allocs);
    material* red    = new_lambertian(color(0.7, 0.3, 0.3), allocs);
    material* earth  = new_lambertian(load_image_texture(RT_EARTH_IMG, allocs), allocs);
    material* light  = new_diffuse_light(color(10, 10, 10), allocs);

    add_sphere(w, point3(0, -1000, 0), 1000, ground, allocs);
    add_sphere(w, point3(0, 1, 0),     1.0,  red,    allocs);
    add_sphere(w, point3(-2.5, 1, 1),  1.0,  earth,  allocs);
    add_sphere(w, point3(0, 30, -30),  10.0, light,  allocs);

    // build bvh
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

    cam->vfov     = 20;
    cam->lookfrom = point3(13, 10, 20);
    cam->lookat   = point3(-12, -5, -20);
    cam->vup      = vec3(0, 1, 0);

    cam->defocus_angle = 0;
    cam->focus_dist    = 100.0;

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

#endif // SCENE_SPHERES_H
