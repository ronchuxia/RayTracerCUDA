#ifndef SCENE_CORNELL_SMOKE_H
#define SCENE_CORNELL_SMOKE_H

#include <chrono>
#include <iostream>
#include <vector>

#include "camera.h"
#include "scenes/scene_utils.h"

inline void cornell_smoke() {
    auto start = std::chrono::system_clock::now();
    std::clog << "Creating Scene.\n" << std::flush;

    // world
    world* w;
    checkCudaErrors(cudaMallocManaged((void**)&w, sizeof(world)));
    new(w) world();

    std::vector<void*> allocs;
    std::vector<mesh*> mesh_dtors;

    material* red   = new_lambertian(color(.65, .05, .05), allocs);
    material* white = new_lambertian(color(.73, .73, .73), allocs);
    material* green = new_lambertian(color(.12, .45, .15), allocs);
    material* light = new_diffuse_light(color(7, 7, 7), allocs);

    // walls + light
    add_quad(w, point3(555,0,0), vec3(0,555,0), vec3(0,0,555), green, allocs);
    add_quad(w, point3(0,0,0),   vec3(0,555,0), vec3(0,0,555), red,   allocs);
    add_quad(w, point3(113,554,127), vec3(330,0,0), vec3(0,0,305), light, allocs);
    add_quad(w, point3(0,555,0),     vec3(555,0,0), vec3(0,0,555), white, allocs);
    add_quad(w, point3(0,0,0),       vec3(555,0,0), vec3(0,0,555), white, allocs);
    add_quad(w, point3(0,0,555),     vec3(555,0,0), vec3(0,555,0), white, allocs);

    // the two boxes
    mesh* box1 = new_box(point3(0,0,0), point3(165,330,165), white, allocs, mesh_dtors);
    w->add(make_medium(box1, vec3(265,0,295), vec3(0,15,0), vec3(1,1,1), 0.01, make_solid_color(color(0,0,0)), allocs));
    mesh* box2 = new_box(point3(0,0,0), point3(165,165,165), white, allocs, mesh_dtors);
    w->add(make_medium(box2, vec3(130,0,65), vec3(0,-18,0), vec3(1,1,1), 0.01, make_solid_color(color(1,1,1)), allocs));

    // build bvh
    w->build();

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
    cam->render(*w);

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::clog << "Completed. Total time: " << duration.count() << "ms.\n" << std::flush;

    auto render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - render_start);
    std::clog << "Render time: " << render_duration.count() << "ms.\n" << std::flush;

    // clean up
    for (mesh* m : mesh_dtors)
        m->~mesh();
    for (void* p : allocs)
        cudaFree(p);
    w->~world();
    cudaFree(w);
    cudaFree(cam);
}

#endif // SCENE_CORNELL_SMOKE_H
