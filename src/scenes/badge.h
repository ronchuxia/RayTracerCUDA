#ifndef SCENE_BADGE_H
#define SCENE_BADGE_H

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "camera.h"
#include "scenes/scene_utils.h"
#include "loaders/stl_loader.h"

#ifndef RT_BADGE_STL
#define RT_BADGE_STL "assets/cmu_badge.stl"
#endif
#ifndef RT_BADGE_FIELD
#define RT_BADGE_FIELD 1
#endif

inline void badge() {
    auto start = std::chrono::system_clock::now();
    std::clog << "Creating Scene.\n" << std::flush;

    // world
    world* w;
    checkCudaErrors(cudaMallocManaged((void**)&w, sizeof(world)));
    new(w) world();

    std::vector<void*> allocs;
    std::vector<mesh*> mesh_dtors;

    // ground
    material* ground = new_lambertian(color(0.5, 0.5, 0.5), allocs);
    add_sphere(w, point3(0, -1000, 0), 1000, ground, allocs);

#if RT_BADGE_FIELD
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
                        add_sphere(w, center, 0.2, new_lambertian(albedo, allocs), allocs);
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = color(0.5 + 0.5*rd(rng), 0.5 + 0.5*rd(rng), 0.5 + 0.5*rd(rng));
                        auto fuzz = 0.5*rd(rng);
                        add_sphere(w, center, 0.2, new_metal(albedo, fuzz, allocs), allocs);
                    } else {
                        // glass
                        add_sphere(w, center, 0.2, new_dielectric(1.5, allocs), allocs);
                    }
                }
            }
        }
        std::clog << "Number of spheres: " << w->item_count << "\n" << std::flush;
    }
#endif

    // the three big spheres
    add_sphere(w, point3(0, 1, 0),  1.0, new_dielectric(1.5, allocs), allocs);
    add_sphere(w, point3(-4, 1, 0), 1.0, new_lambertian(color(0.4, 0.2, 0.1), allocs), allocs);
    add_sphere(w, point3(4, 1, 0),  1.0, new_metal(color(0.7, 0.6, 0.5), 0.0, allocs), allocs);

    // badge
    material* badge_mat = new_metal(color(0.7, 0.6, 0.5), 0.5, allocs);
    mesh* badge_mesh = load_stl(RT_BADGE_STL, badge_mat, allocs, mesh_dtors);
    w->add(make_instance(badge_mesh, vec3(-3, 0, -5), vec3(0, 0, 0), vec3(0.033, 0.033, 0.033)));

    // diffuse_light
    material* light = new_diffuse_light(color(10, 10, 10), allocs);
    add_sphere(w, point3(0, 30, -30), 10.0, light, allocs);

    // build bvh
    w->build();

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

#endif // SCENE_BADGE_H
