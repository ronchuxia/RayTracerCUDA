#include <iostream>
#include <chrono>
#include "camera.h"
#include "hittable.h"
#include "sphere.h"
#include "material.h"
#include "color.h"
#include "ray.h"
#include "vec3.h"
#include "cuda_helper.h"

void scene() {
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
    
    // ground
    material* ground_material;
    checkCudaErrors(cudaMallocManaged((void**)&ground_material, sizeof(material)));
    ground_material->type = LAMBERTIAN;
    ground_material->lam = lambertian(color(0.5, 0.5, 0.5));

    sphere* ground_sphere;
    checkCudaErrors(cudaMallocManaged((void**)&ground_sphere, sizeof(sphere)));
    new(ground_sphere) sphere(point3(0,-1000,0), 1000, ground_material);

    hittable* ground_sphere_hittable;
    checkCudaErrors(cudaMallocManaged((void**)&ground_sphere_hittable, sizeof(hittable)));
    ground_sphere_hittable->type = SPHERE;
    ground_sphere_hittable->object = ground_sphere;

    world->add(ground_sphere_hittable);

    // sphere1
    material* sphere1_material;
    checkCudaErrors(cudaMallocManaged((void**)&sphere1_material, sizeof(material)));
    sphere1_material->type = LAMBERTIAN;
    sphere1_material->lam = lambertian(color(0.7, 0.3, 0.3));

    sphere* sphere1;
    checkCudaErrors(cudaMallocManaged((void**)&sphere1, sizeof(sphere)));
    new(sphere1) sphere(point3(0,1,0), 1.0, sphere1_material);

    hittable* sphere1_hittable;
    checkCudaErrors(cudaMallocManaged((void**)&sphere1_hittable, sizeof(hittable)));
    sphere1_hittable->type = SPHERE;
    sphere1_hittable->object = sphere1;

    world->add(sphere1_hittable);

    // diffuse_light
    material* diffuse_light_material;
    checkCudaErrors(cudaMallocManaged((void**)&diffuse_light_material, sizeof(material)));
    diffuse_light_material->type = DIFFUSE_LIGHT;
    diffuse_light_material->light = diffuse_light(color(10, 10, 10));

    sphere* diffuse_light_sphere;
    checkCudaErrors(cudaMallocManaged((void**)&diffuse_light_sphere, sizeof(sphere)));
    new(diffuse_light_sphere) sphere(point3(0, 30, -30), 10.0, diffuse_light_material);

    hittable* diffuse_light_sphere_hittable;
    checkCudaErrors(cudaMallocManaged((void**)&diffuse_light_sphere_hittable, sizeof(hittable)));
    diffuse_light_sphere_hittable->type = SPHERE;
    diffuse_light_sphere_hittable->object = diffuse_light_sphere;

    world->add(diffuse_light_sphere_hittable);

    // camera
    camera* cam;
    checkCudaErrors(cudaMallocManaged((void**)&cam, sizeof(camera)));
    new(cam) camera();

    cam->aspect_ratio      = 16.0 / 9.0;
    cam->image_width       = 1200;
    cam->samples_per_pixel = 512;
    cam->max_depth         = 10;

    cam->vfov     = 20;
    cam->lookfrom = point3(13, 10, 20);
    cam->lookat   = point3(-12, -5, -20);
    cam->vup      = vec3(0,1,0);

    cam->defocus_angle = 0;
    cam->focus_dist    = 100.0;

    std::clog << "Rendering.\n" << std::flush;

    auto render_start = std::chrono::system_clock::now();
    cam->render(*world_hittable);

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::clog << "Completed. Total time: " << duration.count() << "ms.\n" << std::flush;

    auto render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - render_start);
    std::clog << "Render time: " << render_duration.count() << "ms.\n" << std::flush;

    // clean up
    cudaFree(ground_material);
    cudaFree(ground_sphere);
    cudaFree(ground_sphere_hittable);
    cudaFree(sphere1_material);
    cudaFree(sphere1);
    cudaFree(sphere1_hittable);
    cudaFree(diffuse_light_material);
    cudaFree(diffuse_light_sphere);
    cudaFree(diffuse_light_sphere_hittable);
    cudaFree(world);
    cudaFree(world_hittable);
    cudaFree(cam);
}

int main() {
    scene();
    return 0;
}