// Reproducer: silent per-thread work loss in the recursive hittable::hit
// dispatch under whole-program device compilation (nvcc's -rdc=false default).
//
//   nvcc tests/repro_rdc_corruption.cu -o build/repro_rdc -std=c++14 -arch=sm_86 -Isrc
//   ./build/repro_rdc            # -> REPRODUCED: counters are incoherent
//
//   nvcc tests/repro_rdc_corruption.cu -o build/repro_rdc_ok -std=c++14 -arch=sm_86 -Isrc -rdc=true
//   ./build/repro_rdc_ok         # -> clean: every thread reports exactly SPP
//
// WHAT IT SHOWS
// Each thread runs a loop bounded by `spp` (a kernel parameter, uniform across
// all threads) and reports three numbers that must all equal SPP:
//   spp_seen : the bound as that thread read it
//   trips    : a REGISTER counter incremented once per iteration, stored after
//              the loop
//   nsamp    : a GLOBAL-MEMORY counter incremented inside each iteration
// Under -rdc=false a warp-aligned group of threads reports e.g.
// spp_seen=8, trips=9, nsamp=6 -- a bound-8 loop cannot run 9 times, and the
// two counters disagree with each other. The kernel still returns cudaSuccess.
//
// WHY THIS SCENE
// The trigger needs the recursive dispatch to nest (a transform wrapping a
// hittable_list of quads) AND transform-wrapped dielectric/metal materials in
// the same scene. Removing the box, or making every material lambertian, or
// leaving the glass/metal spheres unwrapped, all make it go clean. Note that
// this recipe does NOT transfer to other scenes (Cornell with the same
// ingredients added stays clean), so treat it as one known-triggering
// configuration, not a general rule for which scenes are unsafe.
//
// See docs/rdc-corruption.md for the full investigation.
#define RT_SKY 1
#include <cstdio>
#include <vector>

#include "camera.h"
#include "scene.h"
#include "scenes/scene_utils.h"

#define SPP 8
#define MAX_DEPTH 12

__global__ void trace(const camera& cam, int md, const hittable& world, color* accum,
                      curandState* rs, int spp, int* nsamp, int* trips, int* spp_seen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= cam.image_width || j >= cam.image_height) return;
    int idx = j * cam.image_width + i;

    int t = 0;
    for (int s = 0; s < spp; ++s) {
        ray r = cam.get_ray(i, j, &rs[idx]);
        accum[idx] += cam.ray_color(r, world, md, &rs[idx]);
        nsamp[idx]++;   // global-memory counter, incremented inside the loop
        t++;            // register counter
    }
    trips[idx]    = t;    // stored once, after the loop
    spp_seen[idx] = spp;  // the bound this thread actually read
}

// The triggering scene: transform-wrapped composite (box) + transform-wrapped
// dielectric/metal. This is the viewer's scene (src/viewer/viewer.cu).
static void build_repro_scene(scene& sc) {
    sc.init();
    material* ground  = new_lambertian(
        make_checker(0.32, color(.2, .3, .1), color(.9, .9, .9)), sc.allocs);
    material* diffuse = new_lambertian(color(0.4, 0.2, 0.1), sc.allocs);
    material* glass   = new_dielectric(1.5, sc.allocs);
    material* metal_m = new_metal(color(0.7, 0.6, 0.5), 0.0, sc.allocs);
    material* box_mat = new_lambertian(color(0.2, 0.4, 0.7), sc.allocs);
    material* tri_mat = new_lambertian(color(0.9, 0.75, 0.2), sc.allocs);

    sc.add(make_sphere(point3(0, -1000, 0), 1000, ground, sc.allocs));
    sc.add(new_transform(make_sphere(point3(0,0,0), 1.0, diffuse, sc.allocs),
                         vec3(-4,1,0), vec3(0,0,0), vec3(1,1,1), sc.allocs));
    sc.add(new_transform(make_sphere(point3(0,0,0), 1.0, glass, sc.allocs),
                         vec3(0,1,0), vec3(0,0,0), vec3(1,1,1), sc.allocs));
    sc.add(new_transform(make_sphere(point3(0,0,0), 1.0, metal_m, sc.allocs),
                         vec3(4,1,0), vec3(0,0,0), vec3(1,1,1), sc.allocs));
    sc.add(new_transform(new_box(point3(-0.6,-0.6,-0.6), point3(0.6,0.6,0.6),
                                 box_mat, sc.allocs, sc.list_dtors),
                         vec3(-2,0.6,-3), vec3(0,35,0), vec3(1,1,1), sc.allocs));
    sc.add(new_transform(make_triangle(point3(-0.8,-0.6,0), point3(0.8,-0.6,0),
                                       point3(0,0.9,0), vec3(0,0,1), tri_mat, sc.allocs),
                         vec3(2,1.3,-3), vec3(0,0,0), vec3(1,1,1), sc.allocs));
    sc.build();
}

int main() {
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 2048));

    scene sc;
    build_repro_scene(sc);

    camera* cam;
    checkCudaErrors(cudaMallocManaged((void**)&cam, sizeof(camera)));
    new(cam) camera();
    cam->aspect_ratio = 16.0 / 9.0;
    cam->image_width  = 200;
    cam->max_depth    = MAX_DEPTH;
    cam->seed         = 42;
    cam->vfov         = 20;
    cam->lookfrom     = point3(13, 2, 3);
    cam->lookat       = point3(0, 0, 0);
    cam->vup          = vec3(0, 1, 0);
    cam->defocus_angle = 0;
    cam->focus_dist    = 10;
    cam->initialize();

    const int W = cam->image_width, H = cam->image_height, N = W * H;
    curandState* rs; color* accum; int *nsamp, *trips, *spp_seen;
    checkCudaErrors(cudaMallocManaged(&rs,       N * sizeof(curandState)));
    checkCudaErrors(cudaMallocManaged(&accum,    N * sizeof(color)));
    checkCudaErrors(cudaMallocManaged(&nsamp,    N * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&trips,    N * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&spp_seen, N * sizeof(int)));
    checkCudaErrors(cudaMemset(accum,    0, N * sizeof(color)));
    checkCudaErrors(cudaMemset(nsamp,    0, N * sizeof(int)));
    checkCudaErrors(cudaMemset(trips,    0, N * sizeof(int)));
    checkCudaErrors(cudaMemset(spp_seen, 0, N * sizeof(int)));

    initialize_rand<<<(N + 255) / 256, 256>>>(*cam, rs, 42UL);
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 threads(16, 16), blocks((W + 15) / 16, (H + 15) / 16);
    trace<<<blocks, threads>>>(*cam, MAX_DEPTH, sc.root(), accum, rs, SPP,
                               nsamp, trips, spp_seen);
    cudaError_t launch = cudaGetLastError();
    checkCudaErrors(cudaDeviceSynchronize());

    int bad = 0, shown = 0;
    long long total = 0;
    for (int i = 0; i < N; i++) {
        total += nsamp[i];
        if (nsamp[i] != SPP || trips[i] != SPP || spp_seen[i] != SPP) bad++;
    }
    printf("launch status : %s\n", cudaGetErrorName(launch));
    printf("threads       : %d\n", N);
    printf("iterations    : %lld / %lld\n", total, (long long)N * SPP);
    printf("bad threads   : %d\n", bad);
    for (int i = 0; i < N && shown < 5; i++) {
        if (nsamp[i] != SPP || trips[i] != SPP || spp_seen[i] != SPP) {
            printf("   px(%3d,%3d): spp_seen=%d  trips=%d  nsamp=%d   <-- incoherent\n",
                   i % W, i / W, spp_seen[i], trips[i], nsamp[i]);
            shown++;
        }
    }
    printf("\n%s\n", bad ? "REPRODUCED: rebuild with -rdc=true and this goes to 0."
                         : "CLEAN: every thread ran exactly SPP iterations.");

    cudaFree(rs); cudaFree(accum); cudaFree(nsamp); cudaFree(trips);
    cudaFree(spp_seen); cudaFree(cam);
    sc.release();
    return bad ? 1 : 0;
}
