// Scene-id foundation test (mutable scene + stable ids):
//  1. ids registered via scene::add() are stamped into hit_record.id by the
//     OUTERMOST tagged wrapper — a box hit through translate(rotate_y(box))
//     reports the transform chain's id, not an interior quad;
//  2. flat-list and BVH traversal stamp the same ids;
//  3. the mutate -> restore-bbox -> refit() -> re-pick loop works: after
//     moving a sphere through its id, picking finds it at the new position
//     and misses at the old one. This is the exact loop B4 picking and
//     workstream C (dynamic scenes) build on.
#include <cstdio>
#include <vector>

#include "scene.h"
#include "scenes/scene_utils.h"

__global__ void pick(const hittable* root, const ray* rays, int n, int* out_ids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    hit_record rec;
    out_ids[i] = root->hit(rays[i], interval(real(0.001), infinity), rec, nullptr)
               ? rec.id : -1;
}

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); return 1; } \
    printf("ok: %s\n", msg); } while (0)

int main() {
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

    scene sc;
    sc.init();

    material* m = new_lambertian(color(0.5, 0.5, 0.5), sc.allocs);

    int id_ground = sc.add(make_sphere(point3(0, -1000, 0), 1000, m, sc.allocs));
    int id_ball   = sc.add(make_sphere(point3(0, 1, 0), 0.5, m, sc.allocs));
    hittable* box = new_box(point3(-1, 0, -1), point3(1, 2, 1), m, sc.allocs, sc.list_dtors);
    int id_box    = sc.add(new_translate(new_rotate_y(box, 45, sc.allocs),
                                         vec3(4, 0, 0), sc.allocs));
    sc.build();

    const int N = 4;
    ray* rays; int* out;
    checkCudaErrors(cudaMallocManaged(&rays, N * sizeof(ray)));
    checkCudaErrors(cudaMallocManaged(&out,  N * sizeof(int)));
    rays[0] = ray(point3(0, 1, -5),  vec3(0, 0, 1));   // -> ball
    rays[1] = ray(point3(4, 1, -5),  vec3(0, 0, 1));   // -> box, via transform chain
    rays[2] = ray(point3(10, 5, 0),  vec3(0, -1, 0));  // -> ground
    rays[3] = ray(point3(0, 50, -5), vec3(0, 0, 1));   // -> miss

    pick<<<1, N>>>(&sc.root(), rays, N, out);
    checkCudaErrors(cudaDeviceSynchronize());
    CHECK(out[0] == id_ball,   "ray at ball stamps the sphere's id");
    CHECK(out[1] == id_box,    "ray at box stamps the OUTERMOST transform id (not an inner quad)");
    CHECK(out[2] == id_ground, "ray at ground stamps the ground's id");
    CHECK(out[3] == -1,        "miss leaves id -1");

    pick<<<1, N>>>(sc.world_hittable, rays, N, out);   // flat path: same ids expected
    checkCudaErrors(cudaDeviceSynchronize());
    CHECK(out[0] == id_ball && out[1] == id_box && out[2] == id_ground && out[3] == -1,
          "flat-list traversal stamps the same ids");

    // Mutate through the registry: move the ball, restore its bbox invariant
    // (placement-new recomputes it), refit the BVH, and pick again.
    sphere* s = static_cast<sphere*>(sc.get(id_ball)->object);
    new(s) sphere(point3(2, 1, 0), 0.5, m);
    sc.refit();

    rays[0] = ray(point3(0, 1, -5), vec3(0, 0, 1));    // old spot -> now empty
    rays[1] = ray(point3(2, 1, -5), vec3(0, 0, 1));    // new spot -> ball
    pick<<<1, 2>>>(&sc.root(), rays, 2, out);
    checkCudaErrors(cudaDeviceSynchronize());
    CHECK(out[0] == -1,      "old position misses after the move");
    CHECK(out[1] == id_ball, "moved ball is picked at its new position after refit()");

    cudaFree(rays);
    cudaFree(out);
    sc.release();
    printf("ALL ID TESTS PASSED\n");
    return 0;
}
