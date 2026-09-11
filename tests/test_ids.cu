// Scene-id foundation test (mutable scene + stable ids):
//  1. ids registered via scene::add() are stamped into hit_record.id by the
//     INSTANCE — a box hit reports the instance's id, not a face of its mesh;
//  2. a single-leaf reference tree and the BVH traversal stamp the same ids;
//  3. the mutate -> restore-bbox -> refit() -> re-pick loop works: after
//     moving a sphere's instance through its id, picking finds it at the new
//     position and misses at the old one. This is the exact loop B4 picking and
//     workstream C (dynamic scenes) build on.
#include <cstdio>
#include <vector>

#include "viewer/scene.h"
#include "scenes/scene_utils.h"
#include "single_leaf.h"

__global__ void pick(const world* w, const ray* rays, int n, int* out_ids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    hit_record rec;
    out_ids[i] = w->hit(rays[i], interval(real(0.001), infinity), rec, nullptr)
               ? rec.id : -1;
}

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); return 1; } \
    printf("ok: %s\n", msg); } while (0)

int main() {
    scene sc;
    sc.init();

    material* m = new_lambertian(color(0.5, 0.5, 0.5), sc.allocs);

    const vec3 no_rot(0, 0, 0), unit(1, 1, 1);
    int id_ground = sc.add(make_instance(make_sphere(point3(0, 0, 0), 1000, m, sc.allocs),
                                         vec3(0, -1000, 0), no_rot, unit));
    int id_ball   = sc.add(make_instance(make_sphere(point3(0, 0, 0), 0.5, m, sc.allocs),
                                         vec3(0, 1, 0), no_rot, unit));
    mesh* box = new_box(point3(-1, 0, -1), point3(1, 2, 1), m, sc.allocs, sc.mesh_dtors);
    int id_box    = sc.add(make_instance(box, vec3(4, 0, 0), vec3(0, 45, 0), unit));
    sc.build();

    const int N = 4;
    ray* rays; int* out;
    checkCudaErrors(cudaMallocManaged(&rays, N * sizeof(ray)));
    checkCudaErrors(cudaMallocManaged(&out,  N * sizeof(int)));
    rays[0] = ray(point3(0, 1, -5),  vec3(0, 0, 1));   // -> ball
    rays[1] = ray(point3(4, 1, -5),  vec3(0, 0, 1));   // -> box, through its instance transform
    rays[2] = ray(point3(10, 5, 0),  vec3(0, -1, 0));  // -> ground
    rays[3] = ray(point3(0, 50, -5), vec3(0, 0, 1));   // -> miss

    world* w = &sc.root();
    pick<<<1, N>>>(w, rays, N, out);
    checkCudaErrors(cudaDeviceSynchronize());
    CHECK(out[0] == id_ball,   "ray at ball stamps the sphere's id");
    CHECK(out[1] == id_box,    "ray at box stamps the INSTANCE id (not a face of its mesh)");
    CHECK(out[2] == id_ground, "ray at ground stamps the ground's id");
    CHECK(out[3] == -1,        "miss leaves id -1");

    world* flat;                                     // single-leaf copy: the brute-force reference
    checkCudaErrors(cudaMallocManaged((void**)&flat, sizeof(world)));
    new(flat) world();
    for (int i = 0; i < w->item_count; i++) flat->add(w->items[i]);
    single_leaf(flat);
    pick<<<1, N>>>(flat, rays, N, out);
    checkCudaErrors(cudaDeviceSynchronize());
    CHECK(out[0] == id_ball && out[1] == id_box && out[2] == id_ground && out[3] == -1,
          "the single-leaf reference stamps the same ids");
    flat->~world();
    cudaFree(flat);

    // Mutate through the registry: move the ball's instance, restore its bbox
    // invariant (set_transform recomputes it), refit the BVH, and pick again.
    sc.get(id_ball)->set_transform(vec3(2, 1, 0), no_rot, unit);
    sc.refit();

    rays[0] = ray(point3(0, 1, -5), vec3(0, 0, 1));    // old spot -> now empty
    rays[1] = ray(point3(2, 1, -5), vec3(0, 0, 1));    // new spot -> ball
    pick<<<1, 2>>>(w, rays, 2, out);
    checkCudaErrors(cudaDeviceSynchronize());
    CHECK(out[0] == -1,      "old position misses after the move");
    CHECK(out[1] == id_ball, "moved ball is picked at its new position after refit()");

    cudaFree(rays);
    cudaFree(out);
    sc.release();
    printf("ALL ID TESTS PASSED\n");
    return 0;
}
