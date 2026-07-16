// Unit tests for the flattened BVH (bvh.h).
//
// Strategy: build the SAME set of spheres into a flat hittable_list and a
// bvh_scene, shoot a deterministic batch of rays through both via the
// hittable::hit dispatch on the device, and require exactly equal results.
// Both paths run the identical sphere::hit arithmetic on identical inputs and
// the closest hit is unique, so t / p / normal must match bit-for-bit —
// traversal order cannot change the answer, only which primitives get tested.
//
// Also checked, host-side, after every build: structural invariants
// (prim_index is a permutation, child_index > parent_index, leaves disjointly
// cover [0, N), every internal bbox is exactly the union of its children).
//
// All randomness is host-side std::mt19937 with fixed seeds — no cuRAND, so
// every run is deterministic.
//
// Build & run (from the repo root):
//   nvcc tests/test_bvh.cu -o build/test_bvh -std=c++14 -arch=sm_86 -I. && ./build/test_bvh

#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

#include "hittable.h"   // also pulls in hittables/bvh.h
#include "hittables/sphere.h"
#include "material.h"
#include "cuda_helper.h"

static int g_failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        g_failures++; \
        std::printf("    FAIL %s:%d: ", __FILE__, __LINE__); \
        std::printf(__VA_ARGS__); \
        std::printf("\n"); \
    } \
} while (0)

// ---------------------------------------------------------------- helpers --

material* make_material() {
    material* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(material)));
    m->type = LAMBERTIAN;
    m->lam = lambertian(color(0.5, 0.5, 0.5));
    return m;
}

sphere* make_sphere(point3 c, double r, material* m) {
    sphere* s;
    checkCudaErrors(cudaMallocManaged((void**)&s, sizeof(sphere)));
    new(s) sphere(c, r, m);
    return s;
}

hittable* wrap_sphere(sphere* s) {
    hittable* h;
    checkCudaErrors(cudaMallocManaged((void**)&h, sizeof(hittable)));
    h->type = SPHERE;
    h->object = s;
    return h;
}

// A flat list and a BVH over the same sphere objects.
struct test_scene {
    std::vector<material*> materials;
    std::vector<sphere*>   spheres;
    std::vector<hittable*> wrappers;

    hittable_list* list;
    hittable* list_root;
    bvh_scene* bvh;
    hittable* bvh_root;

    void add_sphere(point3 c, double r, material* m) {
        sphere* s = make_sphere(c, r, m);
        spheres.push_back(s);
        wrappers.push_back(wrap_sphere(s));
    }

    // Build list + BVH from the spheres added so far.
    void build_structures() {
        checkCudaErrors(cudaMallocManaged((void**)&list, sizeof(hittable_list)));
        new(list) hittable_list();
        for (auto* w : wrappers) list->add(w);
        checkCudaErrors(cudaMallocManaged((void**)&list_root, sizeof(hittable)));
        list_root->type = HITTABLE_LIST;
        list_root->object = list;

        checkCudaErrors(cudaMallocManaged((void**)&bvh, sizeof(bvh_scene)));
        new(bvh) bvh_scene();
        for (auto* w : wrappers) bvh->add(*w);
        bvh->build();
        checkCudaErrors(cudaMallocManaged((void**)&bvh_root, sizeof(hittable)));
        bvh_root->type = BVH;
        bvh_root->object = bvh;
    }

    // Rebuild ONLY the flat list (fresh bbox union) — used as ground truth
    // after spheres move, since the old list's bbox would be stale.
    void rebuild_list() {
        list->~hittable_list();
        new(list) hittable_list();
        for (auto* w : wrappers) list->add(w);
    }

    void destroy() {
        list->~hittable_list();
        cudaFree(list);
        cudaFree(list_root);
        bvh->~bvh_scene();
        cudaFree(bvh);
        cudaFree(bvh_root);
        for (auto* w : wrappers)  cudaFree(w);
        for (auto* s : spheres)   cudaFree(s);
        for (auto* m : materials) cudaFree(m);
    }
};

test_scene make_random_scene(int n, unsigned rng_seed, double spread = 20.0) {
    test_scene sc;
    std::mt19937 rng(rng_seed);
    std::uniform_real_distribution<double> pos(-spread, spread);
    std::uniform_real_distribution<double> rad(0.1, 1.5);
    sc.materials.push_back(make_material());
    for (int i = 0; i < n; i++)
        sc.add_sphere(point3(pos(rng), pos(rng), pos(rng)), rad(rng), sc.materials[0]);
    sc.build_structures();
    return sc;
}

// Rays: half aimed at random sphere centers with jitter (guarantees a high
// hit rate), half fully random (exercises misses and grazing hits).
std::vector<ray> make_rays(const test_scene& sc, int n, unsigned rng_seed, double spread = 20.0) {
    std::mt19937 rng(rng_seed);
    std::uniform_real_distribution<double> pos(-2.5 * spread, 2.5 * spread);
    std::uniform_real_distribution<double> jit(-1.0, 1.0);
    std::vector<ray> rays;
    rays.reserve(n);
    for (int i = 0; i < n; i++) {
        point3 origin(pos(rng), pos(rng), pos(rng));
        vec3 dir;
        if (!sc.spheres.empty() && i % 2 == 0) {
            const sphere* target = sc.spheres[rng() % sc.spheres.size()];
            dir = (target->center + vec3(jit(rng), jit(rng), jit(rng))) - origin;
        } else {
            dir = vec3(jit(rng), jit(rng), jit(rng));
        }
        rays.push_back(ray(origin, dir));
    }
    return rays;
}

// ------------------------------------------------------------ hit kernel --

struct hit_result {
    int hit;
    double t;
    point3 p;
    vec3 normal;
    int front_face;
    material* mat;
};

__global__ void hit_kernel(const hittable* root, const ray* rays, int n, hit_result* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    hit_record rec;
    bool h = root->hit(rays[idx], interval(0.001, 1.0/0.0), rec, nullptr);
    out[idx].hit = h ? 1 : 0;
    out[idx].t = h ? rec.t : 0.0;
    out[idx].p = h ? rec.p : point3(0,0,0);
    out[idx].normal = h ? rec.normal : vec3(0,0,0);
    out[idx].front_face = h ? (rec.front_face ? 1 : 0) : 0;
    out[idx].mat = h ? rec.mat : nullptr;
}

hit_result* run_hits(const hittable* root, const ray* rays_dev, int n) {
    hit_result* out;
    checkCudaErrors(cudaMallocManaged((void**)&out, n * sizeof(hit_result)));
    hit_kernel<<<(n + 255) / 256, 256>>>(root, rays_dev, n, out);
    checkCudaErrors(cudaDeviceSynchronize());
    return out;
}

// Shoot the rays through both structures and require exactly equal results.
// Returns the number of rays that hit (to sanity-check test coverage).
int compare_hits(const test_scene& sc, const std::vector<ray>& rays, bool compare_mat = true) {
    int n = (int)rays.size();
    ray* rays_dev;
    checkCudaErrors(cudaMallocManaged((void**)&rays_dev, n * sizeof(ray)));
    for (int i = 0; i < n; i++) rays_dev[i] = rays[i];

    hit_result* flat_res = run_hits(sc.list_root, rays_dev, n);
    hit_result* bvh_res  = run_hits(sc.bvh_root, rays_dev, n);

    int hits = 0, mismatches = 0;
    for (int i = 0; i < n; i++) {
        const hit_result& a = flat_res[i];
        const hit_result& b = bvh_res[i];
        bool same = a.hit == b.hit;
        if (same && a.hit) {
            same = a.t == b.t
                && a.p.x() == b.p.x() && a.p.y() == b.p.y() && a.p.z() == b.p.z()
                && a.normal.x() == b.normal.x() && a.normal.y() == b.normal.y() && a.normal.z() == b.normal.z()
                && a.front_face == b.front_face
                && (!compare_mat || a.mat == b.mat);
        }
        if (!same && mismatches < 5)
            std::printf("    mismatch ray %d: flat(hit=%d t=%.17g) bvh(hit=%d t=%.17g)\n",
                        i, a.hit, a.t, b.hit, b.t);
        if (!same) mismatches++;
        if (a.hit) hits++;
    }
    CHECK(mismatches == 0, "%d of %d rays disagree between flat list and BVH", mismatches, n);

    cudaFree(rays_dev);
    cudaFree(flat_res);
    cudaFree(bvh_res);
    return hits;
}

// ---------------------------------------------------- structural invariants --

void check_invariants(const bvh_scene* b) {
    int N = b->prim_count;
    if (N == 0) {
        CHECK(b->node_count == 0, "empty BVH should have 0 nodes, has %d", b->node_count);
        return;
    }
    CHECK(b->node_count >= 1 && b->node_count <= 2 * N - 1,
          "node_count %d outside [1, %d]", b->node_count, 2 * N - 1);

    // prim_index is a permutation of [0, N)
    std::vector<int> idx(b->prim_index, b->prim_index + N);
    std::sort(idx.begin(), idx.end());
    for (int i = 0; i < N; i++)
        CHECK(idx[i] == i, "prim_index is not a permutation (slot %d)", i);

    std::vector<int> covered(N, 0);
    for (int i = 0; i < b->node_count; i++) {
        const bvh_node& n = b->nodes[i];
        if (n.prim_count > 0) {  // leaf
            CHECK(n.first_prim >= 0 && n.first_prim + n.prim_count <= N,
                  "leaf %d range [%d, %d) out of bounds", i, n.first_prim, n.first_prim + n.prim_count);
            for (int j = n.first_prim; j < n.first_prim + n.prim_count; j++) {
                CHECK(!covered[j], "prim_index slot %d covered by two leaves", j);
                covered[j] = 1;
            }
            // leaf bbox is exactly the union of its prims' boxes
            aabb u;
            for (int j = 0; j < n.prim_count; j++)
                u = aabb(u, b->prims[b->prim_index[n.first_prim + j]].bounding_box());
            CHECK(u.x.min == n.bbox.x.min && u.x.max == n.bbox.x.max &&
                  u.y.min == n.bbox.y.min && u.y.max == n.bbox.y.max &&
                  u.z.min == n.bbox.z.min && u.z.max == n.bbox.z.max,
                  "leaf %d bbox is not the union of its prims", i);
        } else {                 // internal
            CHECK(n.left > i && n.left < b->node_count, "node %d left child %d violates child > parent", i, n.left);
            CHECK(n.right > i && n.right < b->node_count, "node %d right child %d violates child > parent", i, n.right);
            aabb u(b->nodes[n.left].bbox, b->nodes[n.right].bbox);
            CHECK(u.x.min == n.bbox.x.min && u.x.max == n.bbox.x.max &&
                  u.y.min == n.bbox.y.min && u.y.max == n.bbox.y.max &&
                  u.z.min == n.bbox.z.min && u.z.max == n.bbox.z.max,
                  "internal node %d bbox is not the union of its children", i);
        }
    }
    for (int j = 0; j < N; j++)
        CHECK(covered[j], "prim_index slot %d not covered by any leaf", j);
}

// -------------------------------------------------------------------- tests --

void test_empty() {
    std::printf("  test_empty (N=0)\n");
    test_scene sc = make_random_scene(0, 1);
    check_invariants(sc.bvh);
    std::vector<ray> rays = make_rays(sc, 1000, 2);
    int hits = compare_hits(sc, rays);
    CHECK(hits == 0, "empty scene reported %d hits", hits);
    sc.destroy();
}

void test_single() {
    std::printf("  test_single (N=1)\n");
    test_scene sc = make_random_scene(1, 3);
    check_invariants(sc.bvh);
    CHECK(sc.bvh->node_count == 1, "single-prim BVH should be one leaf, has %d nodes", sc.bvh->node_count);
    CHECK(sc.bvh->nodes[0].prim_count == 1, "root should be a leaf with 1 prim");
    std::vector<ray> rays = make_rays(sc, 10000, 4);
    int hits = compare_hits(sc, rays);
    CHECK(hits > 0, "no rays hit the single sphere — weak test");
    sc.destroy();
}

void test_small_counts() {
    int sizes[] = {2, 3, 5, 16, 17};
    for (int n : sizes) {
        std::printf("  test_small (N=%d)\n", n);
        test_scene sc = make_random_scene(n, 100 + n);
        check_invariants(sc.bvh);
        std::vector<ray> rays = make_rays(sc, 20000, 200 + n);
        int hits = compare_hits(sc, rays);
        CHECK(hits > 0, "no hits at N=%d — weak test", n);
        sc.destroy();
    }
}

void test_large() {
    std::printf("  test_large (N=500, 100k rays)\n");
    test_scene sc = make_random_scene(500, 7, 40.0);
    check_invariants(sc.bvh);
    std::vector<ray> rays = make_rays(sc, 100000, 8, 40.0);
    int hits = compare_hits(sc, rays);
    CHECK(hits > 10000, "only %d/100000 rays hit — weak test", hits);
    std::printf("    (%d/100000 rays hit)\n", hits);
    sc.destroy();
}

void test_duplicates() {
    // Identical centers → identical centroids: the median split cannot
    // separate them spatially; the index tie-break must still terminate.
    // All duplicates share one material so the winning-duplicate ambiguity
    // (identical t) cannot cause a spurious mismatch.
    std::printf("  test_duplicates (8 identical + 4 normal)\n");
    test_scene sc;
    sc.materials.push_back(make_material());
    for (int i = 0; i < 8; i++)
        sc.add_sphere(point3(1, 2, 3), 1.0, sc.materials[0]);
    for (int i = 0; i < 4; i++)
        sc.add_sphere(point3(5.0 * i, -3, 0), 0.8, sc.materials[0]);
    sc.build_structures();
    check_invariants(sc.bvh);
    std::vector<ray> rays = make_rays(sc, 20000, 9);
    compare_hits(sc, rays);
    sc.destroy();
}

void test_refit_after_motion() {
    std::printf("  test_refit_after_motion (N=200, move half)\n");
    test_scene sc = make_random_scene(200, 10, 30.0);

    std::mt19937 rng(11);
    std::uniform_real_distribution<double> delta(-10.0, 10.0);
    for (size_t i = 0; i < sc.spheres.size(); i += 2) {
        sphere* s = sc.spheres[i];
        point3 c = s->center + vec3(delta(rng), delta(rng), delta(rng));
        new(s) sphere(c, s->radius, s->mat);  // recomputes the sphere's own bbox
    }

    sc.bvh->refit();       // topology unchanged, boxes refreshed
    sc.rebuild_list();     // fresh flat list = ground truth (old bbox was stale)
    check_invariants(sc.bvh);

    std::vector<ray> rays = make_rays(sc, 50000, 12, 30.0);
    int hits = compare_hits(sc, rays);
    CHECK(hits > 1000, "only %d hits after refit — weak test", hits);
    sc.destroy();
}

void test_rebuild_after_motion() {
    std::printf("  test_rebuild_after_motion (N=200, move half, full rebuild)\n");
    test_scene sc = make_random_scene(200, 13, 30.0);

    std::mt19937 rng(14);
    std::uniform_real_distribution<double> delta(-15.0, 15.0);
    for (size_t i = 0; i < sc.spheres.size(); i += 2) {
        sphere* s = sc.spheres[i];
        point3 c = s->center + vec3(delta(rng), delta(rng), delta(rng));
        new(s) sphere(c, s->radius, s->mat);
    }

    sc.bvh->build();       // full rebuild reusing the same buffers
    sc.rebuild_list();
    check_invariants(sc.bvh);

    std::vector<ray> rays = make_rays(sc, 50000, 15, 30.0);
    compare_hits(sc, rays);
    sc.destroy();
}

void test_incremental_add() {
    // Crosses the initial capacity (16) to exercise the growth path, then
    // rebuilds and revalidates.
    std::printf("  test_incremental_add (10 prims, then +30, rebuild)\n");
    test_scene sc = make_random_scene(10, 16);
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> pos(-20.0, 20.0);
    for (int i = 0; i < 30; i++) {
        sc.add_sphere(point3(pos(rng), pos(rng), pos(rng)), 0.7, sc.materials[0]);
        sc.bvh->add(*sc.wrappers.back());
        sc.list->add(sc.wrappers.back());
    }
    sc.bvh->build();
    check_invariants(sc.bvh);
    std::vector<ray> rays = make_rays(sc, 20000, 18);
    compare_hits(sc, rays);
    sc.destroy();
}

int main() {
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 2048));

    std::printf("BVH unit tests\n");
    test_empty();
    test_single();
    test_small_counts();
    test_large();
    test_duplicates();
    test_refit_after_motion();
    test_rebuild_after_motion();
    test_incremental_add();

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURE(S)\n", g_failures);
    return 1;
}
