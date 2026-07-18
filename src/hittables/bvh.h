#ifndef BVH_H
#define BVH_H

#include <algorithm>
#include <vector>

#include "aabb.h"
#include "cuda_helper.h"
#include "hit_record.h"
#include "hittable.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

// Flattened BVH (see docs/plans/flattened-bvh.md).
//
// The tree lives in one pre-allocated array of bvh_node addressed by integer
// indices — no per-node allocation, no pointer chasing. The host build permutes
// prim_index[] (never the primitives), allocating each parent's slot BEFORE its
// children so that child_index > parent_index always holds; that invariant is
// what makes refit() a single reverse sweep. Device traversal is iterative with
// a small local stack (no device recursion).
//
// NOTE: a future GPU (LBVH / Morton) build would NOT preserve the
// child_index > parent_index numbering — refit() would need to switch to a
// bottom-up pass (parent links + atomic flags) if the build moves to the GPU.

// Max primitives per leaf. Ranges of this size or smaller stop subdividing.
#ifndef BVH_LEAF_SIZE
#define BVH_LEAF_SIZE 2
#endif

struct bvh_node {
    aabb bbox;
    int left;        // internal: left-child index;  leaf: -1
    int right;       // internal: right-child index; leaf: -1
    int first_prim;  // leaf: start offset into prim_index[]; internal: -1
    int prim_count;  // leaf: #prims (> 0); internal: 0  ← leaf/internal discriminator
};

struct bvh {
    bvh_node* nodes;      // pre-allocated flat tree, root at index 0
    int node_count;
    int node_capacity;

    int* prim_index;      // permutation of [0..prim_count); build sorts THIS
    hittable* prims;      // the actual objects (by-value wrappers)
    int prim_count;
    int prim_capacity;

    bvh() {
        prim_count = 0;
        prim_capacity = 16;
        checkCudaErrors(cudaMallocManaged((void**)&prims, prim_capacity * sizeof(hittable)));
        checkCudaErrors(cudaMallocManaged((void**)&prim_index, prim_capacity * sizeof(int)));
        node_count = 0;
        node_capacity = 2 * prim_capacity - 1;
        checkCudaErrors(cudaMallocManaged((void**)&nodes, node_capacity * sizeof(bvh_node)));
    }

    ~bvh() {
        cudaFree(prims);
        cudaFree(prim_index);
        cudaFree(nodes);
    }

    // Append a primitive (copies the {type, object} wrapper; the pointed-at
    // shape is owned by the caller, as in scene()). Call build() afterwards.
    __host__ void add(hittable object) {
        if (prim_count >= prim_capacity) {
            prim_capacity *= 2;
            hittable* new_prims;
            checkCudaErrors(cudaMallocManaged((void**)&new_prims, prim_capacity * sizeof(hittable)));
            int* new_prim_index;
            checkCudaErrors(cudaMallocManaged((void**)&new_prim_index, prim_capacity * sizeof(int)));
            for (int i = 0; i < prim_count; i++) {
                new_prims[i] = prims[i];
                new_prim_index[i] = prim_index[i];
            }
            cudaFree(prims);
            cudaFree(prim_index);
            prims = new_prims;
            prim_index = new_prim_index;
        }
        prims[prim_count] = object;
        prim_count++;
    }

    // (Re)build the tree over prims[0..prim_count). Reuses the node and index
    // buffers, growing them only if the primitive count outgrew them.
    __host__ void build() {
        node_count = 0;
        if (prim_count == 0) return;

        int needed_nodes = 2 * prim_count - 1;  // exact upper bound for a binary tree
        if (needed_nodes > node_capacity) {
            cudaFree(nodes);
            node_capacity = needed_nodes;
            checkCudaErrors(cudaMallocManaged((void**)&nodes, node_capacity * sizeof(bvh_node)));
        }

        for (int i = 0; i < prim_count; i++)
            prim_index[i] = i;

        // Precompute one centroid per primitive (bbox center); the split
        // comparator and axis selection only ever look at these.
        std::vector<point3> centroids(prim_count);
        for (int i = 0; i < prim_count; i++) {
            aabb b = prims[i].bounding_box();
            centroids[i] = point3((b.x.min + b.x.max) / 2,
                                  (b.y.min + b.y.max) / 2,
                                  (b.z.min + b.z.max) / 2);
        }

        build_range(0, prim_count, centroids.data());  // root lands in slot 0
    }

    // Refresh every node's bbox after primitives moved (topology unchanged).
    // Valid because child_index > parent_index: one reverse sweep sees every
    // child before its parent. O(N), no sorting.
    __host__ void refit() {
        if (prim_count == 0 || node_count == 0) return;
        for (int i = node_count - 1; i >= 0; i--) {
            if (nodes[i].prim_count > 0) {  // leaf: union of its prims' boxes
                aabb bounds;
                for (int j = 0; j < nodes[i].prim_count; j++)
                    bounds = aabb(bounds, prims[prim_index[nodes[i].first_prim + j]].bounding_box());
                nodes[i].bbox = bounds;
            } else {                        // internal: union of the children
                nodes[i].bbox = aabb(nodes[nodes[i].left].bbox, nodes[nodes[i].right].bbox);
            }
        }
    }

    __host__ __device__ aabb bounding_box() const {
        return node_count > 0 ? nodes[0].bbox : aabb();
    }

    // Iterative closest-hit traversal (no device recursion). stack[64] bounds
    // tree depth; the median build keeps depth ~log2(N), so 64 is ample.
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
        if (node_count == 0) return false;

        int stack[64];
        int sp = 0;
        stack[sp++] = 0;  // root

        auto hit_anything = false;
        auto closest_so_far = ray_t.max;

        while (sp > 0) {
            const bvh_node& node = nodes[stack[--sp]];

            // Prune this whole subtree if the ray misses its box, or could
            // only hit it beyond the closest hit found so far.
            if (!node.bbox.hit(r, interval(ray_t.min, closest_so_far))) continue;

            if (node.prim_count > 0) {  // leaf: test its primitives
                for (int i = 0; i < node.prim_count; i++) {
                    int pi = prim_index[node.first_prim + i];
                    // Writing rec directly is safe: prims[pi].hit only fills
                    // rec when the hit is inside [ray_t.min, closest_so_far),
                    // i.e. strictly closer than the best so far.
                    if (prims[pi].hit(r, interval(ray_t.min, closest_so_far), rec, state)) {
                        hit_anything = true;
                        closest_so_far = rec.t;
                    }
                }
            } else {                    // internal: descend into both children
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }

        return hit_anything;
    }

    // Recursive host build over prim_index[start, end); returns this
    // subtree's node slot. Claims the parent slot BEFORE recursing so that
    // child_index > parent_index (pre-order numbering) — refit() depends on it.
    __host__ int build_range(int start, int end, const point3* centroids) {
        int slot = node_count++;

        if (end - start <= BVH_LEAF_SIZE) {  // leaf
            aabb bounds;
            for (int i = start; i < end; i++)
                bounds = aabb(bounds, prims[prim_index[i]].bounding_box());
            nodes[slot].bbox = bounds;
            nodes[slot].left = -1;
            nodes[slot].right = -1;
            nodes[slot].first_prim = start;
            nodes[slot].prim_count = end - start;
            return slot;
        }

        // Split axis: longest axis of the centroid bounds.
        aabb centroid_bounds;
        for (int i = start; i < end; i++) {
            point3 c = centroids[prim_index[i]];
            centroid_bounds = aabb(centroid_bounds, aabb(c, c));
        }
        int axis = 0;
        if (centroid_bounds.y.size() > centroid_bounds.axis(axis).size()) axis = 1;
        if (centroid_bounds.z.size() > centroid_bounds.axis(axis).size()) axis = 2;

        // Median-of-count partition on that axis (deterministic: ties broken
        // by primitive index). Splitting at the middle INDEX guarantees both
        // halves are non-empty even if all centroids coincide.
        int mid = start + (end - start) / 2;
        std::nth_element(prim_index + start, prim_index + mid, prim_index + end,
            [centroids, axis](int a, int b) {
                real ca = centroids[a][axis];
                real cb = centroids[b][axis];
                return ca < cb || (ca == cb && a < b);
            });

        int L = build_range(start, mid, centroids);
        int R = build_range(mid, end, centroids);
        nodes[slot].bbox = aabb(nodes[L].bbox, nodes[R].bbox);  // == union of the range's prims
        nodes[slot].left = L;
        nodes[slot].right = R;
        nodes[slot].first_prim = -1;
        nodes[slot].prim_count = 0;
        return slot;
    }
};

// Dispatch shims declared in hittable.h (before bvh is complete) and
// defined here, so hittable's switches can route to the BVH.
__device__ bool bvh_hit(const bvh* b, const ray& r, interval ray_t, hit_record& rec, curandState* state) {
    return b->hit(r, ray_t, rec, state);
}

__host__ __device__ aabb bvh_bounding_box(const bvh* b) {
    return b->bounding_box();
}

#endif // BVH_H
