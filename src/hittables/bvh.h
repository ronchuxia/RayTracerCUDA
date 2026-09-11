#ifndef BVH_H
#define BVH_H

#include <algorithm>
#include <curand_kernel.h>
#include <vector>

#include "aabb.h"
#include "cuda_helper.h"
#include "hit_record.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

// Flattened BVH over items of type T
#ifndef BVH_LEAF_SIZE
#define BVH_LEAF_SIZE 2
#endif

struct bvh_node {
    aabb bbox;
    int left;        // internal: left-child index;  leaf: -1
    int right;       // internal: right-child index; leaf: -1
    int first_item;  // leaf: start offset into item_index[]; internal: -1
    int item_count;  // leaf: #items (> 0); internal: 0  ← leaf/internal discriminator
};

template <class T>
struct bvh {
    bvh_node* nodes;      // pre-allocated flat tree
    int node_count;
    int node_capacity;

    int* item_index;      // permutation of [0..item_count)
    T* items;             // the items by value, in insertion order (never permuted)
    int item_count;
    int item_capacity;

    bvh() {
        item_count = 0;
        item_capacity = 16;
        checkCudaErrors(cudaMallocManaged((void**)&items, item_capacity * sizeof(T)));
        checkCudaErrors(cudaMallocManaged((void**)&item_index, item_capacity * sizeof(int)));
        node_count = 0;
        node_capacity = 2 * item_capacity - 1;
        checkCudaErrors(cudaMallocManaged((void**)&nodes, node_capacity * sizeof(bvh_node)));
    }

    ~bvh() {
        cudaFree(items);
        cudaFree(item_index);
        cudaFree(nodes);
    }

    __host__ void add(const T& object) {
        if (item_count >= item_capacity) {
            item_capacity *= 2;
            T* new_items;
            checkCudaErrors(cudaMallocManaged((void**)&new_items, item_capacity * sizeof(T)));
            int* new_item_index;
            checkCudaErrors(cudaMallocManaged((void**)&new_item_index, item_capacity * sizeof(int)));
            for (int i = 0; i < item_count; i++) {
                new_items[i] = items[i];
                new_item_index[i] = item_index[i];
            }
            cudaFree(items);
            cudaFree(item_index);
            items = new_items;
            item_index = new_item_index;
        }
        items[item_count] = object;
        item_count++;
    }

    // (Re)build the tree over items[0..item_count)
    __host__ void build() {
        node_count = 0;
        if (item_count == 0) return;

        int needed_nodes = 2 * item_count - 1;  // exact upper bound for a binary tree
        if (needed_nodes > node_capacity) {
            cudaFree(nodes);
            node_capacity = needed_nodes;
            checkCudaErrors(cudaMallocManaged((void**)&nodes, node_capacity * sizeof(bvh_node)));
        }

        for (int i = 0; i < item_count; i++)
            item_index[i] = i;

        // aabb centroids
        std::vector<point3> centroids(item_count);
        for (int i = 0; i < item_count; i++) {
            aabb b = items[i].bounding_box();
            centroids[i] = point3((b.x.min + b.x.max) / 2,
                                  (b.y.min + b.y.max) / 2,
                                  (b.z.min + b.z.max) / 2);
        }

        build_range(0, item_count, centroids.data());
    }

    // Refresh every node's bbox after items moved
    __host__ void refit() {
        if (item_count == 0 || node_count == 0) return;
        for (int i = node_count - 1; i >= 0; i--) {
            if (nodes[i].item_count > 0) {  // leaf: union of its items' boxes
                aabb bounds;
                for (int j = 0; j < nodes[i].item_count; j++)
                    bounds = aabb(bounds, items[item_index[nodes[i].first_item + j]].bounding_box());
                nodes[i].bbox = bounds;
            } else {                        // internal: union of the children
                nodes[i].bbox = aabb(nodes[nodes[i].left].bbox, nodes[nodes[i].right].bbox);
            }
        }
    }

    __host__ __device__ aabb bounding_box() const {
        return node_count > 0 ? nodes[0].bbox : aabb();
    }

    // Iterative closest-hit traversal
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
        if (node_count == 0) return false;

        int stack[64];
        int sp = 0;
        stack[sp++] = 0;  // root

        auto hit_anything = false;
        auto closest_so_far = ray_t.max;

        while (sp > 0) {
            const bvh_node& node = nodes[stack[--sp]];

            if (!node.bbox.hit(r, interval(ray_t.min, closest_so_far))) continue;

            if (node.item_count > 0) {  // leaf: test its primitives
                for (int i = 0; i < node.item_count; i++) {
                    int pi = item_index[node.first_item + i];
                    if (hit_item(items[pi], r, interval(ray_t.min, closest_so_far), rec, state)) {
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

    // Recursive build over item_index[start, end)
    __host__ int build_range(int start, int end, const point3* centroids) {
        int slot = node_count++;

        if (end - start <= BVH_LEAF_SIZE) {  // leaf
            aabb bounds;
            for (int i = start; i < end; i++)
                bounds = aabb(bounds, items[item_index[i]].bounding_box());
            nodes[slot].bbox = bounds;
            nodes[slot].left = -1;
            nodes[slot].right = -1;
            nodes[slot].first_item = start;
            nodes[slot].item_count = end - start;
            return slot;
        }

        // split axis
        aabb centroid_bounds;
        for (int i = start; i < end; i++) {
            point3 c = centroids[item_index[i]];
            centroid_bounds = aabb(centroid_bounds, aabb(c, c));
        }
        int axis = 0;
        if (centroid_bounds.y.size() > centroid_bounds.axis(axis).size()) axis = 1;
        if (centroid_bounds.z.size() > centroid_bounds.axis(axis).size()) axis = 2;

        // median-of-count partition on split axis
        int mid = start + (end - start) / 2;
        std::nth_element(item_index + start, item_index + mid, item_index + end,
            [centroids, axis](int a, int b) {
                real ca = centroids[a][axis];
                real cb = centroids[b][axis];
                return ca < cb || (ca == cb && a < b);
            });

        int L = build_range(start, mid, centroids);
        int R = build_range(mid, end, centroids);
        nodes[slot].bbox = aabb(nodes[L].bbox, nodes[R].bbox);
        nodes[slot].left = L;
        nodes[slot].right = R;
        nodes[slot].first_item = -1;
        nodes[slot].item_count = 0;
        return slot;
    }
};

#endif // BVH_H
