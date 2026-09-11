#ifndef TESTS_SINGLE_LEAF_H
#define TESTS_SINGLE_LEAF_H

#include "hittable.h"

// The brute-force reference for a bvh<T>: one leaf over every item, walked in
// insertion order by the ordinary traversal. What -DBVH_LEAF_SIZE=1000000
// produces at build time, made by hand so one binary can hold both trees.
template <class T>
void single_leaf(bvh<T>* b) {
    aabb bounds;
    for (int i = 0; i < b->item_count; i++) {
        b->item_index[i] = i;
        bounds = aabb(bounds, b->items[i].bounding_box());
    }
    b->nodes[0].bbox = bounds;
    b->nodes[0].left = -1;
    b->nodes[0].right = -1;
    b->nodes[0].first_item = 0;
    b->nodes[0].item_count = b->item_count;
    b->node_count = b->item_count > 0 ? 1 : 0;
}

#endif // TESTS_SINGLE_LEAF_H
