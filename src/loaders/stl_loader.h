#ifndef STL_LOADER_H
#define STL_LOADER_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda_helper.h"
#include "hittable.h"
#include "vec3.h"

// Host-side binary-STL mesh loader (Phase 4).
//
// Binary STL layout: 80-byte header, uint32 triangle count, then per facet
// 12 little-endian floats (normal, v0, v1, v2) + a uint16 attribute field
// (50 bytes/facet).
//
// The triangles are constructed into ONE contiguous cudaMallocManaged array
// (not one allocation per triangle): a 10k+-triangle mesh in scattered
// per-object allocations would pointer-chase into cache-cold pages on every
// leaf intersection — see roadmap workstream E1. A dedicated bvh is
// built over the array and returned wrapped as a hittable{BVH}, so the caller
// can transform-wrap it (scale/rotate/translate) and add it to the world; the
// mesh BVH then becomes one leaf of the world's BVH (nested BVH).
//
// Allocations (tris array, bvh, wrapper) are recorded in `allocs` for
// the scene's free loop; the bvh* is also recorded in `bvh_dtors` so its
// destructor runs at teardown (it frees the BVH's internal buffers).
inline hittable* load_stl(const char* path, material* mat,
                          std::vector<void*>& allocs,
                          std::vector<bvh*>& bvh_dtors) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        std::cerr << "load_stl: cannot open '" << path << "'\n";
        std::exit(1);
    }

    char header[80];
    uint32_t tri_count = 0;
    if (fread(header, 80, 1, file) != 1 || fread(&tri_count, 4, 1, file) != 1) {
        std::cerr << "load_stl: '" << path << "' is not a binary STL (truncated header)\n";
        std::exit(1);
    }
    std::clog << "Loading STL mesh: " << path << " (" << tri_count << " triangles)\n" << std::flush;

    // One contiguous block for the whole mesh.
    triangle* tris;
    checkCudaErrors(cudaMallocManaged((void**)&tris, tri_count * sizeof(triangle)));
    allocs.push_back(tris);

    // Dedicated BVH over the mesh's triangles.
    bvh* mesh_bvh;
    checkCudaErrors(cudaMallocManaged((void**)&mesh_bvh, sizeof(bvh)));
    new(mesh_bvh) bvh();
    allocs.push_back(mesh_bvh);
    bvh_dtors.push_back(mesh_bvh);

    for (uint32_t i = 0; i < tri_count; i++) {
        float n[3];
        float v[3][3];
        uint16_t attribute;
        if (fread(n, 4, 3, file) != 3 || fread(v, 4, 9, file) != 9 ||
            fread(&attribute, 2, 1, file) != 1) {
            std::cerr << "load_stl: '" << path << "' truncated at facet " << i
                      << " of " << tri_count << "\n";
            std::exit(1);
        }

        point3 v0(v[0][0], v[0][1], v[0][2]);
        point3 v1(v[1][0], v[1][1], v[1][2]);
        point3 v2(v[2][0], v[2][1], v[2][2]);

        // STL facet normals are nominally unit length but are often zero or
        // unnormalized in the wild; normalize, falling back to the winding-
        // derived normal (fully degenerate facets can never be hit anyway).
        vec3 normal(n[0], n[1], n[2]);
        if (normal.length_squared() < 1e-12)
            normal = cross(v1 - v0, v2 - v0);
        normal = (normal.length_squared() < 1e-12) ? vec3(0, 0, 1)
                                                   : unit_vector(normal);

        new(&tris[i]) triangle(v0, v1, v2, normal, mat);

        hittable h;             // stack temp: mesh_bvh->add copies the wrapper by value
        h.type = TRIANGLE;
        h.id = -1;              // mesh triangles are untagged sub-parts
        h.object = &tris[i];
        mesh_bvh->add(h);
    }
    fclose(file);

    mesh_bvh->build();

    hittable* mesh_bvh_hittable;
    checkCudaErrors(cudaMallocManaged((void**)&mesh_bvh_hittable, sizeof(hittable)));
    mesh_bvh_hittable->type = BVH;
    mesh_bvh_hittable->id = -1;
    mesh_bvh_hittable->object = mesh_bvh;
    allocs.push_back(mesh_bvh_hittable);
    return mesh_bvh_hittable;
}

#endif // STL_LOADER_H
