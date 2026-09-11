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

// Host-side binary-STL mesh loader.
//
// Binary STL layout: 80-byte header, uint32 triangle count, per facet 12 little-endian floats (normal, v0, v1, v2) + uint16 attribute.
inline mesh* load_stl(const char* path, material* mat,
                      std::vector<void*>& allocs,
                      std::vector<mesh*>& mesh_dtors) {
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

    triangle* tris;
    checkCudaErrors(cudaMallocManaged((void**)&tris, tri_count * sizeof(triangle)));
    allocs.push_back(tris);

    mesh* m;
    checkCudaErrors(cudaMallocManaged((void**)&m, sizeof(mesh)));
    new(m) mesh();
    allocs.push_back(m);
    mesh_dtors.push_back(m);

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

        // normalize STL facet normals
        vec3 normal(n[0], n[1], n[2]);
        if (normal.length_squared() < 1e-12)
            normal = cross(v1 - v0, v2 - v0);
        normal = (normal.length_squared() < 1e-12) ? vec3(0, 0, 1)
                                                   : unit_vector(normal);

        new(&tris[i]) triangle(v0, v1, v2, normal, mat);

        primitive p;
        p.type = TRIANGLE;
        p.object = &tris[i];
        m->add(p);
    }
    fclose(file);

    m->build();
    return m;
}

#endif // STL_LOADER_H
