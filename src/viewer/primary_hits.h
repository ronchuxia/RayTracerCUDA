#ifndef VIEWER_PRIMARY_HITS_H
#define VIEWER_PRIMARY_HITS_H

#include "cuda_helper.h"
#include "vec3.h"
#include "color.h"

struct primary_hits {
    point3* p = nullptr;          // hit point
    int*    id = nullptr;         // scene_id, -1 on miss
    vec3*   normal = nullptr;
    color*  diffuse = nullptr;    // diffuse albedo
    color*  f0 = nullptr;         // specular f0
    real*   roughness = nullptr;
    real*   depth = nullptr;      // axial distance
    size_t  pixels = 0;

    void allocate(int w, int h) {
        release();
        pixels = (size_t)w * h;
        checkCudaErrors(cudaMalloc(&p,         pixels * sizeof(point3)));
        checkCudaErrors(cudaMalloc(&id,        pixels * sizeof(int)));
        checkCudaErrors(cudaMalloc(&normal,    pixels * sizeof(vec3)));
        checkCudaErrors(cudaMalloc(&diffuse,   pixels * sizeof(color)));
        checkCudaErrors(cudaMalloc(&f0,        pixels * sizeof(color)));
        checkCudaErrors(cudaMalloc(&roughness, pixels * sizeof(real)));
        checkCudaErrors(cudaMalloc(&depth,     pixels * sizeof(real)));
    }
    void release() {
        cudaFree(p); cudaFree(id); cudaFree(normal); cudaFree(diffuse); cudaFree(f0); cudaFree(roughness); cudaFree(depth);
        p = nullptr; id = nullptr; normal = nullptr; diffuse = nullptr; f0 = nullptr; roughness = nullptr; depth = nullptr; pixels = 0;
    }
};

#endif // VIEWER_PRIMARY_HITS_H
