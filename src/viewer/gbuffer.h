#ifndef VIEWER_GBUFFER_H
#define VIEWER_GBUFFER_H

#include "cuda_helper.h"
#include "vec3.h"
#include "color.h"

struct gbuffer {
    color* albedo = nullptr;
    vec3*  normal = nullptr;
    size_t pixels = 0;
    int    samples = 0;

    void allocate(int w, int h) {
        release();
        pixels = (size_t)w * h;
        checkCudaErrors(cudaMalloc(&albedo, pixels * sizeof(color)));
        checkCudaErrors(cudaMalloc(&normal, pixels * sizeof(vec3)));
        clear();
    }
    void clear() {
        checkCudaErrors(cudaMemset(albedo, 0, pixels * sizeof(color)));
        checkCudaErrors(cudaMemset(normal, 0, pixels * sizeof(vec3)));
        samples = 0;
    }
    void release() {
        checkCudaErrors(cudaFree(albedo));
        checkCudaErrors(cudaFree(normal));
        albedo = nullptr; normal = nullptr; pixels = 0;
    }
};

#endif // VIEWER_GBUFFER_H
