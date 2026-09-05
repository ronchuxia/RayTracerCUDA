#ifndef VIEWER_DENOISER_H
#define VIEWER_DENOISER_H

#include "viewer/gbuffer.h"

struct denoiser {
    float3* output = nullptr;
    int out_w = 0, out_h = 0;

    virtual void setup(int in_w, int in_h, int out_w, int out_h) = 0;
    virtual void invoke(const color* accum, const gbuffer& gb, int samples, float blend) = 0;
    virtual ~denoiser() {}
};

// real -> float3
__global__ void prepare_input(const color* accum, gbuffer gb, int samples, int n,
                                float3* beauty, float3* albedo, float3* normal) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    real s = real(1) / samples;
    color c = accum[p] * s;
    color a = gb.albedo[p] * s;
    vec3  m = gb.normal[p] * s;
    beauty[p] = make_float3(c.x(), c.y(), c.z());
    albedo[p] = make_float3(a.x(), a.y(), a.z());
    normal[p] = make_float3(m.x(), m.y(), m.z());
}

#endif // VIEWER_DENOISER_H
