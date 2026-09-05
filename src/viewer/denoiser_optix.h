#ifndef VIEWER_DENOISER_OPTIX_H
#define VIEWER_DENOISER_OPTIX_H

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cstdio>
#include <cstdlib>
#include "viewer/denoiser.h"

#define checkOptix(val) check_optix( (val), #val, __FILE__, __LINE__ )

inline void check_optix(OptixResult result, char const *const func, const char *const file, int const line) {
    if (result != OPTIX_SUCCESS) {
        fprintf(stderr, "OptiX error %s (%d) at %s:%d '%s'\n",
                optixGetErrorName(result), result, file, line, func);
        exit(98);
    }
}

struct optix_denoiser : denoiser {
    // model kind and guide set are OptiX create-time options: to change them,
    // construct a new optix_denoiser (and setup() it again)
    const bool guide_albedo, guide_normal;
    OptixDeviceContext ctx = nullptr;
    OptixDenoiser      dn  = nullptr;
    OptixDenoiserSizes sizes{};
    int in_w = 0, in_h = 0;
    float3 *beauty = nullptr, *albedo = nullptr, *normal = nullptr;
    void*  state = nullptr;
    void*  scratch = nullptr;
    float* intensity = nullptr;

    optix_denoiser(bool guide_albedo, bool guide_normal,
                   OptixDenoiserModelKind kind = OPTIX_DENOISER_MODEL_KIND_AOV)
        : guide_albedo(guide_albedo), guide_normal(guide_normal) {
        checkOptix(optixInit());
        // context
        OptixDeviceContextOptions o{};
        o.logCallbackFunction = log_cb;
        o.logCallbackLevel = 2; // errors + warnings
        checkOptix(optixDeviceContextCreate(0, &o, &ctx));
        // denoiser
        OptixDenoiserOptions d{};
        d.guideAlbedo = guide_albedo;
        d.guideNormal = guide_normal;
        checkOptix(optixDenoiserCreate(ctx, kind, &d, &dn));
    }

    ~optix_denoiser() {
        release();
        if (dn)  optixDenoiserDestroy(dn);
        if (ctx) optixDeviceContextDestroy(ctx);
    }

    static void log_cb(unsigned level, const char* tag, const char* msg, void*) {
        fprintf(stderr, "optix[%u][%s]: %s\n", level, tag, msg);
    }

    // setup or re-setup the buffers after frame size changes
    void setup(int iw, int ih, int ow, int oh) override {
        release();
        in_w = iw; in_h = ih; out_w = ow; out_h = oh;
        checkOptix(optixDenoiserComputeMemoryResources(dn, ow, oh, &sizes));   // sized by OUTPUT
        checkCudaErrors(cudaMalloc(&state,   sizes.stateSizeInBytes));
        checkCudaErrors(cudaMalloc(&scratch, sizes.withoutOverlapScratchSizeInBytes));
        checkCudaErrors(cudaMalloc(&intensity, sizeof(float)));
        size_t nin = (size_t)iw * ih, nout = (size_t)ow * oh;
        checkCudaErrors(cudaMalloc(&beauty, nin  * sizeof(float3)));
        checkCudaErrors(cudaMalloc(&albedo, nin  * sizeof(float3)));
        checkCudaErrors(cudaMalloc(&normal, nin  * sizeof(float3)));
        checkCudaErrors(cudaMalloc(&output, nout * sizeof(float3)));
        checkOptix(optixDenoiserSetup(dn, 0, iw, ih, (CUdeviceptr)state, sizes.stateSizeInBytes,   // sized by INPUT
                                      (CUdeviceptr)scratch, sizes.withoutOverlapScratchSizeInBytes));
    }

    static OptixImage2D image(float3* d, int w, int h) {
        OptixImage2D im{};
        im.data = (CUdeviceptr)d;
        im.width = w; 
        im.height = h;
        im.rowStrideInBytes   = w * sizeof(float3);
        im.pixelStrideInBytes = sizeof(float3);
        im.format = OPTIX_PIXEL_FORMAT_FLOAT3;
        return im;
    }

    void invoke(const color* accum, const gbuffer& gb, int samples, float blend) override {
        int n = in_w * in_h;
        
        prepare_input<<<(n + 255) / 256, 256>>>(accum, gb, samples, n, beauty, albedo, normal);
        
        OptixDenoiserLayer layer{};
        layer.input  = image(beauty, in_w, in_h);
        layer.output = image(output, out_w, out_h);

        checkOptix(optixDenoiserComputeIntensity(dn, 0, &layer.input, (CUdeviceptr)intensity,
                                                 (CUdeviceptr)scratch, sizes.withoutOverlapScratchSizeInBytes));
        
        OptixDenoiserParams params{};
        params.hdrIntensity = (CUdeviceptr)intensity;
        params.blendFactor  = blend;
        
        OptixDenoiserGuideLayer guide{};
        if (guide_albedo) guide.albedo = image(albedo, in_w, in_h);
        if (guide_normal) guide.normal = image(normal, in_w, in_h);
        
        checkOptix(optixDenoiserInvoke(dn, 0, &params, (CUdeviceptr)state, sizes.stateSizeInBytes,
                                       &guide, &layer, 1, 0, 0,
                                       (CUdeviceptr)scratch, sizes.withoutOverlapScratchSizeInBytes));
    }

    // release the buffers
    void release() {
        cudaFree(state); cudaFree(scratch); cudaFree(intensity);
        cudaFree(beauty); cudaFree(albedo); cudaFree(normal); cudaFree(output);
        state = scratch = intensity = nullptr;
        beauty = albedo = normal = output = nullptr;
    }
};

#endif // VIEWER_DENOISER_OPTIX_H
