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

// invert flow field trust for Optix 9.1 on driver 595.84
__global__ void invert_trust(const float* trust, float* out, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) out[p] = 1.f - trust[p];
}

struct optix_denoiser : denoiser {
    const bool guide_albedo, guide_normal;
    const bool temporal;
    OptixDeviceContext ctx = nullptr;
    OptixDenoiser      dn  = nullptr;
    OptixDenoiserSizes sizes{};
    int in_w = 0, in_h = 0;
    float3 *beauty = nullptr, *albedo = nullptr, *normal = nullptr;
    void*  state = nullptr;
    void*  scratch = nullptr;
    float* intensity = nullptr;
    
    // temporal state
    float3* prev_output = nullptr;
    void*   ig_prev = nullptr;           // OptiX internal guide layer, previous frame
    void*   ig_cur  = nullptr;           // OptiX internal guide layer, current frame
    float*  distrust = nullptr;          // 1 - flow_field::trust
    bool    history = false;

    optix_denoiser(bool guide_albedo, bool guide_normal,
                   OptixDenoiserModelKind kind = OPTIX_DENOISER_MODEL_KIND_AOV)
        : guide_albedo(guide_albedo), guide_normal(guide_normal),
          temporal(kind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV ||
                   kind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X) {
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

    // setup or re-setup the buffers after frame size changes, also resets history
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
        if (temporal) {
            checkCudaErrors(cudaMalloc(&prev_output, nout * sizeof(float3)));
            checkCudaErrors(cudaMalloc(&ig_prev, nout * sizes.internalGuideLayerPixelSizeInBytes));
            checkCudaErrors(cudaMalloc(&ig_cur,  nout * sizes.internalGuideLayerPixelSizeInBytes));
            checkCudaErrors(cudaMemset(ig_prev, 0, nout * sizes.internalGuideLayerPixelSizeInBytes));
            checkCudaErrors(cudaMalloc(&distrust, nin * sizeof(float)));
        }
        history = false;
        checkOptix(optixDenoiserSetup(dn, 0, iw, ih, (CUdeviceptr)state, sizes.stateSizeInBytes,   // sized by INPUT
                                      (CUdeviceptr)scratch, sizes.withoutOverlapScratchSizeInBytes));
    }

    static OptixImage2D image(const void* d, int w, int h, size_t pixel_bytes = sizeof(float3), OptixPixelFormat format = OPTIX_PIXEL_FORMAT_FLOAT3) {
        OptixImage2D im{};
        im.data = (CUdeviceptr)d;
        im.width = w;
        im.height = h;
        im.rowStrideInBytes   = (unsigned)(w * pixel_bytes);
        im.pixelStrideInBytes = (unsigned)pixel_bytes;
        im.format = format;
        return im;
    }

    void reset_history() override {
        if (temporal) checkCudaErrors(cudaMemset(ig_prev, 0, (size_t)out_w * out_h * sizes.internalGuideLayerPixelSizeInBytes));
        history = false;
    }

    void invoke(const color* accum, const gbuffer& gb, int samples, float blend, const flow_field& ff) override {
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
        if (temporal) {
            invert_trust<<<(n + 255) / 256, 256>>>(ff.trust, distrust, n);

            unsigned ps = (unsigned)sizes.internalGuideLayerPixelSizeInBytes;
            guide.flow                             = image(ff.flow, in_w, in_h, sizeof(float2), OPTIX_PIXEL_FORMAT_FLOAT2);
            guide.flowTrustworthiness              = image(distrust, in_w, in_h, sizeof(float),  OPTIX_PIXEL_FORMAT_FLOAT1);
            guide.previousOutputInternalGuideLayer = image(ig_prev, out_w, out_h, ps, OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER);
            guide.outputInternalGuideLayer         = image(ig_cur,  out_w, out_h, ps, OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER);
            
            layer.previousOutput                   = image(prev_output, out_w, out_h);
            
            params.temporalModeUsePreviousLayers = history ? 1 : 0;
        }
        
        checkOptix(optixDenoiserInvoke(dn, 0, &params, (CUdeviceptr)state, sizes.stateSizeInBytes,
                                       &guide, &layer, 1, 0, 0,
                                       (CUdeviceptr)scratch, sizes.withoutOverlapScratchSizeInBytes));
        if (temporal) {
            checkCudaErrors(cudaMemcpyAsync(prev_output, output, (size_t)out_w * out_h * sizeof(float3), cudaMemcpyDeviceToDevice, 0));
            void* t = ig_prev; ig_prev = ig_cur; ig_cur = t;
            history = true;
        }
    }

    // release the buffers
    void release() {
        cudaFree(state); cudaFree(scratch); cudaFree(intensity);
        cudaFree(beauty); cudaFree(albedo); cudaFree(normal); cudaFree(output);
        cudaFree(prev_output); cudaFree(ig_prev); cudaFree(ig_cur); cudaFree(distrust);
        state = scratch = intensity = distrust = nullptr;
        beauty = albedo = normal = output = prev_output = nullptr;
        ig_prev = ig_cur = nullptr;
    }
};

#endif // VIEWER_DENOISER_OPTIX_H
