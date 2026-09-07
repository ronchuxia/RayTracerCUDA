#ifndef VIEWER_DENOISER_DLSS_H
#define VIEWER_DENOISER_DLSS_H

// DLSS Ray Reconstruction through NGX CUDA backend
#include <cuda.h>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_dlssd_cuda.h>
#include <cstdio>
#include <cstdlib>
#include "viewer/denoiser.h"

#ifndef DLSS_SNIPPET_DIR
#define DLSS_SNIPPET_DIR L"src/external/dlss/lib"
#endif

#define checkNgx(val) check_ngx((val), #val, __FILE__, __LINE__)

inline void check_ngx(NVSDK_NGX_Result r, const char* func, const char* file, int line) {
    if (NVSDK_NGX_FAILED(r)) {
        fprintf(stderr, "NGX error 0x%x at %s:%d '%s'\n", (unsigned)r, file, line, func);
        exit(97);
    }
}

__device__ inline float3 env_brdf_approx2(float3 f0, float alpha, float nov) {
    nov = fabsf(nov);
    float x1 = nov, x2 = nov * nov, x3 = nov * x2;
    float y1 = alpha, y2 = alpha * alpha, y3 = alpha * y2;
    float bias  = ((0.99044f - 1.28514f * x1) + (1.29678f - 0.755907f * x1) * y1)
                / ((1.f + 2.92338f * x1 + 59.4188f * x3) + (20.3225f - 27.0302f * x1 + 222.592f * x3) * y1 + (121.563f + 626.13f * x1 + 316.627f * x3) * y3);
    float scale = ((0.0365463f + 3.32707f * x1) + (9.0632f - 9.04756f * x1) * y1)
                / ((1.f + 3.59685f * x2 - 1.36772f * x3) + (9.04401f - 16.3174f * x2 + 9.22949f * x3) * y1 + (5.56589f + 19.7886f * x2 - 20.2123f * x3) * y3);
    bias *= fminf(fmaxf(f0.y * 50.f, 0.f), 1.f);       // the guide's hack for a reflectance of 0
    scale = fmaxf(scale, 0.f); bias = fmaxf(bias, 0.f);
    return make_float3(f0.x * scale + bias, f0.y * scale + bias, f0.z * scale + bias);
}

__global__ void prepare_dlss(const color* accum, int samples, primary_hits ph, const float2* flow, point3 center, real jx, real jy, int n,
                             float4* colour, float4* diffuse, float4* specular, float4* normal, float* roughness, float* depth, float2* mv) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;

    color c      = accum[p] / real(samples);
    colour[p]    = make_float4(c.x(), c.y(), c.z(), 1.f);
    
    color d      = ph.diffuse[p];
    diffuse[p]   = make_float4(d.x(), d.y(), d.z(), 1.f);
    
    vec3 nrm     = ph.normal[p];
    normal[p]    = make_float4(nrm.x(), nrm.y(), nrm.z(), 0.f);
    
    float rough  = ph.roughness[p];
    roughness[p] = rough;

    float3 s;
    if (ph.id[p] < 0) s = make_float3(0.5f, 0.5f, 0.5f);      // sky
    else {
        color f0 = ph.f0[p];
        float nov = dot(nrm, unit_vector(center - ph.p[p]));
        s = env_brdf_approx2(make_float3(f0.x(), f0.y(), f0.z()), rough * rough, nov);
    }
    specular[p]  = make_float4(s.x, s.y, s.z, 1.f);
    
    real z       = ph.depth[p];
    depth[p]     = fminf((float)z, 1e4f);

    float2 f     = flow[p];
    mv[p]        = make_float2(-f.x - (float)jx, -f.y - (float)jy);   // current + mv = previous
}

__global__ void unpack_output(cudaSurfaceObject_t surf, float3* out, int w, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;
    float4 c = surf2Dread<float4>(surf, i * sizeof(float4), j);
    out[j * w + i] = make_float3(c.x, c.y, c.z);
}

struct dlss_denoiser : denoiser {
    const NVSDK_NGX_PerfQuality_Value quality;
    CUcontext ctx = nullptr;
    cudaStream_t stream = nullptr;
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    int in_w = 0, in_h = 0;
    bool reset = true;

    struct image { 
        cudaArray_t arr = nullptr; 
        cudaTextureObject_t tex = 0; 
        cudaSurfaceObject_t surf = 0; 
        size_t pitch = 0; 
    };
    
    image colour, diffuse, specular, normal, roughness, depth, mv, out;
    float4 *l_colour = nullptr, *l_diffuse = nullptr, *l_specular = nullptr, *l_normal = nullptr;   // linear staging, prepare_dlss writes here
    float  *l_roughness = nullptr, *l_depth = nullptr;
    float2 *l_mv = nullptr;

    static int& ngx_users() { static int n = 0; return n; }

    dlss_denoiser(NVSDK_NGX_PerfQuality_Value quality) : quality(quality) {
        checkCudaErrors(cudaFree(0));
        cuCtxGetCurrent(&ctx);
        checkCudaErrors(cudaStreamCreate(&stream));
        if (ngx_users()++ == 0) {
            wchar_t* paths[1] = { (wchar_t*)DLSS_SNIPPET_DIR };
            NVSDK_NGX_FeatureCommonInfo info{};
            info.PathListInfo.Path = paths;
            info.PathListInfo.Length = 1;
            info.LoggingInfo.LoggingCallback = log_cb;
            info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;   // ON prints ~60 snippet-search lines per init; errors come back as results
            info.LoggingInfo.DisableOtherLoggingSinks = true;
            checkNgx(NVSDK_NGX_CUDA_Init_with_ProjectID("a0676bfa-99ea-4d5e-9a1b-3c8f2e1d7b44", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", L"build", &info));
            NVSDK_NGX_Parameter* caps = nullptr;
            checkNgx(NVSDK_NGX_CUDA_GetCapabilityParameters(&caps));
            int available = 0;
            NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &available);
            if (!available) { fprintf(stderr, "DLSS-RR is not available on this driver/GPU\n"); exit(97); }
            NVSDK_NGX_CUDA_DestroyParameters(caps);
        }
        checkNgx(NVSDK_NGX_CUDA_AllocateParameters(&params));
    }

    ~dlss_denoiser() {
        release();
        NVSDK_NGX_CUDA_DestroyParameters(params);
        if (--ngx_users() == 0) NVSDK_NGX_CUDA_Shutdown1(nullptr);
        cudaStreamDestroy(stream);
    }

    static void NVSDK_CONV log_cb(const char* msg, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) { fprintf(stderr, "ngx: %s", msg); }

    static image make_image(int w, int h, cudaChannelFormatDesc desc, size_t pixel_bytes) {
        image im;
        im.pitch = (size_t)w * pixel_bytes;
        checkCudaErrors(cudaMallocArray(&im.arr, &desc, w, h, cudaArraySurfaceLoadStore));
        cudaResourceDesc rd{};
        rd.resType = cudaResourceTypeArray;
        rd.res.array.array = im.arr;
        cudaTextureDesc td{};
        td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
        td.filterMode = cudaFilterModePoint;
        td.readMode = cudaReadModeElementType;
        checkCudaErrors(cudaCreateTextureObject(&im.tex, &rd, &td, nullptr));
        checkCudaErrors(cudaCreateSurfaceObject(&im.surf, &rd));
        return im;
    }

    static void free_image(image& im) {
        if (im.tex)  cudaDestroyTextureObject(im.tex);
        if (im.surf) cudaDestroySurfaceObject(im.surf);
        if (im.arr)  cudaFreeArray(im.arr);
        im = image{};
    }

    void setup(int iw, int ih, int ow, int oh) override {
        release();
        in_w = iw; in_h = ih; out_w = ow; out_h = oh;
        size_t n = (size_t)iw * ih;
        cudaChannelFormatDesc f4 = cudaCreateChannelDesc<float4>(), f2 = cudaCreateChannelDesc<float2>(), f1 = cudaCreateChannelDesc<float>();
        colour    = make_image(iw, ih, f4, 16);
        diffuse   = make_image(iw, ih, f4, 16);
        specular  = make_image(iw, ih, f4, 16);
        normal    = make_image(iw, ih, f4, 16);
        roughness = make_image(iw, ih, f1, 4);
        depth     = make_image(iw, ih, f1, 4);
        mv        = make_image(iw, ih, f2, 8);
        out       = make_image(ow, oh, f4, 16);
        checkCudaErrors(cudaMalloc(&l_colour,    n * sizeof(float4)));
        checkCudaErrors(cudaMalloc(&l_diffuse,   n * sizeof(float4)));
        checkCudaErrors(cudaMalloc(&l_specular,  n * sizeof(float4)));
        checkCudaErrors(cudaMalloc(&l_normal,    n * sizeof(float4)));
        checkCudaErrors(cudaMalloc(&l_roughness, n * sizeof(float)));
        checkCudaErrors(cudaMalloc(&l_depth,     n * sizeof(float)));
        checkCudaErrors(cudaMalloc(&l_mv,        n * sizeof(float2)));
        checkCudaErrors(cudaMalloc(&output, (size_t)ow * oh * sizeof(float3)));

        NVSDK_NGX_CUDA_DLSSD_Create_Params cp{};
        cp.Feature.InDenoiseMode    = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
        cp.Feature.InRoughnessMode  = NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;
        cp.Feature.InUseHWDepth     = NVSDK_NGX_DLSS_Depth_Type_Linear;
        cp.Feature.InWidth          = iw;
        cp.Feature.InHeight         = ih;
        cp.Feature.InTargetWidth    = ow;
        cp.Feature.InTargetHeight   = oh;
        cp.Feature.InPerfQualityValue = quality;
        cp.Feature.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
        cp.InCUContext = ctx;
        cp.InCUStream  = stream;
        checkNgx(NGX_CUDA_CREATE_DLSSD_EXT(&feature, params, &cp));
        reset = true;
    }

    void reset_history() override { reset = true; }

    void invoke(const color* accum, const gbuffer&, int samples, float, const flow_field& ff,
                const primary_hits& ph, const camera& cam) override {
        int n = in_w * in_h;
        prepare_dlss<<<(n + 255) / 256, 256>>>(accum, samples, ph, ff.flow, cam.center, cam.jitter_x, cam.jitter_y, n,
                                               l_colour, l_diffuse, l_specular, l_normal, l_roughness, l_depth, l_mv);
        auto upload = [&](image& im, const void* src) {
            checkCudaErrors(cudaMemcpy2DToArrayAsync(im.arr, 0, 0, src, im.pitch, im.pitch, in_h, cudaMemcpyDeviceToDevice, 0));
        };
        upload(colour, l_colour); upload(diffuse, l_diffuse); upload(specular, l_specular); upload(normal, l_normal);
        upload(roughness, l_roughness); upload(depth, l_depth); upload(mv, l_mv);

        NVSDK_NGX_CUDA_DLSSD_Eval_Params ep{};
        ep.pInColor          = &colour.tex;
        ep.pInDiffuseAlbedo  = &diffuse.tex;
        ep.pInSpecularAlbedo = &specular.tex;
        ep.pInNormals        = &normal.tex;
        ep.pInRoughness      = &roughness.tex;
        ep.pInDepth          = &depth.tex;
        ep.pInMotionVectors  = &mv.tex;
        ep.pInOutput         = &out.surf;
        ep.InRenderSubrectDimensions = { (unsigned)in_w, (unsigned)in_h };
        ep.InJitterOffsetX = -(float)cam.jitter_x;
        ep.InJitterOffsetY = -(float)cam.jitter_y;
        ep.InMVScaleX = ep.InMVScaleY = 1.f;
        ep.InReset = reset;
        checkNgx(NGX_CUDA_EVALUATE_DLSSD_EXT(feature, params, &ep));
        reset = false;

        dim3 t(16, 16), b((out_w + 15) / 16, (out_h + 15) / 16);
        unpack_output<<<b, t>>>(out.surf, output, out_w, out_h);
    }

    void release() {
        if (feature) { NVSDK_NGX_CUDA_ReleaseFeature(feature); feature = nullptr; }
        for (image* im : { &colour, &diffuse, &specular, &normal, &roughness, &depth, &mv, &out }) free_image(*im);
        cudaFree(l_colour); cudaFree(l_diffuse); cudaFree(l_specular); cudaFree(l_normal);
        cudaFree(l_roughness); cudaFree(l_depth); cudaFree(l_mv); cudaFree(output);
        l_colour = l_diffuse = l_specular = l_normal = nullptr;
        l_roughness = l_depth = nullptr; l_mv = nullptr; output = nullptr;
    }
};

#endif // VIEWER_DENOISER_DLSS_H
