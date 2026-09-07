#ifndef VIEWER_FRAMEGEN_VK_H
#define VIEWER_FRAMEGEN_VK_H

// DLSS Frame Generation on the NGX Vulkan backend (docs/plans/dlss-fg.md steps 2–3).
// Owns the NGX Vulkan init, the DLSS-G feature and its images; present_vk drives it:
// CUDA writes hardware depth and motion vectors (render size) into two exported
// buffers, evaluate() copies them into images and records one NGX evaluate per
// generated frame into present_vk's command buffer, reading the tonemapped frame
// image (window size, HUD-less: ImGui is drawn on the swapchain afterwards) and
// writing out[k]; present_vk then presents out[0..n-1] and the real frame.
// prepare_fg / fg_params at the bottom are the renderer's side (the inputs).

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstring>
#include "cuda_helper.h"
#include "camera.h"
#include "viewer/primary_hits.h"
#include "viewer/view_matrices.h"
#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_helpers_dlssg_vk.h"

#ifndef DLSS_SNIPPET_DIR
#define DLSS_SNIPPET_DIR L"src/external/dlss/lib"
#endif

struct framegen_vk {
    VkInstance       inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice         dev = VK_NULL_HANDLE;
    VkQueue          queue = VK_NULL_HANDLE;
    uint32_t         qfam = 0;
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;
    VkCommandPool    pool = VK_NULL_HANDLE;
    VkCommandBuffer  cmd = VK_NULL_HANDLE;
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle*    feature = nullptr;
    bool     available = false;
    bool     wanted = false;
    unsigned max_frames = 1;
    int      FW = 0, FH = 0, RW = 0, RH = 0;
    NVSDK_NGX_Result r_last = NVSDK_NGX_Result_Success;

    struct image {
        VkImage        img = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkImageView    view = VK_NULL_HANDLE;
        NVSDK_NGX_Resource_VK res{};
    };

    image depth, mv;                              // render size images
    std::vector<image> out;                       // window size images
    VkImageView backbuffer_view = VK_NULL_HANDLE; // present_vk's frame image
    NVSDK_NGX_Resource_VK backbuffer{};

    struct xbuf {
        VkBuffer       buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        cudaExternalMemory_t ext = nullptr;
        void*          d = nullptr;
        VkDeviceSize   bytes = 0;
    };

    xbuf depth_buf, mv_buf;

    static const std::vector<const char*>& device_extensions() {
        static const std::vector<const char*> e = { 
            VK_NVX_BINARY_IMPORT_EXTENSION_NAME, 
            VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, 
            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME 
        };
        return e;
    }

    void init(VkInstance i, VkPhysicalDevice p, VkDevice d, VkQueue q, uint32_t qf, PFN_vkGetMemoryFdKHR getfd) {
        inst = i; phys = p; dev = d; queue = q; qfam = qf; vkGetMemoryFdKHR = getfd;

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = qfam;
        checkVk(vkCreateCommandPool(dev, &pci, nullptr, &pool));

        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        checkVk(vkAllocateCommandBuffers(dev, &cai, &cmd));

        wchar_t* paths[1] = { (wchar_t*)DLSS_SNIPPET_DIR };
        NVSDK_NGX_FeatureCommonInfo info{};
        info.PathListInfo.Path = paths;
        info.PathListInfo.Length = 1;
        info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
        NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID("a0676bfa-99ea-4d5e-9a1b-3c8f2e1d7b44", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", L"build",
                                                                  inst, phys, dev, vkGetInstanceProcAddr, vkGetDeviceProcAddr, &info);
        if (NVSDK_NGX_FAILED(r)) { fprintf(stderr, "framegen: NGX Vulkan init failed 0x%x\n", (unsigned)r); return; }
        
        NVSDK_NGX_Parameter* caps = nullptr;
        NVSDK_NGX_VULKAN_GetCapabilityParameters(&caps);
        
        int avail = 0;
        NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_FrameGeneration_Available, &avail);
        NVSDK_NGX_Parameter_GetUI(caps, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &max_frames);
        NVSDK_NGX_VULKAN_DestroyParameters(caps);
        available = avail != 0;
        if (available) NVSDK_NGX_VULKAN_AllocateParameters(&params);
    }

    void make_view(VkImage img, VkFormat fmt, VkImageView& view) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        checkVk(vkCreateImageView(dev, &vci, nullptr, &view));
    }

    static void describe(NVSDK_NGX_Resource_VK& res, VkImageView view, VkImage img, VkFormat fmt, int w, int h, bool rw) {
        res.Resource.ImageViewInfo = { view, img, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }, fmt, (unsigned)w, (unsigned)h };
        res.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
        res.ReadWrite = rw;
    }

    void make_image(image& im, int w, int h, VkFormat fmt, VkImageUsageFlags usage) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = { (uint32_t)w, (uint32_t)h, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        checkVk(vkCreateImage(dev, &ici, nullptr, &im.img));

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(dev, im.img, &mr);

        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = memory_type(phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        checkVk(vkAllocateMemory(dev, &mai, nullptr, &im.mem));
        checkVk(vkBindImageMemory(dev, im.img, im.mem, 0));

        make_view(im.img, fmt, im.view);
        describe(im.res, im.view, im.img, fmt, w, h, (usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0);
    }

    void free_image(image& im) {
        if (im.view) vkDestroyImageView(dev, im.view, nullptr);
        if (im.img)  vkDestroyImage(dev, im.img, nullptr);
        if (im.mem)  vkFreeMemory(dev, im.mem, nullptr);
        im = image{};
    }

    void make_xbuf(xbuf& b, VkDeviceSize bytes) {
        b.bytes = bytes;
        VkExternalMemoryBufferCreateInfo ebi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &ebi};
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        checkVk(vkCreateBuffer(dev, &bci, nullptr, &b.buf));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, b.buf, &mr);
        VkExportMemoryAllocateInfo emai{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        emai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &emai};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = memory_type(phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        checkVk(vkAllocateMemory(dev, &mai, nullptr, &b.mem));
        checkVk(vkBindBufferMemory(dev, b.buf, b.mem, 0));
        VkMemoryGetFdInfoKHR gfi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        gfi.memory = b.mem;
        gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        checkVk(vkGetMemoryFdKHR(dev, &gfi, &fd));
        cudaExternalMemoryHandleDesc hd{};
        hd.type = cudaExternalMemoryHandleTypeOpaqueFd;
        hd.handle.fd = fd;
        hd.size = mr.size;
        checkCudaErrors(cudaImportExternalMemory(&b.ext, &hd));
        cudaExternalMemoryBufferDesc bd{};
        bd.size = mr.size;
        checkCudaErrors(cudaExternalMemoryGetMappedBuffer(&b.d, b.ext, &bd));
    }

    void free_xbuf(xbuf& b) {
        if (b.ext) cudaDestroyExternalMemory(b.ext);
        if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
        if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
        b = xbuf{};
    }

    void resize(int w, int h, int rw, int rh, VkImage frame_img) {
        release_feature();

        FW = w; FH = h; RW = rw; RH = rh;
        make_image(depth, RW, RH, VK_FORMAT_R32_SFLOAT,    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        make_image(mv,    RW, RH, VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        
        out.resize(max_frames);
        
        for (image& im : out) make_image(im, FW, FH, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        
        make_view(frame_img, VK_FORMAT_R8G8B8A8_UNORM, backbuffer_view);
        describe(backbuffer, backbuffer_view, frame_img, VK_FORMAT_R8G8B8A8_UNORM, FW, FH, false);
        
        make_xbuf(depth_buf, (VkDeviceSize)RW * RH * sizeof(float));
        make_xbuf(mv_buf,    (VkDeviceSize)RW * RH * sizeof(float2));

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(cmd, &bi));

        NVSDK_NGX_DLSSG_Create_Params cp{ (unsigned)FW, (unsigned)FH, (unsigned)VK_FORMAT_R8G8B8A8_UNORM, (unsigned)RW, (unsigned)RH, false };
        NVSDK_NGX_Result r = NGX_VK_CREATE_DLSSG(cmd, 0, 0, &feature, params, &cp);
        
        checkVk(vkEndCommandBuffer(cmd));
        
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        checkVk(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        
        checkVk(vkQueueWaitIdle(queue));
        
        if (NVSDK_NGX_FAILED(r)) { fprintf(stderr, "framegen: DLSSG create failed 0x%x\n", (unsigned)r); feature = nullptr; }
    }

   
    void evaluate(VkCommandBuffer c, VkImage frame_img, NVSDK_NGX_DLSSG_Opt_Eval_Params& opt, int count) {
        const VkPipelineStageFlags T = VK_PIPELINE_STAGE_TRANSFER_BIT, C = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        
        for (image* im : { &depth, &mv }) {
            barrier(c, im->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, T, 0, T, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkBufferImageCopy region{};
            region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.imageExtent = { (uint32_t)RW, (uint32_t)RH, 1 };
            vkCmdCopyBufferToImage(c, im == &depth ? depth_buf.buf : mv_buf.buf, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            barrier(c, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, T, VK_ACCESS_TRANSFER_WRITE_BIT, C, VK_ACCESS_SHADER_READ_BIT);
        }

        barrier(c, frame_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, T, VK_ACCESS_TRANSFER_READ_BIT, C, VK_ACCESS_SHADER_READ_BIT);
        
        for (int k = 0; k < count; k++) {
            barrier(c, out[k].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, T, 0, C, VK_ACCESS_SHADER_WRITE_BIT);
            
            NVSDK_NGX_VK_DLSSG_Eval_Params ev{};
            ev.pBackbuffer = &backbuffer;
            ev.pDepth = &depth.res;
            ev.pMVecs = &mv.res;
            ev.pOutputInterpFrame = &out[k].res;
            opt.multiFrameCount = count;
            opt.multiFrameIndex = k + 1;
            NVSDK_NGX_Result r = NGX_VK_EVALUATE_DLSSG(c, feature, params, &ev, &opt);

            if (NVSDK_NGX_FAILED(r) && r != r_last) fprintf(stderr, "framegen: evaluate failed (0x%x)\n", (unsigned)r);
            r_last = r;
            barrier(c, out[k].img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, C, VK_ACCESS_SHADER_WRITE_BIT, T, VK_ACCESS_TRANSFER_READ_BIT);
        }
        barrier(c, frame_img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, C, VK_ACCESS_SHADER_READ_BIT, T, VK_ACCESS_TRANSFER_READ_BIT);
    }

    void release_feature() {
        if (feature) { NVSDK_NGX_VULKAN_ReleaseFeature(feature); feature = nullptr; }
        free_image(depth);
        free_image(mv);
        for (image& im : out) free_image(im);
        out.clear();
        if (backbuffer_view) { vkDestroyImageView(dev, backbuffer_view, nullptr); backbuffer_view = VK_NULL_HANDLE; }
        free_xbuf(depth_buf);
        free_xbuf(mv_buf);
    }

    void release() {
        release_feature();
        if (params) { NVSDK_NGX_VULKAN_DestroyParameters(params); params = nullptr; }
        if (pool) { vkDestroyCommandPool(dev, pool, nullptr); pool = VK_NULL_HANDLE; }
        NVSDK_NGX_VULKAN_Shutdown1(dev);
    }
};

__global__ void prepare_fg(primary_hits ph, const float2* flow, real jx, real jy, int n, float* depth, float2* mv) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;

    // hardware depth
    float z = (float)ph.depth[p];
    float zn = kViewNear, zf = kViewFar;
    depth[p] = z >= zf ? 1.f : zf / (zf - zn) - zn * zf / ((zf - zn) * z);

    // motion vector
    float2 f = flow[p];
    mv[p] = make_float2(-f.x - (float)jx, -f.y - (float)jy);
}

inline void fg_params(NVSDK_NGX_DLSSG_Opt_Eval_Params& o, const camera& cam, const camera& prev, bool reset) {
    float v2c[16], c2v[16], v2w[16], w2v[16], v2c_prev[16], c2v_prev[16], v2w_prev[16], w2v_prev[16], t1[16], t2[16];
    view_to_clip(cam, v2c);       clip_to_view(cam, c2v);       view_to_world(cam, v2w);       world_to_view(cam, w2v);
    view_to_clip(prev, v2c_prev); clip_to_view(prev, c2v_prev); view_to_world(prev, v2w_prev); world_to_view(prev, w2v_prev);
    memcpy(o.cameraViewToClip, v2c, sizeof v2c);
    memcpy(o.clipToCameraView, c2v, sizeof c2v);
    mul4(c2v, v2w, t1); mul4(t1, w2v_prev, t2); mul4(t2, v2c_prev, &o.clipToPrevClip[0][0]);   // clip → view → world → prev view → prev clip
    mul4(c2v_prev, v2w_prev, t1); mul4(t1, w2v, t2); mul4(t2, v2c, &o.prevClipToClip[0][0]);   // and back
    
    float I[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    memcpy(o.clipToLensClip, I, sizeof I);

    o.jitterOffset[0] = o.jitterOffset[1] = 0;
    o.mvecScale[0] = o.mvecScale[1] = 1;
    o.cameraPinholeOffset[0] = o.cameraPinholeOffset[1] = 0;

    vec3 fwd = -cam.w;
    for (int k = 0; k < 3; k++) {
        o.cameraPos[k]   = (float)cam.center[k];
        o.cameraUp[k]    = (float)cam.v[k];
        o.cameraRight[k] = (float)cam.u[k];
        o.cameraFwd[k]   = (float)fwd[k];
    }
    o.cameraNear = kViewNear;
    o.cameraFar = kViewFar;
    o.cameraFOV = (float)degrees_to_radians(cam.vfov);
    o.cameraAspectRatio = (float)cam.image_width / (float)cam.image_height;
    o.colorBuffersHDR = false;
    o.depthInverted = false;
    o.cameraMotionIncluded = true;
    o.reset = reset;
    o.automodeOverrideReset = false;
    o.notRenderingGameFrames = false;
    o.orthoProjection = false;
    o.motionVectorsInvalidValue = 1e30f;
    o.motionVectorsDilated = false;
    o.menuDetectionEnabled = false;
}

#endif // VIEWER_FRAMEGEN_VK_H
