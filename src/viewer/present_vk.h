#ifndef VIEWER_PRESENT_VK_H
#define VIEWER_PRESENT_VK_H

#include <vulkan/vulkan.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include "cuda_helper.h"
#include "imgui.h"
#include "imgui_impl_vulkan.h"

#define checkVk(val) check_vk((val), #val, __FILE__, __LINE__)

inline void check_vk(VkResult r, const char* func, const char* file, int line) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "Vulkan error %d at %s:%d '%s'\n", (int)r, file, line, func);
        exit(97);
    }
}

inline uint32_t memory_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "Vulkan: no memory type for 0x%x / 0x%x\n", bits, want);
    exit(97);
}

inline void barrier(VkCommandBuffer c, VkImage img, VkImageLayout from, VkImageLayout to,
                    VkPipelineStageFlags src_stage, VkAccessFlags src, VkPipelineStageFlags dst_stage, VkAccessFlags dst) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    vkCmdPipelineBarrier(c, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

inline void begin_cmd(VkCommandBuffer c) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checkVk(vkBeginCommandBuffer(c, &bi));
}

inline void make_image_2d(VkDevice dev, VkPhysicalDevice phys, int w, int h, VkFormat fmt, VkImageUsageFlags usage, VkImage& img, VkDeviceMemory& mem) {
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
    checkVk(vkCreateImage(dev, &ici, nullptr, &img));

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, img, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memory_type(phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    checkVk(vkAllocateMemory(dev, &mai, nullptr, &mem));
    checkVk(vkBindImageMemory(dev, img, mem, 0));
}

#include "viewer/framegen_vk.h"

struct present_vk {
    SDL_Window*      win = nullptr;
    VkInstance       instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice         dev = VK_NULL_HANDLE;
    uint32_t         qfam = 0;
    VkQueue          queue = VK_NULL_HANDLE;
    VkSurfaceKHR     surface = VK_NULL_HANDLE;
    VkFormat         format = VK_FORMAT_UNDEFINED;
    bool             bgra = false;                   // the frame buffer's channel order
    VkRenderPass     render_pass = VK_NULL_HANDLE;   // ImGui draws onto the blitted swapchain image
    VkCommandPool    pool = VK_NULL_HANDLE;
    VkCommandBuffer  cmd = VK_NULL_HANDLE;
    VkSemaphore      sem_acquire = VK_NULL_HANDLE;
    std::vector<VkSemaphore> sem_done;               // one per swapchain image
    VkFence          fence = VK_NULL_HANDLE;
    bool             recording = false;
    float            ms_present = 0;
    std::atomic<long> presented{0};

    // pacer
    struct pace_set { std::vector<VkImage> imgs; std::chrono::steady_clock::time_point t0; };
    std::thread             pacer;
    std::mutex              mtx, qmtx;
    std::condition_variable cv;
    bool                    pacer_run = false, pacer_busy = false, has_pending = false;
    pace_set                pending;
    std::atomic<bool>       stale{false};            // the pacer hit an out-of-date swapchain
    double                  interval = 0;
    std::chrono::steady_clock::time_point last_handoff{};
    VkSemaphore             sem_render = VK_NULL_HANDLE;
    VkCommandPool           pool_p = VK_NULL_HANDLE;
    VkCommandBuffer         cmd_p = VK_NULL_HANDLE;
    VkFence                 fence_p = VK_NULL_HANDLE;
    VkImage                 present_img = VK_NULL_HANDLE;
    VkDeviceMemory          present_mem = VK_NULL_HANDLE;
    VkImageView             present_view = VK_NULL_HANDLE;
    VkFramebuffer           present_fb = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> out_fbs;
    char             device_name[256] = "";
    bool             same_gpu = false;               // the Vulkan device is the CUDA device (UUID match)
    bool             fg_capable = false;             // the device offers NGX's extensions
    framegen_vk      fg;                             // DLSS frame generation

    // swapchain (window size)
    VkSwapchainKHR             swapchain = VK_NULL_HANDLE;
    VkExtent2D                 extent{};
    std::vector<VkImage>       images;
    std::vector<VkImageView>   views;
    std::vector<VkFramebuffer> framebuffers;

    int            FW = 0, FH = 0, RW = 0, RH = 0;
    VkBuffer       frame_buf = VK_NULL_HANDLE;
    VkDeviceMemory frame_mem = VK_NULL_HANDLE;
    VkImage        frame_img = VK_NULL_HANDLE;
    VkDeviceMemory frame_img_mem = VK_NULL_HANDLE;
    bool                 interop = false;
    cudaExternalMemory_t ext_mem = nullptr;
    uchar4*              d_frame = nullptr;          // interop
    uchar4*              d_rgba = nullptr;           // fallback
    void*                h_frame = nullptr;          // fallback
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;

    void init(SDL_Window* w) {
        win = w;
        // instance
        unsigned n = 0;
        SDL_Vulkan_GetInstanceExtensions(win, &n, nullptr);
        std::vector<const char*> iext(n);
        SDL_Vulkan_GetInstanceExtensions(win, &n, iext.data());
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "RayTracingCUDA Viewer";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        ici.enabledExtensionCount = (uint32_t)iext.size();
        ici.ppEnabledExtensionNames = iext.data();
        checkVk(vkCreateInstance(&ici, nullptr, &instance));

        // surface
        if (!SDL_Vulkan_CreateSurface(win, instance, &surface)) {
            fprintf(stderr, "SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
            exit(97);
        }

        // physical device
        cudaDeviceProp prop; 
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        uint32_t np = 0; 
        vkEnumeratePhysicalDevices(instance, &np, nullptr);
        std::vector<VkPhysicalDevice> pds(np); 
        vkEnumeratePhysicalDevices(instance, &np, pds.data());
        for (VkPhysicalDevice pd : pds) {
            VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
            vkGetPhysicalDeviceProperties2(pd, &p2);
            bool match = memcmp(id.deviceUUID, prop.uuid.bytes, 16) == 0;
            if (phys == VK_NULL_HANDLE) {   // fallback: first device
                phys = pd;
                strncpy(device_name, p2.properties.deviceName, 255);
            }
            if (match) {                    // preferred: CUDA GPU
                same_gpu = true;
                phys = pd;
                strncpy(device_name, p2.properties.deviceName, 255);
                break;
            }
        }

        // queue family
        uint32_t nq = 0; 
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq); 
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qs.data());
        for (uint32_t i = 0; i < nq; i++) {
            VkBool32 present = VK_FALSE; 
            vkGetPhysicalDeviceSurfaceSupportKHR(phys, i, surface, &present);
            if ((qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                qfam = i;
                break;
            }
        }

        // surface format
        uint32_t nf = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &nf, nullptr);
        std::vector<VkSurfaceFormatKHR> fmts(nf); 
        vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &nf, fmts.data());
        format = VK_FORMAT_UNDEFINED;
        for (auto& f : fmts)
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) {
                format = f.format;
                break;
            }
        if (format == VK_FORMAT_UNDEFINED) {
            fprintf(stderr, "Vulkan: the surface offers no 8-bit UNORM format\n");
            exit(97);
        }
        bgra = format == VK_FORMAT_B8G8R8A8_UNORM;

        // logical device
        // device queue
        float prio = 1.f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfam; 
        qci.queueCount = 1; 
        qci.pQueuePriorities = &prio;
        // device extension
        std::vector<const char*> dext = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
        if (same_gpu) {
            dext.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
            dext.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);

            uint32_t ne = 0;
            vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, nullptr);
            std::vector<VkExtensionProperties> have(ne);
            vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, have.data());
            fg_capable = true;
            for (const char* want : framegen_vk::device_extensions()) {
                bool found = false;
                for (auto& e : have) found |= strcmp(e.extensionName, want) == 0;
                fg_capable &= found;
            }
            if (fg_capable) for (const char* e : framegen_vk::device_extensions()) dext.push_back(e);
        }
        // device creation
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1; 
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = (uint32_t)dext.size(); 
        dci.ppEnabledExtensionNames = dext.data();
        checkVk(vkCreateDevice(phys, &dci, nullptr, &dev));
        vkGetDeviceQueue(dev, qfam, 0, &queue);
        if (same_gpu) vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
        if (fg_capable) fg.init(instance, phys, dev, queue, qfam, vkGetMemoryFdKHR);

        // render pass for ImGui
        // attachment
        VkAttachmentDescription att{};
        att.format = format; 
        att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; 
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; 
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; 
        att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        // subpass
        VkSubpassDescription sub{}; 
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; 
        sub.colorAttachmentCount = 1; 
        sub.pColorAttachments = &ref;
        // dependency
        VkSubpassDependency dep{}; 
        dep.srcSubpass = VK_SUBPASS_EXTERNAL; 
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT; 
        dep.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; 
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        // render pass creation
        VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpci.attachmentCount = 1; 
        rpci.pAttachments = &att; 
        rpci.subpassCount = 1; 
        rpci.pSubpasses = &sub; 
        rpci.dependencyCount = 1; 
        rpci.pDependencies = &dep;
        checkVk(vkCreateRenderPass(dev, &rpci, nullptr, &render_pass));

        // command buffer
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; 
        pci.queueFamilyIndex = qfam;
        checkVk(vkCreateCommandPool(dev, &pci, nullptr, &pool));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool; 
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; 
        cai.commandBufferCount = 1;
        checkVk(vkAllocateCommandBuffers(dev, &cai, &cmd));
        // sync objects
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        checkVk(vkCreateSemaphore(dev, &sci, nullptr, &sem_acquire));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        checkVk(vkCreateFence(dev, &fci, nullptr, &fence));

        // the pacer's command buffer
        checkVk(vkCreateCommandPool(dev, &pci, nullptr, &pool_p));
        cai.commandPool = pool_p;
        checkVk(vkAllocateCommandBuffers(dev, &cai, &cmd_p));
        // the pacer's sync objects
        checkVk(vkCreateFence(dev, &fci, nullptr, &fence_p));
        checkVk(vkCreateSemaphore(dev, &sci, nullptr, &sem_render));

        create_swapchain();
        pacer_run = true;
        pacer = std::thread([this] { pacer_loop(); });

        // ImGui
        ImGui_ImplVulkan_InitInfo ii{};
        ii.ApiVersion = VK_API_VERSION_1_1;
        ii.Instance = instance;
        ii.PhysicalDevice = phys;
        ii.Device = dev;
        ii.QueueFamily = qfam;
        ii.Queue = queue;
        ii.DescriptorPoolSize = 8;
        ii.MinImageCount = 2;
        ii.ImageCount = (uint32_t)images.size() + fg.max_frames + 1;
        ii.PipelineInfoMain.RenderPass = render_pass;
        ii.CheckVkResultFn = [](VkResult r) { check_vk(r, "ImGui_ImplVulkan", __FILE__, __LINE__); };
        ImGui_ImplVulkan_Init(&ii);
    }

    // target of tonemap_frame
    uchar4* frame_target() { return interop ? d_frame : d_rgba; }

    void create_swapchain() {
        // capabilities
        VkSurfaceCapabilitiesKHR caps;
        checkVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface, &caps));
        int dw, dh;
        SDL_Vulkan_GetDrawableSize(win, &dw, &dh);
        extent = caps.currentExtent.width != 0xFFFFFFFF ? caps.currentExtent : VkExtent2D{ (uint32_t)dw, (uint32_t)dh };
        uint32_t count = caps.minImageCount + 1;
        if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;
        // swapchain creation
        VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        sci.surface = surface;
        sci.minImageCount = count;
        sci.imageFormat = format;
        sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        sci.imageExtent = extent;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = caps.currentTransform;
        sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        sci.clipped = VK_TRUE;
        sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;   // vsync, the one mode every surface offers; ~1 fps while the window is covered (docs/issues/vulkan-fifo-covered-window.md)
        checkVk(vkCreateSwapchainKHR(dev, &sci, nullptr, &swapchain));
        // images, views, framebuffers
        uint32_t ni = 0;
        vkGetSwapchainImagesKHR(dev, swapchain, &ni, nullptr);
        images.resize(ni);
        vkGetSwapchainImagesKHR(dev, swapchain, &ni, images.data());
        views.resize(ni);
        framebuffers.resize(ni);
        // semaphores
        sem_done.resize(ni);
        VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (uint32_t i = 0; i < ni; i++) checkVk(vkCreateSemaphore(dev, &semci, nullptr, &sem_done[i]));
        
        for (uint32_t i = 0; i < ni; i++) {
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = images[i];
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = format;
            vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            checkVk(vkCreateImageView(dev, &vci, nullptr, &views[i]));
            framebuffers[i] = make_fb(views[i], extent.width, extent.height);
        }
    }

    void destroy_swapchain() {
        for (auto fb : framebuffers) vkDestroyFramebuffer(dev, fb, nullptr);
        for (auto v : views) vkDestroyImageView(dev, v, nullptr);
        for (auto sd : sem_done) vkDestroySemaphore(dev, sd, nullptr);
        sem_done.clear();
        framebuffers.clear();
        views.clear();
        images.clear();
        if (swapchain) vkDestroySwapchainKHR(dev, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }

    // create or recreate the swapchain
    void resize(int fw, int fh, int rw, int rh) {
        drain();
        vkDeviceWaitIdle(dev);
        stale = false;

        vkDestroySemaphore(dev, sem_render, nullptr);
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        checkVk(vkCreateSemaphore(dev, &sci, nullptr, &sem_render));

        destroy_swapchain();
        release_frame();
        create_swapchain();
        FW = fw;
        FH = fh;
        RW = rw;
        RH = rh;
        VkDeviceSize bytes = (VkDeviceSize)FW * FH * 4;

        // frame image
        make_image_2d(dev, phys, FW, FH, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, frame_img, frame_img_mem);

        // frame buffer
        VkMemoryRequirements mr;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        VkExternalMemoryBufferCreateInfo ebi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (same_gpu) bci.pNext = &ebi;
        checkVk(vkCreateBuffer(dev, &bci, nullptr, &frame_buf));
        vkGetBufferMemoryRequirements(dev, frame_buf, &mr);
        VkExportMemoryAllocateInfo emai{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        emai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        mai.allocationSize = mr.size;
        mai.pNext = same_gpu ? &emai : nullptr;
        mai.memoryTypeIndex = memory_type(phys, mr.memoryTypeBits, same_gpu ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                                                      : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        checkVk(vkAllocateMemory(dev, &mai, nullptr, &frame_mem));
        checkVk(vkBindBufferMemory(dev, frame_buf, frame_mem, 0));

        // interop: export the buffer's memory as an fd and map it into CUDA
        interop = same_gpu;
        if (interop) {
            VkMemoryGetFdInfoKHR gfi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
            gfi.memory = frame_mem;
            gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            int fd = -1;
            checkVk(vkGetMemoryFdKHR(dev, &gfi, &fd));
            cudaExternalMemoryHandleDesc hd{};
            hd.type = cudaExternalMemoryHandleTypeOpaqueFd;
            hd.handle.fd = fd;
            hd.size = mr.size;
            checkCudaErrors(cudaImportExternalMemory(&ext_mem, &hd));   // CUDA owns the fd now
            cudaExternalMemoryBufferDesc bd{};
            bd.size = mr.size;
            checkCudaErrors(cudaExternalMemoryGetMappedBuffer((void**)&d_frame, ext_mem, &bd));
        }
        // readback: map the host-visible buffer, give CUDA its own device buffer
        else {
            checkVk(vkMapMemory(dev, frame_mem, 0, bytes, 0, &h_frame));
            checkCudaErrors(cudaMalloc(&d_rgba, bytes));
        }
        if (fg.wanted) { fg.resize(FW, FH, RW, RH, frame_img, format); make_fg_targets(); }
    }

    // enable/disable frame generation
    void enable_fg(bool on) {
        if (on == fg.wanted) return;
        fg.wanted = on;
        drain();
        vkDeviceWaitIdle(dev);
        if (on) { fg.resize(FW, FH, RW, RH, frame_img, format); make_fg_targets(); }
        else    { free_fg_targets(); fg.release_feature(); }
    }

    VkFramebuffer make_fb(VkImageView view, uint32_t w, uint32_t h) {
        VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbi.renderPass = render_pass;
        fbi.attachmentCount = 1;
        fbi.pAttachments = &view;
        fbi.width = w;
        fbi.height = h;
        fbi.layers = 1;
        VkFramebuffer fb;
        checkVk(vkCreateFramebuffer(dev, &fbi, nullptr, &fb));
        return fb;
    }

    void make_fg_targets() {
        free_fg_targets();
        make_image_2d(dev, phys, FW, FH, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, present_img, present_mem);
        fg.make_view(present_img, format, present_view);
        present_fb = make_fb(present_view, FW, FH);
        for (auto& im : fg.out) out_fbs.push_back(make_fb(im.view, FW, FH));
    }

    void free_fg_targets() {
        for (auto fb : out_fbs) vkDestroyFramebuffer(dev, fb, nullptr);
        out_fbs.clear();
        if (present_fb)   vkDestroyFramebuffer(dev, present_fb, nullptr);
        if (present_view) vkDestroyImageView(dev, present_view, nullptr);
        if (present_img)  vkDestroyImage(dev, present_img, nullptr);
        if (present_mem)  vkFreeMemory(dev, present_mem, nullptr);
        present_fb = VK_NULL_HANDLE; present_view = VK_NULL_HANDLE; present_img = VK_NULL_HANDLE; present_mem = VK_NULL_HANDLE;
    }

    // wait until the pacer has shown the previous set
    void drain() {
        std::unique_lock<std::mutex> lk(mtx);   // lock mtx
        cv.wait(lk, [&] { return !pacer_busy && !has_pending; });   // unlock mtx, wait until (!pacer_busy && !has_pending)
    }

    // begin recording after the previous submit finished
    void begin() {
        wait_fence(fence);
        begin_cmd(cmd);
        recording = true;
    }

    void wait_fence(VkFence f) {
        checkVk(vkWaitForFences(dev, 1, &f, VK_TRUE, UINT64_MAX));
        checkVk(vkResetFences(dev, 1, &f));
    }

    bool acquire(uint32_t& idx) {
        return vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX, sem_acquire, VK_NULL_HANDLE, &idx) != VK_ERROR_OUT_OF_DATE_KHR;
    }

    void blit_to_swapchain(VkCommandBuffer c, VkImage src, uint32_t idx) {
        barrier(c, images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.srcOffsets[1] = { FW, FH, 1 };
        blit.dstOffsets[1] = { (int)extent.width, (int)extent.height, 1 };
        vkCmdBlitImage(c, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    }

    VkResult submit_present(VkCommandBuffer c, VkFence f, uint32_t idx, bool wait_render) {
        VkSemaphore waits[2] = { sem_acquire, sem_render };
        VkPipelineStageFlags stages[2] = { VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT };
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = wait_render ? 2 : 1;
        si.pWaitSemaphores = waits;
        si.pWaitDstStageMask = stages;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &sem_done[idx];
        std::lock_guard<std::mutex> lk(qmtx);
        checkVk(vkQueueSubmit(queue, 1, &si, f));
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &sem_done[idx];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &idx;
        VkResult r = vkQueuePresentKHR(queue, &pi);
        presented++;
        return r;
    }

    // the CUDA-written frame → frame_img
    void upload() {
        VkDeviceSize bytes = (VkDeviceSize)FW * FH * 4;
        
        if (interop) checkCudaErrors(cudaStreamSynchronize(0));
        else         checkCudaErrors(cudaMemcpy(h_frame, d_rgba, bytes, cudaMemcpyDeviceToHost));
        
        begin();

        barrier(cmd, frame_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        
        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent = { (uint32_t)FW, (uint32_t)FH, 1 };
        vkCmdCopyBufferToImage(cmd, frame_buf, frame_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        
        barrier(cmd, frame_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    }

    bool show(VkImage src, ImDrawData* ui, VkCommandBuffer c, VkFence f, bool wait_render) {
        uint32_t idx;
        if (!acquire(idx)) {
            checkVk(vkEndCommandBuffer(c));
            std::lock_guard<std::mutex> lk(qmtx);
            checkVk(vkQueueSubmit(queue, 0, nullptr, f));   // the fence was reset for this submit; signal it without work
            return false;
        }
        blit_to_swapchain(c, src, idx);
        if (ui) {                                            // the render pass ends in PRESENT_SRC
            VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rbi.renderPass = render_pass;
            rbi.framebuffer = framebuffers[idx];
            rbi.renderArea.extent = extent;
            vkCmdBeginRenderPass(c, &rbi, VK_SUBPASS_CONTENTS_INLINE);
            { std::lock_guard<std::mutex> lk(qmtx); ImGui_ImplVulkan_RenderDrawData(ui, c); }   // may upload a texture (queue submit + wait)
            vkCmdEndRenderPass(c);
        } else {
            barrier(c, images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
        }
        checkVk(vkEndCommandBuffer(c));
        VkResult r = submit_present(c, f, idx, wait_render);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) return false;
        checkVk(r);
        return true;
    }

    void draw_ui(VkImage img, VkFramebuffer fb, ImDrawData* ui, VkImageLayout from, VkAccessFlags from_access) {
        barrier(cmd, img, from, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, from_access, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);   // the render pass's initialLayout
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = render_pass;
        rbi.framebuffer = fb;
        rbi.renderArea.extent = { (uint32_t)FW, (uint32_t)FH };
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        { std::lock_guard<std::mutex> lk(qmtx); ImGui_ImplVulkan_RenderDrawData(ui, cmd); }
        vkCmdEndRenderPass(cmd);
        barrier(cmd, img, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    }

    void present(ImDrawData* ui, NVSDK_NGX_DLSSG_Opt_Eval_Params* fgp = nullptr, int fg_count = 0) {
        double t0 = SDL_GetPerformanceCounter() * 1000.0 / SDL_GetPerformanceFrequency();
        drain();
        if (stale) { resize(FW, FH, RW, RH); return; }             // the pacer found the swapchain out of date
        upload();
        if (fgp && fg.feature) {
            fg.evaluate(cmd, frame_img, *fgp, fg_count);
            barrier(cmd, present_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            
            VkImageCopy region{};
            region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.extent = { (uint32_t)FW, (uint32_t)FH, 1 };
            vkCmdCopyImage(cmd, frame_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, present_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            
            pace_set set;
            for (int k = 0; k < fg_count; k++) {
                draw_ui(fg.out[k].img, out_fbs[k], ui, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT);
                set.imgs.push_back(fg.out[k].img);
            }
            draw_ui(present_img, present_fb, ui, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT);
            set.imgs.push_back(present_img);

            checkVk(vkEndCommandBuffer(cmd));
            recording = false;

            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &sem_render;
            { std::lock_guard<std::mutex> lk(qmtx); checkVk(vkQueueSubmit(queue, 1, &si, fence)); }

            auto now = std::chrono::steady_clock::now();
            if (last_handoff.time_since_epoch().count()) {
                double dt = std::chrono::duration<double>(now - last_handoff).count();
                interval = interval == 0 ? dt : 0.8 * interval + 0.2 * dt;
            }
            last_handoff = now;
            set.t0 = now;
            { std::lock_guard<std::mutex> lk(mtx); pending = std::move(set); has_pending = true; }
            cv.notify_all();
        } else {
            last_handoff = {};
            interval = 0;
            if (!recording) begin();
            bool ok = show(frame_img, ui, cmd, fence, false);
            recording = false;
            if (!ok) resize(FW, FH, RW, RH);
        }
        ms_present = (float)(SDL_GetPerformanceCounter() * 1000.0 / SDL_GetPerformanceFrequency() - t0);
    }

    void pacer_loop() {
        for (;;) {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&] { return has_pending || !pacer_run; });

            if (!pacer_run) return;

            pace_set set = std::move(pending);
            has_pending = false;
            pacer_busy = true;

            lk.unlock();
            
            int n = (int)set.imgs.size();
            for (int k = 0; k < n; k++) {
                std::this_thread::sleep_until(set.t0 + std::chrono::duration<double>(interval * k / n));
                wait_fence(fence_p);
                begin_cmd(cmd_p);
                if (!show(set.imgs[k], nullptr, cmd_p, fence_p, k == 0)) { stale = true; break; }
            }

            lk.lock();
            pacer_busy = false;
            lk.unlock();
            cv.notify_all();
        }
    }

    void release_frame() {
        if (ext_mem) { cudaDestroyExternalMemory(ext_mem); ext_mem = nullptr; d_frame = nullptr; }
        if (d_rgba)  { cudaFree(d_rgba); d_rgba = nullptr; }
        if (h_frame) { vkUnmapMemory(dev, frame_mem); h_frame = nullptr; }
        if (frame_buf) vkDestroyBuffer(dev, frame_buf, nullptr);
        if (frame_mem) vkFreeMemory(dev, frame_mem, nullptr);
        if (frame_img) vkDestroyImage(dev, frame_img, nullptr);
        if (frame_img_mem) vkFreeMemory(dev, frame_img_mem, nullptr);
        frame_buf = VK_NULL_HANDLE;
        frame_mem = VK_NULL_HANDLE;
        frame_img = VK_NULL_HANDLE;
        frame_img_mem = VK_NULL_HANDLE;
    }

    void release() {
        drain();
        { std::lock_guard<std::mutex> lk(mtx); pacer_run = false; }
        cv.notify_all();
        pacer.join();
        vkDeviceWaitIdle(dev);
        ImGui_ImplVulkan_Shutdown();
        free_fg_targets();
        if (fg_capable) fg.release();
        release_frame();
        destroy_swapchain();
        vkDestroyFence(dev, fence, nullptr);
        vkDestroyFence(dev, fence_p, nullptr);
        vkDestroySemaphore(dev, sem_acquire, nullptr);
        vkDestroySemaphore(dev, sem_render, nullptr);
        vkDestroyCommandPool(dev, pool, nullptr);
        vkDestroyCommandPool(dev, pool_p, nullptr);
        vkDestroyRenderPass(dev, render_pass, nullptr);
        vkDestroyDevice(dev, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
};

#endif // VIEWER_PRESENT_VK_H
