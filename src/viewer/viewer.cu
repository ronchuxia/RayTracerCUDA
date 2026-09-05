// interactive real-time viewer

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <cuda_gl_interop.h>
#include <nvml.h>       // per-process VRAM query
#include <unistd.h>     // getpid()

// Dear ImGui v1.92.8
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl2.h"

#include <cstdio>
#include <ctime>
#include <memory>
#include <vector>

#ifndef RT_IMAGE_WIDTH
#define RT_IMAGE_WIDTH 800
#endif
#ifndef RT_SAMPLES
#define RT_SAMPLES 2
#endif
#ifndef RT_TARGET_SAMPLES
#define RT_TARGET_SAMPLES 8192
#endif
#ifndef RT_MAX_DEPTH
#define RT_MAX_DEPTH 12
#endif
#ifndef RT_SEED
#define RT_SEED 42
#endif
#ifndef RT_SKY
#define RT_SKY 1
#endif
#ifndef VIEWER_SCENE
#define VIEWER_SCENE 0
#endif

#include "camera.h"
#include "physics.h"
#include "viewer/scene.h"
#include "viewer/gbuffer.h"
#include "viewer/denoiser_optix.h"
#include "scenes/scene_utils.h"
#include "viewer/physics_utils.h"
#include "viewer/scenes/primitives.h"
#include "viewer/scenes/ball_pit.h"
#include "viewer/scenes/spin.h"
#include "viewer/scenes/denoise_room.h"

// color/gbuffer -> RGBA8
__global__ void tonemap_frame(const color* accum, gbuffer gb, int view, int rw, int rh,
                              const float3* denoised,
                              uchar4* out, int w, int h, int samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;
    int idx = (j * rh / h) * rw + (i * rw / w);
    real inv = real(1) / samples;
    color c;
    if (view == 1)      c = gb.albedo[idx] * inv;                               // albedo
    else if (view == 2) c = real(0.5) * (gb.normal[idx] * inv + vec3(1, 1, 1)); // normal, [-1,1] -> [0,1]
    else if (denoised)  c = color(denoised[j * w + i].x, denoised[j * w + i].y, denoised[j * w + i].z); // denoised
    else                c = accum[idx] * inv;                                   // beauty
    unsigned char r, g, b;
    tonemap_pixel(c, 1, r, g, b);
    out[j * w + i] = make_uchar4(r, g, b, 255);
}

// frame accumulation
__global__ void accumulate_frame(const camera& cam, int max_depth, const hittable& world,
                                 color* accum, gbuffer gb, curandState* rand_states, int spp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= cam.image_width || j >= cam.image_height) return;

    int pixel_index = j * cam.image_width + i;
    curandState* rand_state = &rand_states[pixel_index];

    for (int sample = 0; sample < spp; ++sample) {
        ray r = cam.get_ray(i, j, rand_state);
        camera::first_hit fh;
        accum[pixel_index] += cam.ray_color(r, world, max_depth, rand_state, &fh);
        gb.albedo[pixel_index] += fh.albedo;
        gb.normal[pixel_index] += fh.normal;
    }
}

// object picking
__global__ void pick(const camera& cam, const hittable& world, int px, int py,
                     curandState* state, int* out_id) {
    ray r = cam.get_ray_through_pixel(px, py);
    hit_record rec;
    *out_id = world.hit(r, interval(real(0.001), infinity), rec, state) ? rec.id : -1;
}

__global__ void initialize_rand_pick(curandState* state, unsigned long seed) {
    curand_init(seed, 0, 0, state);
}

int main() {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init(SDL_INIT_VIDEO) failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    // scene
    scene sc;
#if VIEWER_SCENE == 5
    build_denoise_room_scene(sc);
#elif VIEWER_SCENE == 4
    build_spin_scene(sc);
#elif VIEWER_SCENE == 3
    build_ball_pit_rolling_scene(sc);
#elif VIEWER_SCENE == 2
    build_ball_pit_tight_scene(sc);
#elif VIEWER_SCENE == 1
    build_ball_pit_scene(sc);
#else
    build_primitives_scene(sc);
#endif
    hittable& world = sc.root();

    // camera
    camera* cam;
    checkCudaErrors(cudaMallocManaged((void**)&cam, sizeof(camera)));
    new(cam) camera();

    cam->aspect_ratio      = 16.0 / 9.0;
    cam->image_width       = RT_IMAGE_WIDTH;
    cam->samples_per_pixel = RT_SAMPLES;
    cam->max_depth         = RT_MAX_DEPTH;
    cam->seed              = RT_SEED;
    cam->vfov     = sc.vfov;
    cam->lookfrom = sc.lookfrom;
    cam->lookat   = sc.lookat;
    cam->vup      = vec3(0, 1, 0);
    cam->defocus_angle = 0;
    cam->focus_dist    = 10.0;
    cam->initialize();
    int W = cam->image_width, H = cam->image_height;    // window size
    int RW = W, RH = H;                                 // render size = window / the denoiser's upscale factor

    // SDL window and GL context
    SDL_Window* win = SDL_CreateWindow(
        "RayTracingCUDA Viewer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!win) { fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError()); return 1; }
    SDL_GLContext gl = SDL_GL_CreateContext(win);
    if (!gl) { fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError()); return 1; }

    glewExperimental = GL_TRUE;
    GLenum ge = glewInit();
    if (ge != GLEW_OK) { fprintf(stderr, "glewInit failed: %s\n", glewGetErrorString(ge)); return 1; }

    fprintf(stderr, "viewer: GL vendor: %s\n", (const char*)glGetString(GL_VENDOR));
    fprintf(stderr, "viewer: renderer: %s\n", (const char*)glGetString(GL_RENDERER));

    // Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(win, gl);
    ImGui_ImplOpenGL2_Init();

    // GL texture
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // frame resources
    size_t frame_bytes = 0;
    GLuint pbo = 0;                          // CUDA-GL interop path
    cudaGraphicsResource* cuda_pbo = nullptr;
    bool use_interop = false;
    uchar4* d_rgba = nullptr;                // CPU readback path
    uchar4* h_rgba = nullptr;
    color* accum = nullptr;                  // accumulation buffer
    gbuffer gb;                              // denoiser guides
    curandState* rand_states = nullptr;
    dim3 threads(16, 16), blocks, blocks_out;   // trace grid, tonemap grid

    // rendering configs
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 2048));

    int spp_per_frame = RT_SAMPLES;
    int total_samples = 0;
    int target_samples = RT_TARGET_SAMPLES;
    int view = 0;                            // displayed buffer: 0 beauty, 1 albedo, 2 normal

    // denoiser
    enum { DENOISE_OFF = 0, DENOISE_OPTIX_AOV = 1 };
    int   denoise_mode = DENOISE_OFF;
    auto  upscale_factor = [&]() { return 1; };
    bool  guide_albedo = true;
    bool  guide_normal = true;
    float blend = 0.0f;                      // 0 = fully denoised, 1 = untouched input
    std::unique_ptr<denoiser> dn = std::make_unique<optix_denoiser>(guide_albedo, guide_normal);

    unsigned long rng_seed = (cam->seed < 0) ? (unsigned long)time(0) : (unsigned long)cam->seed;

    auto resize_frame = [&](int w, int h) {
        // release stale frame resources
        checkCudaErrors(cudaDeviceSynchronize());
        if (cuda_pbo) { checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo)); cuda_pbo = nullptr; }
        glDeleteBuffers(1, &pbo); pbo = 0;
        if (d_rgba) { checkCudaErrors(cudaFree(d_rgba)); d_rgba = nullptr; }
        if (h_rgba) { checkCudaErrors(cudaFreeHost(h_rgba)); h_rgba = nullptr; }
        checkCudaErrors(cudaFree(accum));
        checkCudaErrors(cudaFree(rand_states));

        W = w; H = h;
        RW = W / upscale_factor(); RH = H / upscale_factor();
        cam->image_width  = RW;
        cam->aspect_ratio = real(RW) / (real(RH) + real(0.5));
        cam->initialize();
        frame_bytes = (size_t)W * H * 4;
        blocks     = dim3((RW + threads.x - 1) / threads.x, (RH + threads.y - 1) / threads.y);
        blocks_out = dim3((W  + threads.x - 1) / threads.x, (H  + threads.y - 1) / threads.y);
        glViewport(0, 0, W, H);

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

        // present path
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)frame_bytes, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        cudaError_t err = cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);
        use_interop = (err == cudaSuccess);
        if (!use_interop) {
            cudaGetLastError();
            fprintf(stderr, "viewer: CUDA-GL interop unavailable (%s), using CPU readback.\n", cudaGetErrorString(err));
            glDeleteBuffers(1, &pbo);
            pbo = 0;
            checkCudaErrors(cudaMalloc(&d_rgba, frame_bytes));
            checkCudaErrors(cudaMallocHost(&h_rgba, frame_bytes));
        }

        // accumulation buffer
        checkCudaErrors(cudaMalloc(&accum, (size_t)RW * RH * sizeof(color)));
        checkCudaErrors(cudaMemset(accum, 0, (size_t)RW * RH * sizeof(color)));
        gb.allocate(RW, RH);
        dn->setup(RW, RH, W, H);
        total_samples = 0;

        // rng
        checkCudaErrors(cudaMalloc(&rand_states, (size_t)RW * RH * sizeof(curandState)));
        initialize_rand<<<(RW * RH + 255) / 256, 256>>>(*cam, rand_states, rng_seed);
        checkCudaErrors(cudaDeviceSynchronize());
    };
    
    resize_frame(W, H);

    fprintf(stderr, "viewer: rendering: %dx%d frame, %dx%d window, %d spp/frame (target %d), depth %d\n",
            RW, RH, W, H, spp_per_frame, target_samples, cam->max_depth);
    fprintf(stderr, "viewer: presentation: %s\n",
            use_interop ? "CUDA-GL interop" : "CPU readback");
    fprintf(stderr, "viewer: controls: drag orbit, scroll zoom, shift-drag pan, R reset, ESC quit\n");

    // camera orbit state
    point3 target    = cam->lookat;
    vec3   offset    = cam->lookfrom - target;
    double radius    = offset.length();
    double azimuth   = atan2(offset.z(), offset.x());
    double elevation = asin(offset.y() / radius);
    const vec3 world_up = cam->vup;

    // store each object's initial T/R/S
    struct init_trs { vec3 t, r, s; };
    std::vector<init_trs> initial_trs;
    for (int id = 0; id < (int)sc.objects.size(); id++) {
        transform* tr = static_cast<transform*>(sc.get(id)->object);
        initial_trs.push_back({tr->translation, tr->rotation, tr->scale});
    }

    // store inital rendering configs
    const int spp_per_frame0  = spp_per_frame;
    const int target_samples0 = target_samples;
    const int max_depth0      = cam->max_depth;

    // store initial camera orbit state
    const point3 target0 = target;
    const double radius0 = radius, azimuth0 = azimuth, elevation0 = elevation;
    const double vfov0 = cam->vfov;

    // ray-tracing GPU timing events
    cudaEvent_t ev_trace0, ev_trace1;
    checkCudaErrors(cudaEventCreate(&ev_trace0));
    checkCudaErrors(cudaEventCreate(&ev_trace1));
    float ms_trace = 0.0f;
    cudaEvent_t ev_dn0, ev_dn1;
    checkCudaErrors(cudaEventCreate(&ev_dn0));
    checkCudaErrors(cudaEventCreate(&ev_dn1));
    float ms_denoise = 0.0f;

    // panel visibility
    bool show_performance = true, show_rendering = true, show_display = true, show_camera = true, show_object = true, show_physics = true;

    // simulation configs
    bool   playing     = false;
    bool   show_bodies = false;   // collider wireframe overlay
    bool   asleep      = false;
    int    still_steps = 0;
    double phys_accum   = 0.0;
    float  gravity     = -9.8f;
    int    friction_combine    = (int)COMBINE_AVERAGE;
    int    restitution_combine = (int)COMBINE_AVERAGE;
    const double PHYS_DT = 1.0 / 240.0;
    const int    PHYS_MAX_STEPS = 8;
    const real   SLEEP_VEL   = real(0.1);
    const int    SLEEP_STEPS = 60;

    std::vector<phys_body>& bodies = sc.bodies;
    const std::vector<phys_body> initial_bodies = bodies;   // authored snapshot for reset

    std::vector<int> body_of_scene_id((size_t)sc.objects.size(), -1);
    for (int i = 0; i < (int)bodies.size(); i++)
        if (bodies[i].scene_id >= 0) body_of_scene_id[bodies[i].scene_id] = i;

    // object picking state
    int selected_id = -1;
    int press_x = 0, press_y = 0;
    bool maybe_click = false;

    // object picking rng
    curandState* pick_state;
    checkCudaErrors(cudaMalloc(&pick_state, sizeof(curandState)));
    initialize_rand_pick<<<1, 1>>>(pick_state, rng_seed + 1);
    checkCudaErrors(cudaDeviceSynchronize());

    // object picking result
    int* pick_result;
    checkCudaErrors(cudaMallocManaged(&pick_result, sizeof(int)));

    // NVML
    nvmlDevice_t nvml_dev;
    bool nvml_ok = (nvmlInit() == NVML_SUCCESS);
    if (nvml_ok) {
        int cuda_dev; cudaGetDevice(&cuda_dev);
        char pci[32]; cudaDeviceGetPCIBusId(pci, sizeof(pci), cuda_dev);
        nvml_ok = (nvmlDeviceGetHandleByPciBusId(pci, &nvml_dev) == NVML_SUCCESS);
    }
    const unsigned int my_pid = (unsigned int)getpid();
    int vram_used_mb = -1;
    int vram_poll = 0;

    auto reset_accumulation = [&]() {
        checkCudaErrors(cudaMemset(accum, 0, (size_t)RW * RH * sizeof(color)));
        gb.clear();
        total_samples = 0;
    };

    auto sync_body_from_transform = [&](int scene_id) {
        if (scene_id < 0 || body_of_scene_id[scene_id] < 0) return; // not simulated
        transform* tr = static_cast<transform*>(sc.get(scene_id)->object);
        phys_body& b = bodies[body_of_scene_id[scene_id]];
        b.vel   = vec3(0, 0, 0);
        b.omega = vec3(0, 0, 0);
        b.scale = tr->scale;
        quat orient;
        if (b.shape == COLLIDER_SPHERE) sphere_collider_of(tr, b.pos, b.radius, orient, b.offset);
        else                            box_collider_of(tr, b.pos, b.half, orient, b.offset);
        set_orientation(b, orient);
        asleep = false; still_steps = 0;   // a moved body disturbs the pile -> resume stepping
    };

    auto sync_transform_from_body = [&](const phys_body& b) {
        transform* tr = static_cast<transform*>(sc.get(b.scene_id)->object);
        new(tr) transform(tr->child, transform_translation_of(b), quat_to_euler_zyx_degrees(b.orient), b.scale);
    };

    // reset simulation
    auto reset_sim = [&]() {
        // reset scene object transforms
        for (int id = 0; id < (int)sc.objects.size(); id++) {
            const init_trs& in = initial_trs[id];
            transform* tr = static_cast<transform*>(sc.get(id)->object);
            new(tr) transform(tr->child, in.t, in.r, in.s);
        }
        // reset physics bodies
        bodies = initial_bodies;
        sc.refit();
        reset_accumulation();
        phys_accum = 0.0; still_steps = 0; asleep = false; playing = false;
    };

    // rebuild the camera from the camera orbit state
    auto rebuild_camera = [&]() {
        double ce = cos(elevation), se = sin(elevation);
        double ca = cos(azimuth),   sa = sin(azimuth);
        cam->lookfrom = target + radius * vec3(ce * ca, se, ce * sa);
        cam->lookat   = target;
        cam->initialize();
        reset_accumulation();
    };

    // presentation loop
    bool running = true;
    while (running) {
        bool camera_dirty = false;
        bool denoise_dirty = false;

        // mouse and keyboard events
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            ImGuiIO& io = ImGui::GetIO();
            if (e.type == SDL_QUIT) running = false;
            else if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                if (e.window.data1 != W || e.window.data2 != H)
                    resize_frame(e.window.data1, e.window.data2);
            }
            else if (io.WantCaptureKeyboard && (e.type == SDL_KEYDOWN || e.type == SDL_KEYUP)) {
                // typing in the UI
            }
            else if (io.WantCaptureMouse && e.type != SDL_KEYDOWN && e.type != SDL_KEYUP) {
                // hovering/dragging the UI
            }
            else if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) running = false;
            else if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_r) camera_dirty = true;
            else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                press_x = e.button.x; press_y = e.button.y;
                maybe_click = true;
            }
            else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
                // picking
                if (maybe_click && abs(e.button.x - press_x) <= 2 && abs(e.button.y - press_y) <= 2) {
                    pick<<<1, 1>>>(*cam, world, e.button.x * RW / W, e.button.y * RH / H, pick_state, pick_result);
                    checkCudaErrors(cudaDeviceSynchronize());
                    selected_id = *pick_result;
                }
                maybe_click = false;
            }
            else if (e.type == SDL_MOUSEWHEEL) {    
                // zooming
                radius *= pow(0.9, e.wheel.y);
                if (radius < 0.1) radius = 0.1;
                camera_dirty = true;
            }
            else if (e.type == SDL_MOUSEMOTION && (e.motion.state & SDL_BUTTON_LMASK)
                     && !(SDL_GetModState() & KMOD_SHIFT)) {    
                // orbitting
                azimuth   += e.motion.xrel * 0.005;
                elevation += e.motion.yrel * 0.005;
                const double lim = 1.55;    // ~89°, avoid the poles
                if (elevation >  lim) elevation =  lim;
                if (elevation < -lim) elevation = -lim;
                camera_dirty = true;
            }
            else if (e.type == SDL_MOUSEMOTION && (e.motion.state & SDL_BUTTON_LMASK)
                     && (SDL_GetModState() & KMOD_SHIFT)) { 
                // panning
                vec3 fwd   = unit_vector(cam->lookat - cam->lookfrom);
                vec3 right = unit_vector(cross(fwd, world_up));
                vec3 up    = cross(right, fwd);
                double k = radius * 0.002;
                target = target + (-e.motion.xrel * right + e.motion.yrel * up) * k;
                camera_dirty = true;
            }
        }

        // UI
        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // top menu bar
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Quit", "Esc")) running = false;
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Performance", nullptr, &show_performance);
                ImGui::MenuItem("Rendering",   nullptr, &show_rendering);
                ImGui::MenuItem("Display",     nullptr, &show_display);
                ImGui::MenuItem("Physics",     nullptr, &show_physics);
                ImGui::MenuItem("Camera",      nullptr, &show_camera);
                ImGui::MenuItem("Object",      nullptr, &show_object);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        // panels
        ImGui::SetNextWindowPos(ImVec2(10, 30), ImGuiCond_FirstUseEver);
        ImGui::Begin("Renderer");
        {
            ImGuiIO& io = ImGui::GetIO();
            
            bool any_section = false;
            auto section_break = [&]() { if (any_section) ImGui::Separator(); any_section = true; };

            if (show_performance) {
                section_break();
                ImGui::Text("%.0f fps", io.Framerate);
                ImGui::Text("frame   %6.2f ms", io.DeltaTime * 1000.0f);
                ImGui::Text("trace   %6.2f ms", ms_trace);
                ImGui::Text("denoise %6.2f ms", ms_denoise);
                if (nvml_ok && vram_poll++ % 30 == 0) {
                    unsigned int n = 64;
                    nvmlProcessInfo_t procs[64];
                    if (nvmlDeviceGetComputeRunningProcesses(nvml_dev, &n, procs) == NVML_SUCCESS) {
                        vram_used_mb = 0;
                        for (unsigned int k = 0; k < n; k++)
                            if (procs[k].pid == my_pid)
                                vram_used_mb = (int)(procs[k].usedGpuMemory / (1024 * 1024));
                    }
                }
                if (vram_used_mb >= 0) ImGui::Text("vram    %6d MB", vram_used_mb);
                else                   ImGui::Text("vram    %6s MB", "n/a");
            }

            if (show_rendering) {
                section_break();
                ImGui::Text("%d spp", total_samples);
                ImGui::SliderInt("spp", &spp_per_frame, 1, 64);
                ImGui::SliderInt("target spp", &target_samples, 16, 65536, "%d",
                                 ImGuiSliderFlags_Logarithmic);
                if (ImGui::SliderInt("max depth", &cam->max_depth, 1, 50)) camera_dirty = true;
                if (ImGui::Button("Reset rendering")) {
                    spp_per_frame  = spp_per_frame0;
                    target_samples = target_samples0;
                    if (cam->max_depth != max_depth0) camera_dirty = true;
                    cam->max_depth = max_depth0;
                }
            }

            if (show_display) {
                section_break();
                ImGui::Combo("view", &view, "beauty\0albedo\0normal\0");
                if (ImGui::Combo("denoiser", &denoise_mode, "off\0optix aov\0")) {
                    if (RW != W / upscale_factor()) resize_frame(W, H);
                    denoise_dirty = true;
                }
                if (ImGui::SliderFloat("blend", &blend, 0.0f, 1.0f)) denoise_dirty = true;
                bool guides_changed = ImGui::Checkbox("albedo guide", &guide_albedo);
                guides_changed     |= ImGui::Checkbox("normal guide", &guide_normal);
                if (guides_changed) {
                    dn = std::make_unique<optix_denoiser>(guide_albedo, guide_normal);
                    dn->setup(RW, RH, W, H);
                    denoise_dirty = true;
                }
            }

            if (show_physics) {
                section_break();
                if (ImGui::Button("Play"))  playing = true;
                ImGui::SameLine();
                if (ImGui::Button("Pause")) playing = false;
                ImGui::SameLine();
                if (ImGui::Button("Stop"))  reset_sim();
                ImGui::Checkbox("show bodies", &show_bodies);
                if (ImGui::SliderFloat("gravity", &gravity, -30.0f, 0.0f, "%.1f"))
                    { asleep = false; still_steps = 0; }
                static const char* kCombine[] = { "multiply", "min", "geometric", "average", "max" };
                if (ImGui::Combo("friction mix", &friction_combine, kCombine, IM_ARRAYSIZE(kCombine)))
                    { asleep = false; still_steps = 0; }
                if (ImGui::Combo("restitution mix", &restitution_combine, kCombine, IM_ARRAYSIZE(kCombine)))
                    { asleep = false; still_steps = 0; }
            }

            if (show_camera) {
                section_break();
                float pos[3] = {(float)cam->lookfrom.x(), (float)cam->lookfrom.y(), (float)cam->lookfrom.z()};
                if (ImGui::DragFloat3("cam", pos, 0.05f)) {
                    // rigid translate
                    target = target + (point3(pos[0], pos[1], pos[2]) - cam->lookfrom);
                    camera_dirty = true;
                }
                float tgt[3] = {(float)target.x(), (float)target.y(), (float)target.z()};
                if (ImGui::DragFloat3("target", tgt, 0.05f)) {
                    // re-aim
                    point3 nt(tgt[0], tgt[1], tgt[2]);
                    vec3 off = cam->lookfrom - nt;
                    target = nt;
                    radius = off.length();
                    if (radius < 0.1) radius = 0.1;
                    azimuth = atan2(off.z(), off.x());
                    double s = off.y() / radius;
                    elevation = asin(s < -1.0 ? -1.0 : (s > 1.0 ? 1.0 : s));
                    if (elevation >  1.55) elevation =  1.55;
                    if (elevation < -1.55) elevation = -1.55;
                    camera_dirty = true;
                }
                float rad_f = (float)radius;
                if (ImGui::DragFloat("r", &rad_f, 0.05f, 0.1f, 1.0e4f)) {
                    radius = rad_f < 0.1f ? 0.1 : (double)rad_f;
                    camera_dirty = true;
                }
                float vfov_f = (float)cam->vfov;
                if (ImGui::SliderFloat("vfov", &vfov_f, 5.0f, 90.0f, "%.0f deg")) {
                    cam->vfov = vfov_f;
                    camera_dirty = true;
                }
                if (ImGui::Button("Reset camera")) {
                    target = target0; radius = radius0;
                    azimuth = azimuth0; elevation = elevation0;
                    cam->vfov = vfov0;
                    camera_dirty = true;
                }
            }

            if (show_object) {
                section_break();
                if (selected_id >= 0) {
                    ImGui::Text("selected  id %d", selected_id);

                    transform* tr = static_cast<transform*>(sc.get(selected_id)->object);
                    float t[3] = {(float)tr->translation.x(), (float)tr->translation.y(), (float)tr->translation.z()};
                    float r[3] = {(float)tr->rotation.x(),    (float)tr->rotation.y(),    (float)tr->rotation.z()};
                    float s[3] = {(float)tr->scale.x(),       (float)tr->scale.y(),       (float)tr->scale.z()};
                    bool edited = false;
                    edited |= ImGui::DragFloat3("translate", t, 0.05f);
                    edited |= ImGui::DragFloat3("rotate",    r, 1.0f);
                    int bi = body_of_scene_id[selected_id];
                    if (bi >= 0 && bodies[bi].shape == COLLIDER_SPHERE) {
                        // a sphere collider needs uniform scale
                        if (ImGui::DragFloat("scale", &s[0], 0.02f, 0.01f, 100.0f)) {
                            s[1] = s[2] = s[0];
                            edited = true;
                        }
                    } else {
                        edited |= ImGui::DragFloat3("scale", s, 0.02f, 0.01f, 100.0f);
                    }
                    if (edited) {
                        for (int c = 0; c < 3; c++) s[c] = fmaxf(s[c], 0.01f);
                        new(tr) transform(tr->child, point3(t[0], t[1], t[2]), vec3(r[0], r[1], r[2]), vec3(s[0], s[1], s[2]));
                        sc.refit();
                        reset_accumulation();
                        sync_body_from_transform(selected_id);
                    }
                    if (ImGui::Button("Reset transform")) {
                        const init_trs& in = initial_trs[selected_id];
                        new(tr) transform(tr->child, in.t, in.r, in.s);
                        sc.refit();
                        reset_accumulation();
                        sync_body_from_transform(selected_id);
                    }

                    // collision type and motion type
                    if (bi >= 0) {
                        phys_body& b = bodies[bi];
                        ImGui::Spacing();
                        // collision type
                        if (ImGui::Checkbox("collidable", &b.collidable)) {
                            asleep = false; still_steps = 0;
                        }
                        // motion type
                        static const char* kMotion[] = { "static", "kinematic", "dynamic" };
                        int m = (int)b.motion;
                        if (ImGui::Combo("motion", &m, kMotion, IM_ARRAYSIZE(kMotion))) {
                            b.motion = (motion_type)m;
                            if (b.motion != DYNAMIC) { b.vel = vec3(0, 0, 0); b.omega = vec3(0, 0, 0); }
                            asleep = false; still_steps = 0;
                        }
                        // mass of dynamic object
                        if (b.motion == DYNAMIC) {
                            float mass_f = (float)b.mass;
                            if (ImGui::DragFloat("mass", &mass_f, 0.05f, 0.01f, 1000.0f, "%.2f")) {
                                b.mass = real(fmaxf(mass_f, 0.01f));
                                asleep = false; still_steps = 0;
                            }
                        }
                        // surface properties
                        float fr = (float)b.friction, rest = (float)b.restitution;
                        if (ImGui::SliderFloat("friction", &fr, 0.0f, 2.0f, "%.2f")) {
                            b.friction = real(fr);
                            asleep = false; still_steps = 0;
                        }
                        if (ImGui::SliderFloat("restitution", &rest, 0.0f, 1.0f, "%.2f")) {
                            b.restitution = real(rest);
                            asleep = false; still_steps = 0;
                        }
                        float roll = (float)b.rolling_friction;
                        float spin = (float)b.spinning_friction;
                        if (ImGui::SliderFloat("rolling", &roll, 0.0f, 0.1f, "%.3f")) {
                            b.rolling_friction = real(roll);
                            asleep = false; still_steps = 0;
                        }
                        if (ImGui::SliderFloat("spinning", &spin, 0.0f, 0.1f, "%.3f")) {
                            b.spinning_friction = real(spin);
                            asleep = false; still_steps = 0;
                        }
                        if (b.shape == COLLIDER_SPHERE && inv_mass(b) > real(0))
                            ImGui::Text("spin      %.2f rad/s", (double)b.omega.length());
                    } else {
                        ImGui::TextDisabled("not simulated");
                    }
                } else {
                    ImGui::Text("selected  none");
                }
            }
        }
        ImGui::End();

        if (camera_dirty) rebuild_camera();

        // step physics
        if (playing && !asleep && !bodies.empty()) {
            const phys_params pp{ real(gravity), (combine_mode)friction_combine,
                                                 (combine_mode)restitution_combine };
            // advance timeline
            phys_accum += ImGui::GetIO().DeltaTime;
            const double cap = PHYS_DT * PHYS_MAX_STEPS;
            if (phys_accum > cap) phys_accum = cap;     // spiral-of-death clamp
            bool stepped = false;
            while (phys_accum >= PHYS_DT) {
                real maxv = physics_step(bodies, pp, real(PHYS_DT));
                if (maxv < SLEEP_VEL) still_steps++; else still_steps = 0;
                phys_accum -= PHYS_DT;
                stepped = true;
            }
            if (still_steps > SLEEP_STEPS) asleep = true;
            // update transform of scene objects
            if (stepped) {
                for (phys_body& b : bodies) {
                    if (b.scene_id < 0 || b.motion != DYNAMIC) continue;
                    sync_transform_from_body(b);
                }
                sc.refit();
                reset_accumulation();
            }
        }

        // overlays: world-space line segments projected through the camera
        const float sx = (float)W / RW, sy = (float)H / RH;
        auto draw_line = [&](const point3& a, const point3& b, ImU32 col, float width) {
            real ax, ay, bx, by;
            if (cam->world_to_pixel(a, ax, ay) && cam->world_to_pixel(b, bx, by))
                ImGui::GetForegroundDrawList()->AddLine(ImVec2((float)ax * sx, (float)ay * sy),
                                                        ImVec2((float)bx * sx, (float)by * sy), col, width);
        };
        static const int edge[12][2] = {{0,1},{0,2},{0,4},{1,3},{1,5},{2,3},
                                        {2,6},{3,7},{4,5},{4,6},{5,7},{6,7}};

        // selected object's bbox
        if (selected_id >= 0) {
            aabb bb = sc.get(selected_id)->bounding_box();
            point3 corner[8];
            for (int k = 0; k < 8; k++)
                corner[k] = point3(k & 1 ? bb.x.max : bb.x.min,
                                   k & 2 ? bb.y.max : bb.y.min,
                                   k & 4 ? bb.z.max : bb.z.min);
            for (int k = 0; k < 12; k++)
                draw_line(corner[edge[k][0]], corner[edge[k][1]], IM_COL32(255, 220, 0, 255), 1.5f);
        }

        // physics bodies
        if (show_bodies) {
            const ImU32 col = IM_COL32(80, 220, 120, 255);
            for (const phys_body& b : bodies) {
                if (b.shape == COLLIDER_BOX) {
                    point3 corner[8];
                    for (int k = 0; k < 8; k++)
                        corner[k] = b.pos + b.axes[0] * (k & 1 ? b.half.x() : -b.half.x())
                                          + b.axes[1] * (k & 2 ? b.half.y() : -b.half.y())
                                          + b.axes[2] * (k & 4 ? b.half.z() : -b.half.z());
                    for (int k = 0; k < 12; k++)
                        draw_line(corner[edge[k][0]], corner[edge[k][1]], col, 1.0f);
                } else {
                    const int N = 32;
                    for (int i = 0; i < 3; i++) {
                        const vec3& u = b.axes[(i + 1) % 3];
                        const vec3& v = b.axes[(i + 2) % 3];
                        point3 prev = b.pos + u * b.radius;
                        for (int k = 1; k <= N; k++) {
                            real t = real(2) * real(3.14159265358979323846) * real(k) / real(N);
                            point3 p = b.pos + (u * cos(t) + v * sin(t)) * b.radius;
                            draw_line(prev, p, col, 1.0f);
                            prev = p;
                        }
                    }
                }
            }
        }

        // frame accumulation
        bool did_accumulate = false;
        if (total_samples < target_samples) {
            checkCudaErrors(cudaEventRecord(ev_trace0));
            accumulate_frame<<<blocks, threads>>>(*cam, cam->max_depth, world, accum, gb, rand_states, spp_per_frame);
            checkCudaErrors(cudaEventRecord(ev_trace1));
            did_accumulate = true;
            total_samples += spp_per_frame;
        }

        // denoise
        bool did_denoise = false;
        if (denoise_mode != DENOISE_OFF && (denoise_dirty || did_accumulate)) {
            checkCudaErrors(cudaEventRecord(ev_dn0));
            dn->invoke(accum, gb, total_samples, blend);
            checkCudaErrors(cudaEventRecord(ev_dn1));
            did_denoise = true;
            denoise_dirty = false;
        }
        const float3* denoised = denoise_mode != DENOISE_OFF ? dn->output : nullptr;

        // presentation
        if (use_interop) {  
            // CUDA-GL interop
            uchar4* dptr = nullptr;
            size_t nbytes = 0;
            checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo, 0));
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &nbytes, cuda_pbo));
            tonemap_frame<<<blocks_out, threads>>>(accum, gb, view, RW, RH, denoised, dptr, W, H, total_samples);
            checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo, 0));   // syncs the stream

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        } else {    
            // CPU readback
            tonemap_frame<<<blocks_out, threads>>>(accum, gb, view, RW, RH, denoised, d_rgba, W, H, total_samples);
            checkCudaErrors(cudaMemcpy(h_rgba, d_rgba, frame_bytes, cudaMemcpyDeviceToHost));

            glBindTexture(GL_TEXTURE_2D, tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, h_rgba);
        }

        if (did_accumulate) {
            checkCudaErrors(cudaEventSynchronize(ev_trace1));
            checkCudaErrors(cudaEventElapsedTime(&ms_trace, ev_trace0, ev_trace1));
        } else {
            ms_trace = 0.0f;
        }
        if (did_denoise) {
            checkCudaErrors(cudaEventSynchronize(ev_dn1));
            checkCudaErrors(cudaEventElapsedTime(&ms_denoise, ev_dn0, ev_dn1));
        } else {
            ms_denoise = 0.0f;
        }

        // draw a fullscreen textured quad
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex2f(-1,  1);
            glTexCoord2f(1, 0); glVertex2f( 1,  1);
            glTexCoord2f(1, 1); glVertex2f( 1, -1);
            glTexCoord2f(0, 1); glVertex2f(-1, -1);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        // UI
        ImGui::Render();
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

        // present the rendered back buffer to the window
        SDL_GL_SwapWindow(win);
    }

    // cleanup
    if (nvml_ok) nvmlShutdown();
    cudaFree(pick_state);
    cudaFree(pick_result);
    cudaEventDestroy(ev_trace0);
    cudaEventDestroy(ev_trace1);
    cudaEventDestroy(ev_dn0);
    cudaEventDestroy(ev_dn1);
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    if (d_rgba) cudaFree(d_rgba);
    if (h_rgba) cudaFreeHost(h_rgba);
    cudaFree(accum);
    gb.release();
    cudaFree(rand_states);
    cudaFree(cam);
    sc.release();
    if (cuda_pbo) cudaGraphicsUnregisterResource(cuda_pbo);
    if (pbo) glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    SDL_GL_DeleteContext(gl);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
