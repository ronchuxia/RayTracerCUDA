// viewer.cu — interactive real-time viewer (roadmap workstreams B1–B3).
//
// A separate executable/translation unit from the offline renderer (main.cu):
// it opens a window, builds a persistent scene, renders it on the GPU, and
// presents it through one of two paths:
//   - CUDA<->GL interop (fast path): a Pixel Buffer Object shared between CUDA
//     and GL; requires the GL context to live on the same NVIDIA GPU as CUDA.
//   - CPU readback (portable path): tonemap into a device buffer, cudaMemcpy to
//     pinned host memory, glTexSubImage2D from the host pointer. Works on any
//     GL context — including software GL (llvmpipe) over VNC / `ssh -X`, where
//     interop registration fails because GL isn't on the NVIDIA GPU.
// The path is chosen at startup: interop is attempted and, if registration
// fails (e.g. software GL over VNC), the viewer falls back to readback.
//
// `./build/viewer --headless` skips SDL/GL entirely: it accumulates the same
// image through the readback pipeline and writes build/viewer_headless.ppm, so
// the full CUDA path can run (and be verified) on a box with no display at all.
// Runtime flags: --spp N (samples added per frame, default RT_SAMPLES) and
// --frames N (headless: frames to accumulate, default RT_FRAMES).
// The offline path and tests are untouched.
//
// B1 was the display + render loop; B2 adds progressive accumulation: each
// loop iteration adds a few samples per pixel into a persistent accumulator
// and presents the running average, so the image refines over time while the
// app stays responsive. R resets the accumulation; past RT_TARGET_SAMPLES the
// viewer stops adding and just presents. Because each pixel's cuRAND state
// persists across launches, K frames of M spp consume the same RNG stream —
// and add in the same order — as one K*M-spp render, so the accumulated image
// is byte-identical to the single-shot one (tests/run_tests.sh stage [7/7]).
//
// B3 makes the camera interactive: an orbit/arcball model around a target
// point — left-drag orbits (azimuth/elevation), scroll wheel zooms (radius),
// shift-left-drag pans the target. Any camera change re-runs camera::initialize()
// and resets the accumulation (reusing B2's reset), so the view re-converges
// from the new angle. Headless mode uses the default camera (no input).
//
// A Dear ImGui panel (vendored v1.92.8, SDL2 + fixed-function GL2 backends)
// adds live controls on top: spp/frame and target-spp sliders (pacing only —
// no reset), max-depth and vfov sliders (change what's rendered — restart
// accumulation), restart/reset-camera buttons, and fps/spp/path readouts.
// When the panel has the mouse or keyboard (io.WantCapture*), viewer input
// (orbit/zoom/pan/hotkeys) is suppressed so the two never fight.
//
// Needs a display (local, VNC, or X-forward) plus SDL2 + GLEW + OpenGL dev libs.
// Build (match -arch to your GPU; see scripts/build_viewer.sh):
//   nvcc src/viewer/viewer.cu -o build/viewer -std=c++14 -arch=sm_86 -Isrc \
//        -lSDL2 -lGLEW -lGL
//   ./build/viewer            # ESC or close the window to quit

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <cuda_gl_interop.h>
#include <nvml.h>       // per-process VRAM query (links -lnvidia-ml)
#include <unistd.h>     // getpid()

// Dear ImGui (vendored in src/external/imgui, pinned v1.92.8): UI panel with
// live render controls. SDL2 + fixed-function GL2 backends — the GL2 one
// matches the GL 2.1 compatibility context this viewer creates.
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

// Viewer defaults (all -D-overridable). Low sample count for interactivity;
// sky background so the classic book scene is lit without an emitter.
#ifndef RT_IMAGE_WIDTH
#define RT_IMAGE_WIDTH 800
#endif
#ifndef RT_SAMPLES
#define RT_SAMPLES 2        // samples added per frame (the accumulation step)
#endif
#ifndef RT_FRAMES
#define RT_FRAMES 8         // --headless: frames to accumulate (8 x 2 = 16 spp default)
#endif
#ifndef RT_TARGET_SAMPLES
#define RT_TARGET_SAMPLES 8192   // windowed: stop accumulating past this many spp
#endif
#ifndef RT_MAX_DEPTH
#define RT_MAX_DEPTH 12
#endif
#ifndef RT_SEED
#define RT_SEED 42
#endif
#ifndef RT_SKY
#define RT_SKY 1            // viewer lights the scene with the sky gradient
#endif

#include "camera.h"
#include "scene.h"
#include "scenes/scene_utils.h"

// --- device tonemap: accumulator (sum of samples) -> RGBA8 -------------------
// Per pixel, calls color.h's shared tonemap_pixel (the same routine the offline
// PPM writer uses), so a viewer pixel is byte-identical to the offline one.
__global__ void tonemap_frame(const color* accum, uchar4* out, int w, int h, int samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;
    int idx = j * w + i;
    unsigned char r, g, b;
    tonemap_pixel(accum[idx], samples, r, g, b);
    out[idx] = make_uchar4(r, g, b, 255);
}

// --- B2 accumulation pass: add `spp` fresh samples per pixel -----------------
// The same per-sample pattern as camera.h::render_pixel, but WITHOUT the
// initial clear, so consecutive launches extend the running sum in accum.
// Each pixel's cuRAND state advances across launches and the += order matches
// render_pixel's, so K launches of M spp produce an accumulator byte-identical
// to a single K*M-spp launch.
__global__ void accumulate_frame(const camera& cam, int max_depth, const hittable& world,
                                 color* accum, curandState* rand_states, int spp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= cam.image_width || j >= cam.image_height) return;

    int pixel_index = j * cam.image_width + i;
    curandState* rand_state = &rand_states[pixel_index];

    for (int sample = 0; sample < spp; ++sample) {
        ray r = cam.get_ray(i, j, rand_state);
        accum[pixel_index] += cam.ray_color(r, world, max_depth, rand_state);
    }
}

// --- B4 object picking ---------------------------------------------------
// On click, one thread casts the deterministic ray through the clicked pixel
// center (camera::get_ray_through_pixel) and reports hit_record.id — the stable
// scene-object id stamped by the outermost tagged wrapper. The pick uses a
// DEDICATED cuRAND state: hitting a stochastic hittable (constant_medium)
// must never consume the render states, or accumulation determinism breaks.
__global__ void pick(const camera& cam, const hittable& world, int px, int py,
                     curandState* state, int* out_id) {
    ray r = cam.get_ray_through_pixel(px, py);
    hit_record rec;
    *out_id = world.hit(r, interval(real(0.001), infinity), rec, state) ? rec.id : -1;
}

__global__ void initialize_rand_pick(curandState* state, unsigned long seed) {
    curand_init(seed, 0, 0, state);
}

// --- scene: checker ground + editable objects, sky-lit. Every editable object
// (spheres, a box, a triangle — one of each prim type) is registered as a
// transform(prim), so B5 can drive full T/R/S on any of them uniformly. The
// underlying prims are UNIT and centred at the origin; the transform supplies
// position (translation) and size (scale). The ground stays a plain sphere —
// it's the floor, not something you manipulate. Ids are registration order.
static void build_viewer_scene(scene& sc) {
    sc.init();

    material* ground  = new_lambertian(
        make_checker(0.32, color(.2, .3, .1), color(.9, .9, .9)), sc.allocs);
    material* diffuse = new_lambertian(color(0.4, 0.2, 0.1), sc.allocs);
    material* glass   = new_dielectric(1.5, sc.allocs);
    material* metal_m = new_metal(color(0.7, 0.6, 0.5), 0.0, sc.allocs);
    material* box_mat = new_lambertian(color(0.2, 0.4, 0.7), sc.allocs);
    material* tri_mat = new_lambertian(color(0.9, 0.75, 0.2), sc.allocs);

    sc.add(make_sphere(point3(0, -1000, 0), 1000, ground, sc.allocs));   // id 0: floor (plain)

    // Editable objects: unit prim at origin + transform(T, R°, S).
    sc.add(new_transform(make_sphere(point3(0,0,0), 1.0, diffuse, sc.allocs),
                         vec3(-4, 1, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));   // id 1: diffuse sphere
    sc.add(new_transform(make_sphere(point3(0,0,0), 1.0, glass, sc.allocs),
                         vec3(0, 1, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));    // id 2: glass sphere
    sc.add(new_transform(make_sphere(point3(0,0,0), 1.0, metal_m, sc.allocs),
                         vec3(4, 1, 0), vec3(0,0,0), vec3(1,1,1), sc.allocs));    // id 3: metal sphere
    sc.add(new_transform(new_box(point3(-0.6,-0.6,-0.6), point3(0.6,0.6,0.6), box_mat, sc.allocs, sc.list_dtors),
                         vec3(-2, 0.6, -3), vec3(0, 35, 0), vec3(1,1,1), sc.allocs));  // id 4: box (composite)
    sc.add(new_transform(make_triangle(point3(-0.8,-0.6,0), point3(0.8,-0.6,0), point3(0,0.9,0),
                                       vec3(0,0,1), tri_mat, sc.allocs),
                         vec3(2, 1.3, -3), vec3(0,0,0), vec3(1,1,1), sc.allocs));      // id 5: triangle

    sc.build();
}

int main(int argc, char** argv) {
    bool headless = false;
    int frames = RT_FRAMES;          // --frames N: headless accumulation count
    int spp_per_frame = RT_SAMPLES;  // --spp N: samples added per frame
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--headless") == 0) headless = true;
        else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--spp") == 0 && i + 1 < argc) spp_per_frame = atoi(argv[++i]);
    }
    if (frames < 1) frames = 1;
    if (spp_per_frame < 1) spp_per_frame = 1;

    // ---- window + GL context (skipped entirely in --headless mode) ----
    SDL_Window* win = nullptr;
    SDL_GLContext gl = nullptr;
    if (headless) {
        fprintf(stderr, "viewer: headless mode — accumulating %d frame(s) x %d spp to build/viewer_headless.ppm\n",
                frames, spp_per_frame);
    } else {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            fprintf(stderr,
                    "SDL_Init(VIDEO) failed: %s\n"
                    "The viewer needs a display. On a headless box, run it over VNC or\n"
                    "an X-forwarded session (ssh -X) so DISPLAY is set — or use --headless\n"
                    "to render a PPM frame with no display.\n",
                    SDL_GetError());
            return 1;
        }
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);   // compatibility profile:
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);   // immediate-mode fullscreen quad
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    }

    // ---- camera: compute image_height before sizing the window/buffers ----
    // In MANAGED memory, like the offline scenes: the render kernels take the
    // camera by reference (i.e. by pointer), so the device dereferences it —
    // a host-stack camera would be an illegal address on the GPU.
    camera* cam;
    checkCudaErrors(cudaMallocManaged((void**)&cam, sizeof(camera)));
    new(cam) camera();
    cam->aspect_ratio      = 16.0 / 9.0;
    cam->image_width       = RT_IMAGE_WIDTH;
    cam->samples_per_pixel = spp_per_frame;   // per accumulation pass (B2)
    cam->max_depth         = RT_MAX_DEPTH;
    cam->seed              = RT_SEED;
    cam->vfov     = 20;
    cam->lookfrom = point3(13, 2, 3);
    cam->lookat   = point3(0, 0, 0);
    cam->vup      = vec3(0, 1, 0);
    cam->defocus_angle = 0;
    cam->focus_dist    = 10.0;
    cam->initialize();                   // fills image_height + camera frame
    const int W = cam->image_width, H = cam->image_height;

    GLuint tex = 0;
    if (!headless) {
        win = SDL_CreateWindow(
            "RayTracingCUDA — viewer",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H, SDL_WINDOW_OPENGL);
        if (!win) { fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError()); return 1; }
        gl = SDL_GL_CreateContext(win);
        if (!gl) { fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError()); return 1; }

        glewExperimental = GL_TRUE;
        GLenum ge = glewInit();
        if (ge != GLEW_OK) { fprintf(stderr, "glewInit failed: %s\n", glewGetErrorString(ge)); return 1; }

        // Which GPU (or software rasterizer) is serving this GL context? Over
        // VNC/`ssh -X` this is typically llvmpipe/Mesa — the tell that CUDA-GL
        // interop can't work and the readback path will be used.
        fprintf(stderr, "viewer: GL vendor: %s | renderer: %s\n",
                (const char*)glGetString(GL_VENDOR), (const char*)glGetString(GL_RENDERER));

        // ---- Dear ImGui: context + SDL2/GL2 backends ----
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplSDL2_InitForOpenGL(win, gl);
        ImGui_ImplOpenGL2_Init();

        // ---- GL texture (both present paths upload into this) ----
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }

    // ---- choose the present path: CUDA-GL interop, else CPU readback ----
    // Try interop first; on failure (or in headless mode, which has no GL) fall
    // back to the readback pipeline.
    const size_t frame_bytes = (size_t)W * H * 4;
    GLuint pbo = 0;
    cudaGraphicsResource* cuda_pbo = nullptr;
    bool use_interop = false;

    if (!headless) {
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)frame_bytes, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Attempt registration WITHOUT checkCudaErrors: failure here is an
        // expected condition (GL context not on the NVIDIA GPU), not a bug.
        cudaError_t err = cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);
        if (err == cudaSuccess) {
            use_interop = true;
        } else {
            cudaGetLastError();   // clear the error so later CUDA calls aren't poisoned
            fprintf(stderr, "viewer: CUDA-GL interop unavailable (%s) — the GL context "
                            "is probably not on the NVIDIA GPU; using CPU readback.\n",
                    cudaGetErrorString(err));
            glDeleteBuffers(1, &pbo);
            pbo = 0;
        }
    }

    // Readback staging buffers: device tonemap target + pinned host copy.
    uchar4* d_rgba = nullptr;
    uchar4* h_rgba = nullptr;
    if (!use_interop) {
        if (!headless) fprintf(stderr, "viewer: using CPU-readback present path.\n");
        checkCudaErrors(cudaMalloc(&d_rgba, frame_bytes));
        checkCudaErrors(cudaMallocHost(&h_rgba, frame_bytes));
    }

    // ---- build scene + accumulation buffers ----
    // Raise the device stack limit like the offline scenes do: the recursive
    // hittable dispatch (world BVH → shapes) can exceed the 1 KB default.
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 2048));

    scene sc;
    build_viewer_scene(sc);
    hittable& world = sc.root();

    // Snapshot each editable object's initial T/R/S so the Selection panel's
    // "Reset transform" button (B5) can restore it. Indexed by scene id; a
    // non-transform object records editable=false. Host-side only — the initial
    // pose is editor state, kept out of the device `transform` struct.
    struct init_trs { bool editable; vec3 t, r, s; };
    std::vector<init_trs> initial_trs;
    for (int id = 0; id < (int)sc.objects.size(); id++) {
        hittable* h = sc.get(id);
        if (h && h->type == TRANSFORM) {
            transform* tr = static_cast<transform*>(h->object);
            initial_trs.push_back({true, tr->translation, tr->rotation, tr->scale});
        } else {
            initial_trs.push_back({false, vec3(), vec3(), vec3()});
        }
    }

    curandState* rand_states;
    color* accum;
    checkCudaErrors(cudaMalloc(&rand_states, (size_t)W * H * sizeof(curandState)));
    checkCudaErrors(cudaMalloc(&accum,       (size_t)W * H * sizeof(color)));

    unsigned long rng_seed = (cam->seed < 0) ? (unsigned long)time(0) : (unsigned long)cam->seed;
    initialize_rand<<<(W * H + 255) / 256, 256>>>(*cam, rand_states, rng_seed);
    checkCudaErrors(cudaDeviceSynchronize());

    // B2: the accumulator starts at zero; every accumulate_frame launch adds
    // spp_per_frame samples per pixel on top of it.
    checkCudaErrors(cudaMemset(accum, 0, (size_t)W * H * sizeof(color)));
    long long total_samples = 0;                 // per-pixel samples accumulated so far
    int target_samples = RT_TARGET_SAMPLES;      // runtime-adjustable in the UI panel

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    fprintf(stderr, "viewer: %dx%d, %d spp/frame (target %d), depth %d — present: %s\n",
            W, H, spp_per_frame, target_samples, cam->max_depth,
            headless    ? "headless PPM dump"
          : use_interop ? "CUDA-GL interop — drag orbit, scroll zoom, shift-left-drag pan, R reset, ESC quit"
                        : "CPU readback — drag orbit, scroll zoom, shift-left-drag pan, R reset, ESC quit");

    // ---- headless: accumulate `frames` passes, tonemap, write PPM, exit ----
    if (headless) {
        for (int f = 0; f < frames; f++)
            accumulate_frame<<<blocks, threads>>>(*cam, cam->max_depth, world, accum, rand_states, spp_per_frame);
        checkCudaErrors(cudaDeviceSynchronize());
        total_samples = (long long)frames * spp_per_frame;

        tonemap_frame<<<blocks, threads>>>(accum, d_rgba, W, H, (int)total_samples);
        checkCudaErrors(cudaMemcpy(h_rgba, d_rgba, frame_bytes, cudaMemcpyDeviceToHost));

        const char* out_path = "build/viewer_headless.ppm";
        FILE* f = fopen(out_path, "w");
        if (!f) { fprintf(stderr, "viewer: cannot open %s for writing\n", out_path); return 1; }
        fprintf(f, "P3\n%d %d\n255\n", W, H);
        for (int j = 0; j < H; j++)
            for (int i = 0; i < W; i++) {
                uchar4 p = h_rgba[j * W + i];
                fprintf(f, "%d %d %d\n", p.x, p.y, p.z);
            }
        fclose(f);
        fprintf(stderr, "viewer: wrote %s\n", out_path);

        cudaFree(d_rgba);
        cudaFreeHost(h_rgba);
        cudaFree(accum);
        cudaFree(rand_states);
        cudaFree(cam);
        sc.release();
        return 0;
    }

    // ---- B3: orbit-camera state (spherical coords around a target point) ----
    // offset = lookfrom - target = radius * (cosφ cosθ, sinφ, cosφ sinθ).
    point3 target    = cam->lookat;
    vec3   offset    = cam->lookfrom - target;
    double radius    = offset.length();
    double azimuth   = atan2(offset.z(), offset.x());
    double elevation = asin(offset.y() / radius);
    const vec3 world_up = cam->vup;

    // Initial view, restored by the UI panel's "Reset camera" button.
    const point3 target0 = target;
    const double radius0 = radius, azimuth0 = azimuth, elevation0 = elevation;
    const double vfov0 = cam->vfov;

    // Launch config, restored by the UI panel's "Reset config" button.
    const int spp_per_frame0  = spp_per_frame;
    const int target_samples0 = target_samples;
    const int max_depth0      = cam->max_depth;

    // Per-stage GPU timings for the UI panel. Kernel launches are async, so
    // wall-clock timers around them would misattribute the cost to whatever
    // syncs next; CUDA events timestamp the GPU's own timeline instead.
    // The values shown are from the previous frame (the panel is built before
    // this frame's kernels run).
    cudaEvent_t ev_trace0, ev_trace1, ev_tone0, ev_tone1;
    checkCudaErrors(cudaEventCreate(&ev_trace0));
    checkCudaErrors(cudaEventCreate(&ev_trace1));
    checkCudaErrors(cudaEventCreate(&ev_tone0));
    checkCudaErrors(cudaEventCreate(&ev_tone1));
    float ms_trace = 0.0f, ms_tonemap = 0.0f;

    // Which sections of the Renderer panel are visible (toggled in View menu).
    bool show_performance = true, show_samples = true, show_config = true, show_camera = true;
    bool show_selection = true, show_dynamic = true;

    // ---- C/D: dynamic scene — physics-simulated object ----
    // One demo body (a sphere) falls under gravity and bounces on the ground
    // plane (y=0). Each simulated frame rewrites its transform via the B5
    // mutation protocol (placement-new -> refit -> reset accumulation). The
    // integrator steps a FIXED dt (accumulator-paced against wall time), so
    // behaviour is frame-rate-independent; when the body settles it SLEEPS,
    // stopping the accumulation reset so the image converges. Rest pose is
    // initial_trs[anim_id]; the body spawns drop_height above it.
    bool   animating   = false;   // Play/pause the sim (Drop launches it)
    bool   asleep      = false;   // settled -> stop stepping + resetting accumulation
    double sim_accum   = 0.0;     // fixed-step time accumulator (real seconds)
    vec3   phys_pos, phys_vel;    // world center + velocity of the simulated sphere
    float  gravity     = -9.8f;   // world units / s^2 (negative = down)
    float  restitution = 0.7f;    // bounce energy retained (0..1)
    float  drop_height = 3.0f;    // spawn height above the rest pose
    const int anim_id  = 3;       // the metal sphere is the simulated body
    const double PHYS_H = 1.0 / 240.0;  // fixed integration step
    const int    PHYS_MAX_STEPS = 8;    // per-frame substep cap (spiral-of-death guard)
    const real   SLEEP_VEL = real(0.08);// |v| below this on the ground => sleep

    // ---- B4 picking state ----
    // A click (press+release with ≤2 px of motion) picks; a drag orbits.
    int selected_id = -1;
    int press_x = 0, press_y = 0;
    bool maybe_click = false;

    // Dedicated pick RNG (never touches the render states), seeded once.
    curandState* pick_state;
    checkCudaErrors(cudaMalloc(&pick_state, sizeof(curandState)));
    initialize_rand_pick<<<1, 1>>>(pick_state, rng_seed + 1);
    checkCudaErrors(cudaDeviceSynchronize());

    // The pick kernel's answer, in managed memory so the host can read it back.
    int* pick_result;
    checkCudaErrors(cudaMallocManaged(&pick_result, sizeof(int)));

    // ---- NVML: this process's VRAM on the SAME physical GPU CUDA is using ----
    // cudaMemGetInfo is device-wide (counts other apps); NVML per-process is our
    // own footprint, matching nvidia-smi. Bind NVML by the CUDA device's PCI bus
    // id — device indices need not match on a multi-GPU box. Polled every ~30
    // frames (the process-list query is heavier than a memory read).
    nvmlDevice_t nvml_dev;
    bool nvml_ok = (nvmlInit() == NVML_SUCCESS);
    if (nvml_ok) {
        int cuda_dev; cudaGetDevice(&cuda_dev);
        char pci[32]; cudaDeviceGetPCIBusId(pci, sizeof(pci), cuda_dev);
        nvml_ok = (nvmlDeviceGetHandleByPciBusId(pci, &nvml_dev) == NVML_SUCCESS);
    }
    const unsigned int my_pid = (unsigned int)getpid();
    int vram_used_mb = -1;   // cached poll result (-1 = unknown)
    int vram_poll = 0;

    // Throw away accumulated samples and start converging afresh. Anything that
    // changes what is rendered calls this: camera motion (below) and B5 object
    // edits — mixing samples from different scenes would be wrong.
    auto reset_accumulation = [&]() {
        checkCudaErrors(cudaMemset(accum, 0, (size_t)W * H * sizeof(color)));
        total_samples = 0;
    };

    // D: (re)launch the simulated body — lift it drop_height above its rest pose
    // with zero velocity and wake the sim. Rest pose = initial_trs[anim_id].
    auto drop_ball = [&]() {
        const init_trs& base = initial_trs[anim_id];
        phys_pos  = base.t + vec3(0, drop_height, 0);
        phys_vel  = vec3(0, 0, 0);
        sim_accum = 0.0;
        asleep    = false;
        animating = true;
    };

    // Rebuild the camera from the orbit state and reset accumulation. Called
    // whenever the view changes (drag / scroll / pan / R).
    auto rebuild_camera = [&]() {
        double ce = cos(elevation), se = sin(elevation);
        double ca = cos(azimuth),   sa = sin(azimuth);
        cam->lookfrom = target + radius * vec3(ce * ca, se, ce * sa);
        cam->lookat   = target;
        cam->initialize();
        reset_accumulation();
    };

    // ---- present loop: accumulate a few spp, then present the running average ----
    bool running = true;
    while (running) {
        bool camera_dirty = false;
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);   // the UI sees every event first
            ImGuiIO& io = ImGui::GetIO();
            if (e.type == SDL_QUIT) running = false;
            else if (io.WantCaptureKeyboard && (e.type == SDL_KEYDOWN || e.type == SDL_KEYUP)) {
                // typing in the UI — don't treat as viewer hotkeys
            }
            else if (io.WantCaptureMouse && e.type != SDL_KEYDOWN && e.type != SDL_KEYUP) {
                // hovering/dragging the UI — don't orbit/zoom/pan the camera
            }
            else if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) running = false;
            else if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_r) camera_dirty = true;
            else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                press_x = e.button.x; press_y = e.button.y;
                maybe_click = true;
            }
            else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
                // B4: a release near the press point is a pick, not a drag.
                if (maybe_click && abs(e.button.x - press_x) <= 2 && abs(e.button.y - press_y) <= 2) {
                    pick<<<1, 1>>>(*cam, world, e.button.x, e.button.y, pick_state, pick_result);
                    checkCudaErrors(cudaDeviceSynchronize());
                    selected_id = *pick_result;   // -1 on miss = deselect
                }
                maybe_click = false;
            }
            else if (e.type == SDL_MOUSEWHEEL) {
                radius *= pow(0.9, e.wheel.y);          // wheel up = zoom in
                if (radius < 0.1) radius = 0.1;
                camera_dirty = true;
            }
            else if (e.type == SDL_MOUSEMOTION && (e.motion.state & SDL_BUTTON_LMASK)
                     && !(SDL_GetModState() & KMOD_SHIFT)) {
                azimuth   += e.motion.xrel * 0.005;
                elevation += e.motion.yrel * 0.005;
                const double lim = 1.55;                // ~89°, avoid the poles
                if (elevation >  lim) elevation =  lim;
                if (elevation < -lim) elevation = -lim;
                camera_dirty = true;
            }
            else if (e.type == SDL_MOUSEMOTION
                     && (e.motion.state & SDL_BUTTON_LMASK) && (SDL_GetModState() & KMOD_SHIFT)) {
                // pan: slide the target in the camera's screen plane
                vec3 fwd   = unit_vector(cam->lookat - cam->lookfrom);
                vec3 right = unit_vector(cross(fwd, world_up));
                vec3 up    = cross(right, fwd);
                double k = radius * 0.002;
                target = target + (-e.motion.xrel * right + e.motion.yrel * up) * k;
                camera_dirty = true;
            }
        }

        // ---- UI panel (Dear ImGui). Built every frame; widgets that change
        // what is being rendered set camera_dirty so accumulation restarts —
        // mixing samples taken under different settings would be wrong.
        // spp/frame and the target only change pacing, so they don't reset.
        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // Top menu bar: File (quit) and View (toggle panel sections).
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Quit", "Esc")) running = false;
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Performance", nullptr, &show_performance);
                ImGui::MenuItem("Samples", nullptr, &show_samples);
                ImGui::MenuItem("Config",  nullptr, &show_config);
                ImGui::MenuItem("Camera",  nullptr, &show_camera);
                ImGui::MenuItem("Selection", nullptr, &show_selection);
                ImGui::MenuItem("Dynamic", nullptr, &show_dynamic);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        ImGui::SetNextWindowPos(ImVec2(10, 30), ImGuiCond_FirstUseEver);  // below the menu bar
        ImGui::Begin("Renderer");
        {
            ImGuiIO& io = ImGui::GetIO();
            // Emit a separator between consecutive *visible* sections only.
            bool any_section = false;
            auto section_break = [&]() { if (any_section) ImGui::Separator(); any_section = true; };

            if (show_performance) {
                section_break();
                ImGui::Text("%.0f fps", io.Framerate);
                // Per-stage breakdown of the last frame: trace/tonemap from CUDA
                // events; present = the rest of the frame (GL upload, quad, UI,
                // swap, event polling), derived from the frame delta.
                ImGui::Text("trace   %6.2f ms", ms_trace);
                ImGui::Text("tonemap %6.2f ms", ms_tonemap);
                ImGui::Text("present %6.2f ms",
                            fmaxf(0.0f, io.DeltaTime * 1000.0f - ms_trace - ms_tonemap));
                // This process's VRAM (NVML, matching nvidia-smi), re-polled
                // every 30 frames; the value persists between polls.
                if (nvml_ok && vram_poll++ % 30 == 0) {
                    unsigned int n = 64;
                    nvmlProcessInfo_t procs[64];
                    if (nvmlDeviceGetComputeRunningProcesses(nvml_dev, &n, procs) == NVML_SUCCESS) {
                        vram_used_mb = 0;   // our pid absent from the list = 0 MB
                        for (unsigned int k = 0; k < n; k++)
                            if (procs[k].pid == my_pid)
                                vram_used_mb = (int)(procs[k].usedGpuMemory / (1024 * 1024));
                    }
                }
                if (vram_used_mb >= 0) ImGui::Text("vram    %6d MB", vram_used_mb);
                else                   ImGui::Text("vram    %6s MB", "n/a");
            }

            if (show_samples) {
                section_break();
                ImGui::Text("%lld spp", total_samples);
            }

            if (show_config) {
                section_break();
                ImGui::SliderInt("spp", &spp_per_frame, 1, 64);
                ImGui::SliderInt("target spp", &target_samples, 16, 65536, "%d",
                                 ImGuiSliderFlags_Logarithmic);
                if (ImGui::SliderInt("max depth", &cam->max_depth, 1, 50)) camera_dirty = true;
                float vfov_f = (float)cam->vfov;
                if (ImGui::SliderFloat("vfov", &vfov_f, 5.0f, 90.0f, "%.0f deg")) {
                    cam->vfov = vfov_f;
                    camera_dirty = true;
                }
                if (ImGui::Button("Reset config")) {
                    spp_per_frame  = spp_per_frame0;
                    target_samples = target_samples0;
                    // max depth / vfov change what's rendered — restart only if they moved
                    if (cam->max_depth != max_depth0 || cam->vfov != vfov0) camera_dirty = true;
                    cam->max_depth = max_depth0;
                    cam->vfov      = vfov0;
                }
            }

            if (show_dynamic) {
                section_break();
                if (ImGui::Button("Drop")) drop_ball();   // D: launch the sim
                ImGui::SameLine();
                ImGui::Checkbox("Play", &animating);      // pause/resume
                ImGui::SliderFloat("gravity",     &gravity,     -30.0f, 0.0f, "%.1f");
                ImGui::SliderFloat("restitution", &restitution,   0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("drop height",  &drop_height,  0.0f, 8.0f, "%.1f");
            }

            if (show_camera) {
                section_break();
                ImGui::Text("cam    (%.2f, %.2f, %.2f)", cam->lookfrom.x(), cam->lookfrom.y(), cam->lookfrom.z());
                ImGui::Text("target (%.2f, %.2f, %.2f)", target.x(), target.y(), target.z());
                ImGui::Text("r      %.2f", radius);
                if (ImGui::Button("Reset camera")) {
                    target = target0; radius = radius0;
                    azimuth = azimuth0; elevation = elevation0;
                    cam->vfov = vfov0;
                    camera_dirty = true;
                }
            }

            if (show_selection) {
                section_break();
                if (selected_id >= 0) {
                    ImGui::Text("selected  id %d", selected_id);

                    // B5 object manipulation: edit the selected object's full
                    // TRS, then run the mutation protocol — placement-new the
                    // transform (rewrites params AND recomputes matrices + bbox)
                    // -> refit() -> restart accumulation. Generic over prim type:
                    // every editable object is a transform(prim). The bbox
                    // highlight follows via bounding_box().
                    hittable* h = sc.get(selected_id);
                    if (h->type == TRANSFORM) {
                        transform* tr = static_cast<transform*>(h->object);
                        float t[3] = {(float)tr->translation.x(), (float)tr->translation.y(), (float)tr->translation.z()};
                        float r[3] = {(float)tr->rotation.x(),    (float)tr->rotation.y(),    (float)tr->rotation.z()};
                        float s[3] = {(float)tr->scale.x(),       (float)tr->scale.y(),       (float)tr->scale.z()};
                        bool edited = false;
                        edited |= ImGui::DragFloat3("translate", t, 0.05f);
                        edited |= ImGui::DragFloat3("rotate",    r, 1.0f);
                        edited |= ImGui::DragFloat3("scale",     s, 0.02f, 0.01f, 100.0f);
                        if (edited) {
                            for (int c = 0; c < 3; c++) s[c] = fmaxf(s[c], 0.01f);  // scale must stay positive
                            new(tr) transform(tr->child, point3(t[0], t[1], t[2]),
                                              vec3(r[0], r[1], r[2]), vec3(s[0], s[1], s[2]));
                            sc.refit();
                            reset_accumulation();
                        }
                        // Restore the object's initial pose (same mutation protocol).
                        // The drag fields re-read tr next frame, so the UI follows.
                        if (ImGui::Button("Reset transform")) {
                            const init_trs& in = initial_trs[selected_id];
                            new(tr) transform(tr->child, in.t, in.r, in.s);
                            sc.refit();
                            reset_accumulation();
                        }
                    } else {
                        ImGui::TextDisabled("not editable");
                    }
                } else {
                    ImGui::Text("selected  none");
                }
            }
        }
        ImGui::End();

        // B4 highlight: project the selected object's bbox and outline it as a
        // 2D overlay — the render itself is untouched, so no accumulation
        // reset, and the outline follows camera moves via reprojection.
        if (selected_id >= 0) {
            aabb bb = sc.get(selected_id)->bounding_box();
            point3 corner[8];
            for (int k = 0; k < 8; k++)
                corner[k] = point3(k & 1 ? bb.x.max : bb.x.min,
                                   k & 2 ? bb.y.max : bb.y.min,
                                   k & 4 ? bb.z.max : bb.z.min);
            static const int edge[12][2] = {{0,1},{0,2},{0,4},{1,3},{1,5},{2,3},
                                            {2,6},{3,7},{4,5},{4,6},{5,7},{6,7}};
            ImDrawList* dl = ImGui::GetForegroundDrawList();
            for (int k = 0; k < 12; k++) {
                real ax, ay, bx, by;
                if (cam->world_to_pixel(corner[edge[k][0]], ax, ay) &&
                    cam->world_to_pixel(corner[edge[k][1]], bx, by))
                    dl->AddLine(ImVec2((float)ax, (float)ay), ImVec2((float)bx, (float)by),
                                IM_COL32(255, 220, 0, 255), 1.5f);
            }
        }

        if (camera_dirty) rebuild_camera();   // recompute view + reset accumulation

        // D (physics): fixed-step gravity + ground-plane bounce on the simulated
        // body. Wall dt drives a fixed-h accumulator (frame-rate independent),
        // clamped to PHYS_MAX_STEPS to avoid the spiral of death. Each stepped
        // frame rewrites the transform (B5 mutation protocol) and restarts
        // accumulation; when the body settles it SLEEPS so the image converges.
        hittable* body = sc.get(anim_id);
        if (animating && !asleep && body && body->type == TRANSFORM
            && initial_trs[anim_id].editable) {
            const init_trs& base = initial_trs[anim_id];
            transform* tr = static_cast<transform*>(body->object);
            real radius = real(1);   // vertical extent of the sphere on the ground
            if (tr->child->type == SPHERE)
                radius = static_cast<sphere*>(tr->child->object)->radius * base.s.y();

            sim_accum += ImGui::GetIO().DeltaTime;
            const double cap = PHYS_H * PHYS_MAX_STEPS;
            if (sim_accum > cap) sim_accum = cap;     // spiral-of-death clamp
            bool stepped = false;
            while (sim_accum >= PHYS_H) {
                phys_vel[1] += gravity * real(PHYS_H);        // gravity (y only)
                phys_pos    += phys_vel * real(PHYS_H);       // integrate
                if (phys_pos[1] - radius <= 0) {              // ground collision
                    phys_pos[1] = radius;
                    phys_vel[1] = -restitution * phys_vel[1];
                }
                sim_accum -= PHYS_H;
                stepped = true;
            }
            // Sleep once settled on the ground, so accumulation can converge.
            if (phys_pos[1] <= radius + real(1e-3) && phys_vel.length() < SLEEP_VEL) {
                phys_pos[1] = radius; phys_vel = vec3(0, 0, 0); asleep = true;
            }
            if (stepped) {   // apply pose -> refit -> restart accumulation
                new(tr) transform(tr->child, point3(phys_pos), base.r, base.s);
                sc.refit();
                reset_accumulation();
            }
        }

        // B2: add spp_per_frame fresh samples per pixel until the target is
        // reached; after that just keep presenting the converged image.
        bool did_accumulate = false;
        if (total_samples < target_samples) {
            checkCudaErrors(cudaEventRecord(ev_trace0));
            accumulate_frame<<<blocks, threads>>>(*cam, cam->max_depth, world, accum, rand_states, spp_per_frame);
            checkCudaErrors(cudaEventRecord(ev_trace1));
            did_accumulate = true;
            total_samples += spp_per_frame;

            char title[96];
            snprintf(title, sizeof(title), "RayTracingCUDA — viewer");
            SDL_SetWindowTitle(win, title);
        }

        const int denom = total_samples > 0 ? (int)total_samples : 1;  // tonemap divisor

        if (use_interop) {
            // Fast path: CUDA writes RGBA8 straight into the mapped PBO,
            // then PBO -> texture entirely on the GPU.
            uchar4* dptr = nullptr;
            size_t nbytes = 0;
            checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo, 0));
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &nbytes, cuda_pbo));
            checkCudaErrors(cudaEventRecord(ev_tone0));
            tonemap_frame<<<blocks, threads>>>(accum, dptr, W, H, denom);
            checkCudaErrors(cudaEventRecord(ev_tone1));
            checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo, 0));   // syncs the stream

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        } else {
            // Portable path: tonemap into a device buffer, copy to pinned host
            // memory (the memcpy also synchronizes the kernel), upload from the
            // host pointer. ~1.4 MB/frame at 800x450 — negligible next to VNC.
            checkCudaErrors(cudaEventRecord(ev_tone0));
            tonemap_frame<<<blocks, threads>>>(accum, d_rgba, W, H, denom);
            checkCudaErrors(cudaMemcpy(h_rgba, d_rgba, frame_bytes, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaEventRecord(ev_tone1));   // after the copy: tonemap+copy together

            glBindTexture(GL_TEXTURE_2D, tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, h_rgba);
        }

        // Both paths have synchronized (unmap / blocking memcpy), so the events
        // are complete and these queries don't stall. Shown next frame.
        if (did_accumulate) checkCudaErrors(cudaEventElapsedTime(&ms_trace, ev_trace0, ev_trace1));
        else                ms_trace = 0.0f;
        checkCudaErrors(cudaEventSynchronize(ev_tone1));
        checkCudaErrors(cudaEventElapsedTime(&ms_tonemap, ev_tone0, ev_tone1));

        // draw a fullscreen textured quad (v flipped so image row 0 is on top)
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

        // UI on top of the rendered frame
        ImGui::Render();
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(win);
    }

    // ---- cleanup ----
    if (nvml_ok) nvmlShutdown();
    cudaFree(pick_state);
    cudaFree(pick_result);
    cudaEventDestroy(ev_trace0);
    cudaEventDestroy(ev_trace1);
    cudaEventDestroy(ev_tone0);
    cudaEventDestroy(ev_tone1);
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    if (use_interop) cudaGraphicsUnregisterResource(cuda_pbo);
    if (d_rgba) cudaFree(d_rgba);
    if (h_rgba) cudaFreeHost(h_rgba);
    cudaFree(accum);
    cudaFree(rand_states);
    cudaFree(cam);
    sc.release();
    if (pbo) glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    SDL_GL_DeleteContext(gl);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
