// viewer.cu — interactive real-time viewer (roadmap workstreams B1 + B2).
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
// app stays responsive. R resets the accumulation (the same reset B3 camera
// motion will trigger); past RT_TARGET_SAMPLES the viewer stops adding and
// just presents. Because each pixel's cuRAND state persists across launches,
// K frames of M spp consume the same RNG stream — and add in the same order —
// as one K*M-spp render, so the accumulated image is byte-identical to the
// single-shot one (tests/run_tests.sh stage [7/7] checks exactly this).
// The camera is still static (camera control is B3).
//
// Needs a display (local, VNC, or X-forward) plus SDL2 + GLEW + OpenGL dev libs.
// Build (match -arch to your GPU; see scripts/build_viewer.sh):
//   nvcc src/viewer/viewer.cu -o build/viewer -std=c++14 -arch=sm_86 -Isrc \
//        -lSDL2 -lGLEW -lGL
//   ./build/viewer            # ESC or close the window to quit

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <cuda_gl_interop.h>

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

// --- scene: checker ground + the three book spheres (glass / diffuse / metal),
// sky-lit. Builds into a persistent world + BVH; records allocations in `allocs`.
static hittable* build_scene(std::vector<void*>& allocs, hittable*& bvh_hittable_out) {
    hittable_list* world;
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(hittable_list)));
    new(world) hittable_list();

    material* ground = new_lambertian(
        make_checker(0.32, color(.2, .3, .1), color(.9, .9, .9)), allocs);
    material* diffuse = new_lambertian(color(0.4, 0.2, 0.1), allocs);
    material* glass   = new_dielectric(1.5, allocs);
    material* green   = new_tinted_glass(1.5, color(2.0, 0.2, 2.0), allocs);
    material* metal_m = new_metal(color(0.7, 0.6, 0.5), 0.0, allocs);

    add_sphere(world, point3(0, -1000, 0), 1000, ground,  allocs);
    add_sphere(world, point3(-4, 1, 0),    1.0,  diffuse, allocs);
    add_sphere(world, point3(-1.4, 1, 2),  1.0,  green,   allocs);
    add_sphere(world, point3(0, 1, 0),     1.0,  glass,   allocs);
    add_sphere(world, point3(4, 1, 0),     1.0,  metal_m, allocs);

    // BVH over the same objects
    bvh_scene* bvh;
    checkCudaErrors(cudaMallocManaged((void**)&bvh, sizeof(bvh_scene)));
    new(bvh) bvh_scene();
    for (int i = 0; i < world->size; i++)
        bvh->add(*world->objects[i]);
    bvh->build();
    allocs.push_back(bvh);   // note: bvh_scene dtor not run here — process exits at quit

    hittable* bvh_hittable;
    checkCudaErrors(cudaMallocManaged((void**)&bvh_hittable, sizeof(hittable)));
    bvh_hittable->type = BVH;
    bvh_hittable->object = bvh;
    allocs.push_back(bvh_hittable);
    allocs.push_back(world);

    bvh_hittable_out = bvh_hittable;
    return bvh_hittable;
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
            "RayTracingCUDA — viewer (B2)",
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

    std::vector<void*> allocs;
    hittable* bvh_hittable;
    hittable& world = *build_scene(allocs, bvh_hittable);

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
    long long total_samples = 0;   // per-pixel samples accumulated so far

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    fprintf(stderr, "viewer: %dx%d, %d spp/frame (target %d), depth %d — present: %s\n",
            W, H, spp_per_frame, RT_TARGET_SAMPLES, cam->max_depth,
            headless    ? "headless PPM dump"
          : use_interop ? "CUDA-GL interop — ESC or close the window to quit, R to restart accumulation"
                        : "CPU readback — ESC or close the window to quit, R to restart accumulation");

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
        for (void* p : allocs) cudaFree(p);
        return 0;
    }

    // ---- present loop: accumulate a few spp, then present the running average ----
    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            else if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) running = false;
            else if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_r) {
                // Restart accumulation from scratch (B3 camera motion will
                // trigger this same reset). RNG states keep advancing.
                checkCudaErrors(cudaMemset(accum, 0, (size_t)W * H * sizeof(color)));
                total_samples = 0;
            }
        }

        // B2: add spp_per_frame fresh samples per pixel until the target is
        // reached; after that just keep presenting the converged image.
        if (total_samples < RT_TARGET_SAMPLES) {
            accumulate_frame<<<blocks, threads>>>(*cam, cam->max_depth, world, accum, rand_states, spp_per_frame);
            total_samples += spp_per_frame;

            char title[96];
            snprintf(title, sizeof(title), "RayTracingCUDA — viewer (B2) — %lld spp", total_samples);
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
            tonemap_frame<<<blocks, threads>>>(accum, dptr, W, H, denom);
            checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo, 0));

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        } else {
            // Portable path: tonemap into a device buffer, copy to pinned host
            // memory (the memcpy also synchronizes the kernel), upload from the
            // host pointer. ~1.4 MB/frame at 800x450 — negligible next to VNC.
            tonemap_frame<<<blocks, threads>>>(accum, d_rgba, W, H, denom);
            checkCudaErrors(cudaMemcpy(h_rgba, d_rgba, frame_bytes, cudaMemcpyDeviceToHost));

            glBindTexture(GL_TEXTURE_2D, tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, h_rgba);
        }

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

        SDL_GL_SwapWindow(win);
    }

    // ---- cleanup ----
    if (use_interop) cudaGraphicsUnregisterResource(cuda_pbo);
    if (d_rgba) cudaFree(d_rgba);
    if (h_rgba) cudaFreeHost(h_rgba);
    cudaFree(accum);
    cudaFree(rand_states);
    cudaFree(cam);
    for (void* p : allocs) cudaFree(p);
    if (pbo) glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    SDL_GL_DeleteContext(gl);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
