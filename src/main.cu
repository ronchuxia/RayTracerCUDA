// Entry point. Defines the compile-time knobs, then includes the scene headers
// (which read those knobs) and dispatches to one scene from main(). Scene
// construction lives in scenes/; shared build helpers in scenes/scene_utils.h.

// Compile-time knobs, overridable for tests, e.g.:
//   nvcc src/main.cu -Isrc -DUSE_BVH=0 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16 ...
#ifndef USE_BVH
#define USE_BVH 1          // 1: render through the flattened BVH; 0: flat hittable_list
#endif
#ifndef RT_SCENE
#define RT_SCENE 0         // 0: spheres; 1: Cornell box; 2: badge (STL mesh); 3: Cornell smoke; 4: tinted glass
#endif
#ifndef RT_IMAGE_WIDTH
#if RT_SCENE == 1 || RT_SCENE == 3
#define RT_IMAGE_WIDTH 600   // the Cornell scenes render square
#else
#define RT_IMAGE_WIDTH 1200
#endif
#endif
#ifndef RT_SAMPLES
#define RT_SAMPLES 512
#endif
#ifndef RT_MAX_DEPTH
#define RT_MAX_DEPTH 10    // path-trace bounce cap (the fork's badge render used 100)
#endif
// RT_SKY=1 (default 0, defined in camera.h) swaps the black miss-background
// for the book's sky gradient — see camera.h.

#include "scenes/spheres.h"
#include "scenes/cornell.h"
#include "scenes/badge.h"
#include "scenes/cornell_smoke.h"
#include "scenes/glass.h"

int main() {
#if RT_SCENE == 1
    cornell_box();
#elif RT_SCENE == 2
    badge();
#elif RT_SCENE == 3
    cornell_smoke();
#elif RT_SCENE == 4
    glass();
#else
    spheres();
#endif
    return 0;
}
