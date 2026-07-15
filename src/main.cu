// Entry point. Defines the compile-time knobs, then includes the scene headers
// (which read those knobs) and dispatches to one scene from main(). Scene
// construction lives in scenes/; shared build helpers in scenes/scene_utils.h.

// Compile-time knobs, overridable for tests, e.g.:
//   nvcc src/main.cu -Isrc -DUSE_BVH=0 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16 ...
#ifndef USE_BVH
#define USE_BVH 1          // 1: render through the flattened BVH; 0: flat hittable_list
#endif
#ifndef RT_SCENE
#define RT_SCENE 0         // 0: sphere scene; 1: Cornell box (quads + triangle + boxes)
#endif
#ifndef RT_IMAGE_WIDTH
#if RT_SCENE == 1
#define RT_IMAGE_WIDTH 600   // the Cornell box renders square
#else
#define RT_IMAGE_WIDTH 1200
#endif
#endif
#ifndef RT_SAMPLES
#define RT_SAMPLES 512
#endif

#include "scenes/spheres.h"
#include "scenes/cornell.h"

int main() {
#if RT_SCENE == 1
    cornell_box();
#else
    spheres();
#endif
    return 0;
}
