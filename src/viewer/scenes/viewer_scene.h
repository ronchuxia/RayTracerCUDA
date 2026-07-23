#ifndef VIEWER_SCENES_VIEWER_SCENE_H
#define VIEWER_SCENES_VIEWER_SCENE_H

#include "vec3.h"

// Per-scene config the viewer needs beyond the geometry: the initial camera and
// (for the physics container) the collision wall bounds. Each build_*_scene()
// builds the scene AND returns this, so the viewer reads camera + walls from one
// place per scene instead of scattered `#if VIEWER_SCENE` blocks.
struct viewer_scene {
    point3 lookfrom, lookat;   // initial camera
    real   vfov;
    vec3   wall_min, wall_max; // physics container walls on x/z; +/- huge = none
};

#endif // VIEWER_SCENES_VIEWER_SCENE_H
