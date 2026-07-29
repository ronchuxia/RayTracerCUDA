#ifndef VIEWER_SCENES_VIEWER_SCENE_H
#define VIEWER_SCENES_VIEWER_SCENE_H

#include <vector>

#include "physics.h"
#include "vec3.h"

// Per-scene config the viewer needs beyond the geometry: the initial camera and
// the scene's PHYSICS BODIES. Each build_*_scene() builds the scene AND returns
// this, so the viewer reads both from one place per scene instead of scattered
// `#if VIEWER_SCENE` blocks.
//
// Physics is a scene property, not a viewer one. The scene declares every body
// explicitly — via the factories in viewer/physics_utils.h — because only the
// scene knows what it built and what it means: which sphere falls, which box is
// an obstacle, whether a triangle collides at all. The viewer derives nothing;
// it copies this list and indexes it by scene id. Entries with scene_id < 0 are
// unlinked — no render counterpart, so no pose is written back and they cannot
// be picked. Today those are the ground and walls, whose visual stand-ins are
// deliberately not their collision shape.
struct viewer_scene {
    point3 lookfrom, lookat;        // initial camera
    real   vfov;
    std::vector<phys_body> bodies;  // authored physics bodies; scene_id < 0 = unlinked
};

#endif // VIEWER_SCENES_VIEWER_SCENE_H
