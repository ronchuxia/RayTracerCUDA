#ifndef VIEWER_PHYSICS_UTILS_H
#define VIEWER_PHYSICS_UTILS_H

#include <cstdlib>
#include <iostream>

#include "hittable.h"
#include "hittables/sphere.h"
#include "hittables/transforms.h"
#include "physics.h"
#include "scene.h"

// Host-only physics-construction helpers: the factories a scene uses to author
// its bodies (see viewer/scenes/viewer_scene.h). The physics counterpart of
// scenes/scene_utils.h, which does the same job for geometry.
//
// PHYSICS IS A SCENE PROPERTY. A scene knows what it built and what it means —
// which sphere falls, which box is an obstacle, whether a triangle collides at
// all — so it declares its bodies explicitly rather than the viewer guessing
// from geometry. The viewer only indexes what it is handed. Anything derived
// instead of authored is a guess that is eventually wrong for some scene: the
// rule these replaced gave the showcase's decorative triangle a box collider
// nobody wanted, and could not express a static sphere.
//
// A body is LINKED or UNLINKED, and that is the only distinction — there is no
// second kind of body. Both go through the same narrow phase and the same
// solver, and both carry the same role, mass and surface properties.
//   scene_id >= 0  LINKED to a scene object. The viewer writes simulated poses
//                  back to its transform and the Object panel can select it.
//   scene_id <  0  UNLINKED: no render counterpart. Nothing produces one today
//                  — every collider is read from an object — but the field and
//                  the solver both still support it, for the deferred
//                  "collider with no visual" case.
//
// EVERY COLLIDER IS READ FROM ITS OBJECT. There is no way to author a shape that
// differs from what renders, which is why there is no plane collider: nothing in
// a scene is an infinite plane. The ground collides as the big sphere it draws
// as; a wall collides as the quad it draws as. Letting a scene write its own
// shape — a box collider on a sphere, or a collider with no visual at all — is
// designed in docs/plans/phase3-status.md item 1 and deliberately not built.
//
// The sphere/box factories read the object's live pose, so call them after the
// object is added. Both require a TRANSFORM-wrapped object: the transform is
// what supplies pose (and what a drag edits), which is also what makes the
// object selectable in the first place.

// Every factory needs a TRANSFORM-wrapped object: the transform is what supplies
// the pose a collider is placed by. Checked rather than assumed — a static_cast
// through a wrong tag reads unrelated bytes as a pose. Fatal like the asset
// loaders (loaders/stl_loader.h), because it is an authoring mistake that would
// otherwise produce a silently wrong collider.
inline transform* get_body_transform(scene& sc, int scene_id) {
    hittable* h = sc.get(scene_id);
    if (!h || h->type != TRANSFORM) {
        std::cerr << "scene object " << scene_id << " is not transform-wrapped\n";
        std::exit(1);
    }
    return static_cast<transform*>(h->object);
}

// A sphere collider for scene object `scene_id`. Radius is the prim's radius
// scaled by the transform, matching how the viewer re-derives it after an edit.
//
// Both casts are CHECKED. `hittable` carries a type tag precisely so this is
// checkable, and reading past a wrong tag is not a wrong number but a read of
// unrelated memory: pointing this at the ball pit's obstacle (a transform-wrapped
// BOX) yields radius 0.450050 — the box's own half-height, read out of the
// child list's bounding box — which is plausible enough to survive review.
inline phys_body make_sphere_body(scene& sc, int scene_id, motion_type motion = DYNAMIC,
                                  real mass = real(1),
                                  real friction = real(0.5), real restitution = real(0.7)) {
    transform* tr = get_body_transform(sc, scene_id);
    if (tr->child->type != SPHERE) {
        std::cerr << "scene object " << scene_id << " does not wrap a sphere\n";
        std::exit(1);
    }
    sphere* sp = static_cast<sphere*>(tr->child->object);
    phys_body b{ scene_id, tr->translation, vec3(0,0,0),
                 sp->radius * tr->scale.y(), tr->rotation, tr->scale };
    b.motion = motion;  b.mass = mass;
    b.friction = friction;  b.restitution = restitution;
    return b;
}

// The ORIENTED box collider of a transform-wrapped object: take the child's own
// bounds in its LOCAL frame, scale them, then place the result with the
// transform's rotation and translation.
//
// Deriving locally is what makes a rotated box collide as itself. The world AABB
// (axis-aligned bounding box) of a rotated box is strictly larger than the box —
// a 45-degree turn about y inflates a unit box's footprint by 41% — so a
// collider read from it would stop balls short of the visible surface. Nothing
// here is a new collision test: the rotation goes into `axes` and
// `contact_between` transforms into that frame.
//
// One derivation, two callers: the scene authors a body with make_box_body, and
// the viewer re-runs this after a drag. They cannot disagree.
//
// Still inherits aabb::pad()'s ~5e-5-per-side widening of planar faces (0.01% on
// a half-unit box), because the extents come from a bounding box: new_box builds
// six quads and has no half-extent of its own to read. Authored extents (the
// linked+authored case) would remove it.
inline void box_collider_of(const transform* tr, vec3& pos, vec3& half, vec3 axes[3]) {
    aabb c = tr->child->bounding_box();                     // child's LOCAL bounds
    vec3 lo(c.x.min, c.y.min, c.z.min), hi(c.x.max, c.y.max, c.z.max);
    half = (hi - lo) * real(0.5) * tr->scale;
    pos  = tr->apply_R((lo + hi) * real(0.5) * tr->scale) + tr->translation;
    axes[0] = tr->apply_R(vec3(1, 0, 0));                   // columns of R: the
    axes[1] = tr->apply_R(vec3(0, 1, 0));                   // box's own axes in
    axes[2] = tr->apply_R(vec3(0, 0, 1));                   // world space
}

// A box collider for scene object `scene_id`. Works for any wrapped shape, since the
// extents come from the child's bounding box rather than from a box-specific
// field.
inline phys_body make_box_body(scene& sc, int scene_id, motion_type motion = STATIC,
                               real mass = real(1),
                               real friction = real(0.5), real restitution = real(0.7)) {
    transform* tr = get_body_transform(sc, scene_id);
    phys_body b{ scene_id, vec3(0,0,0), vec3(0,0,0), real(0), tr->rotation, tr->scale };
    b.motion = motion;  b.mass = mass;
    b.shape  = COLLIDER_BOX;
    box_collider_of(tr, b.pos, b.half, b.axes);
    b.friction = friction;  b.restitution = restitution;
    return b;
}

#endif // VIEWER_PHYSICS_UTILS_H
