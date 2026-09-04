#ifndef VIEWER_PHYSICS_UTILS_H
#define VIEWER_PHYSICS_UTILS_H

#include <cstdlib>
#include <iostream>

#include "hittable.h"
#include "hittables/sphere.h"
#include "hittables/transforms.h"
#include "physics/body.h"
#include "viewer/scene.h"

inline transform* get_body_transform(scene& sc, int scene_id) {
    hittable* h = sc.get(scene_id);
    if (!h || h->type != TRANSFORM) {
        std::cerr << "scene object " << scene_id << " is not transform-wrapped\n";
        std::exit(1);
    }
    return static_cast<transform*>(h->object);
}

// make a sphere body from a sphere object
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
                 tr->rotation, tr->scale };
    b.radius = sp->radius * tr->scale.y();
    b.motion = motion;  b.mass = mass;
    b.shape  = COLLIDER_SPHERE;
    b.friction = friction;  b.restitution = restitution;
    return b;
}

inline void box_collider_of(const transform* tr, vec3& pos, vec3& half, vec3 axes[3]) {
    aabb c = tr->child->bounding_box();                     // child's local bounding box
    vec3 lo(c.x.min, c.y.min, c.z.min), hi(c.x.max, c.y.max, c.z.max);
    half = (hi - lo) * real(0.5) * tr->scale;
    pos  = tr->apply_R((lo + hi) * real(0.5) * tr->scale) + tr->translation;
    axes[0] = tr->apply_R(vec3(1, 0, 0));
    axes[1] = tr->apply_R(vec3(0, 1, 0));
    axes[2] = tr->apply_R(vec3(0, 0, 1));
}

// make a box body from ANY object
inline phys_body make_box_body(scene& sc, int scene_id, motion_type motion = STATIC,
                               real mass = real(1),
                               real friction = real(0.5), real restitution = real(0.7)) {
    transform* tr = get_body_transform(sc, scene_id);
    phys_body b{ scene_id, vec3(0,0,0), vec3(0,0,0), tr->rotation, tr->scale };
    b.motion = motion;  b.mass = mass;
    b.shape  = COLLIDER_BOX;
    // derive the box collider from the object
    vec3 pos, half, axes[3];
    box_collider_of(tr, pos, half, axes);
    // set properties of the box collider
    b.pos = pos;
    b.half = half;
    set_box_orientation_from_box_axes(b, axes);
    b.friction = friction;  b.restitution = restitution;
    return b;
}

#endif // VIEWER_PHYSICS_UTILS_H
