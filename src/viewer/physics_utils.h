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
    phys_body b{ scene_id, tr->translation, vec3(0,0,0), tr->scale };
    b.motion = motion;  b.mass = mass;
    b.shape  = COLLIDER_SPHERE;
    b.friction = friction;  b.restitution = restitution;
    // derive the sphere collider from the object
    sphere* sp = static_cast<sphere*>(tr->child->object);
    b.radius = sp->radius * tr->scale.y();
    set_orientation(b, quat_from_euler_zyx_degrees(tr->rotation));
    return b;
}

inline void box_collider_of(const transform* tr, vec3& pos, vec3& half, quat& orient) {
    aabb c = tr->child->bounding_box();                     // child's local bounding box
    vec3 lo(c.x.min, c.y.min, c.z.min), hi(c.x.max, c.y.max, c.z.max);
    half   = (hi - lo) * real(0.5) * tr->scale;
    pos    = tr->apply_R((lo + hi) * real(0.5) * tr->scale) + tr->translation;
    orient = quat_from_euler_zyx_degrees(tr->rotation);
}

// make a box body from ANY object
inline phys_body make_box_body(scene& sc, int scene_id, motion_type motion = STATIC,
                               real mass = real(1),
                               real friction = real(0.5), real restitution = real(0.7)) {
    transform* tr = get_body_transform(sc, scene_id);
    phys_body b{ scene_id, vec3(0,0,0), vec3(0,0,0), tr->scale };
    b.motion = motion;  b.mass = mass;
    b.shape  = COLLIDER_BOX;
    b.friction = friction;  b.restitution = restitution;
    // derive the box collider from the object
    quat orient;
    box_collider_of(tr, b.pos, b.half, orient);
    set_orientation(b, orient);
    return b;
}

#endif // VIEWER_PHYSICS_UTILS_H
