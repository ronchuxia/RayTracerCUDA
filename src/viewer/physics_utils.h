#ifndef VIEWER_PHYSICS_UTILS_H
#define VIEWER_PHYSICS_UTILS_H

#include <cstdlib>
#include <iostream>

#include "hittable.h"
#include "hittables/sphere.h"
#include "hittables/transforms.h"
#include "physics/body.h"
#include "physics/hull.h"
#include "viewer/scene.h"

inline transform* get_body_transform(scene& sc, int scene_id) {
    hittable* h = sc.get(scene_id);
    if (!h || h->type != TRANSFORM) {
        std::cerr << "scene object " << scene_id << " is not transform-wrapped\n";
        std::exit(1);
    }
    return static_cast<transform*>(h->object);
}

inline void sphere_collider_of(const transform* tr, vec3& pos, real& radius, quat& orient, vec3& offset) {
    const sphere* sp = static_cast<const sphere*>(tr->child->object);
    radius = sp->radius * tr->scale.y();
    offset = sp->center * tr->scale;
    pos    = tr->apply_R(offset) + tr->translation;
    orient = quat_from_euler_zyx_degrees(tr->rotation);
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
    if (tr->scale.x() != tr->scale.y() || tr->scale.y() != tr->scale.z()) {
        std::cerr << "scene object " << scene_id << " needs uniform scale\n";
        std::exit(1);
    }
    phys_body b{ scene_id, vec3(0,0,0), vec3(0,0,0), tr->scale };
    b.motion = motion;  b.mass = mass;
    b.shape  = COLLIDER_SPHERE;
    b.friction = friction;  b.restitution = restitution;
    // derive the sphere collider from the object
    quat orient;
    sphere_collider_of(tr, b.pos, b.radius, orient, b.offset);
    set_orientation(b, orient);
    return b;
}

inline void box_collider_of(const transform* tr, vec3& pos, vec3& half, quat& orient, vec3& offset) {
    aabb c = tr->child->bounding_box();                     // child's local bounding box
    vec3 lo(c.x.min, c.y.min, c.z.min), hi(c.x.max, c.y.max, c.z.max);
    half   = (hi - lo) * real(0.5) * tr->scale;
    offset = (lo + hi) * real(0.5) * tr->scale;
    pos    = tr->apply_R(offset) + tr->translation;
    orient = quat_from_euler_zyx_degrees(tr->rotation);
}

// inverse of the centre map above with the body's CURRENT orientation:
// T = pos - R(orient)·offset
inline vec3 transform_translation_of(const phys_body& b) {
    return b.pos - (b.axes[0] * b.offset.x() + b.axes[1] * b.offset.y() + b.axes[2] * b.offset.z());
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
    box_collider_of(tr, b.pos, b.half, orient, b.offset);
    set_orientation(b, orient);
    return b;
}

// collect all vertices of the hittable object
inline void collect_vertices(const hittable* h, std::vector<vec3>& out) {
    const size_t first = out.size();
    switch (h->type) {
    case QUAD: {
        const quad* q = static_cast<const quad*>(h->object);
        out.push_back(q->Q); out.push_back(q->Q + q->u); out.push_back(q->Q + q->v); out.push_back(q->Q + q->u + q->v);
        break;
    }
    case TRIANGLE: {
        const triangle* t = static_cast<const triangle*>(h->object);
        out.push_back(t->v0); out.push_back(t->v1); out.push_back(t->v2);
        break;
    }
    case HITTABLE_LIST: {
        const hittable_list* l = static_cast<const hittable_list*>(h->object);
        for (int i = 0; i < l->size; i++) collect_vertices(l->objects[i], out);
        break;
    }
    case BVH: {
        const bvh* b = static_cast<const bvh*>(h->object);
        for (int i = 0; i < b->prim_count; i++) collect_vertices(&b->prims[i], out);
        break;
    }
    case TRANSFORM: {
        const transform* t = static_cast<const transform*>(h->object);
        collect_vertices(t->child, out);
        for (size_t i = first; i < out.size(); i++) out[i] = t->apply_R(out[i] * t->scale) + t->translation;
        break;
    }
    default:
        std::cerr << "make_hull_body: HittableType not supported.\n";
        std::exit(1);
    }
}

// the hull of the object's scaled vertices about its centre of mass; offset = that centre (c·S), as for a box
inline void hull_collider_of(const transform* tr, hull_shape& h, vec3& pos, quat& orient, vec3& offset) {
    std::vector<vec3> pts;
    collect_vertices(tr->child, pts);

    for (vec3& v : pts) v *= tr->scale;
    
    offset = build_hull(pts, h);
    
    pos    = tr->apply_R(offset) + tr->translation;
    orient = quat_from_euler_zyx_degrees(tr->rotation);
}

// make a hull body from a polygonal object
inline phys_body make_hull_body(scene& sc, int scene_id, motion_type motion = DYNAMIC,
                                real mass = real(1),
                                real friction = real(0.5), real restitution = real(0.7)) {
    transform* tr = get_body_transform(sc, scene_id);
    phys_body b{ scene_id, vec3(0,0,0), vec3(0,0,0), tr->scale };
    b.motion = motion;  b.mass = mass;
    b.shape  = COLLIDER_HULL;
    b.friction = friction;  b.restitution = restitution;
    quat orient;
    hull_collider_of(tr, b.hull, b.pos, orient, b.offset);
    set_orientation(b, orient);
    return b;
}

#endif // VIEWER_PHYSICS_UTILS_H
