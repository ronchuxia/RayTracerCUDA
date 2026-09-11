#ifndef VIEWER_PHYSICS_UTILS_H
#define VIEWER_PHYSICS_UTILS_H

#include <cstdlib>
#include <iostream>

#include "hittable.h"
#include "physics/body.h"
#include "physics/hull.h"
#include "viewer/scene.h"

inline instance* get_body_instance(scene& sc, int scene_id) {
    instance* in = sc.get(scene_id);
    if (!in) {
        std::cerr << "scene object " << scene_id << " does not exist\n";
        std::exit(1);
    }
    return in;
}

inline void sphere_collider_of(const instance* in, vec3& pos, real& radius, quat& orient, vec3& offset) {
    const sphere* sp = static_cast<const sphere*>(in->prim_ptr()->object);
    radius = sp->radius * in->xf.scale.y();
    offset = sp->center * in->xf.scale;
    pos    = in->xf.apply_R(offset) + in->xf.translation;
    orient = quat_from_euler_zyx_degrees(in->xf.rotation);
}

// make a sphere body from a sphere object
inline phys_body make_sphere_body(scene& sc, int scene_id, motion_type motion = DYNAMIC,
                                  real mass = real(1),
                                  real friction = real(0.5), real restitution = real(0.7)) {
    instance* in = get_body_instance(sc, scene_id);
    if (in->leaf_type != LEAF_PRIMITIVE || in->prim_ptr()->type != SPHERE) {
        std::cerr << "scene object " << scene_id << " is not a sphere\n";
        std::exit(1);
    }
    if (in->xf.scale.x() != in->xf.scale.y() || in->xf.scale.y() != in->xf.scale.z()) {
        std::cerr << "scene object " << scene_id << " needs uniform scale\n";
        std::exit(1);
    }
    phys_body b{ scene_id, vec3(0,0,0), vec3(0,0,0), in->xf.scale };
    b.motion = motion;  b.mass = mass;
    b.shape  = COLLIDER_SPHERE;
    b.friction = friction;  b.restitution = restitution;
    // derive the sphere collider from the object
    quat orient;
    sphere_collider_of(in, b.pos, b.radius, orient, b.offset);
    set_orientation(b, orient);
    return b;
}

inline void box_collider_of(const instance* in, vec3& pos, vec3& half, quat& orient, vec3& offset) {
    aabb c = in->leaf_bounding_box();                               // leaf's local bounding box
    vec3 lo(c.x.min, c.y.min, c.z.min), hi(c.x.max, c.y.max, c.z.max);
    half   = (hi - lo) * real(0.5) * in->xf.scale;
    offset = (lo + hi) * real(0.5) * in->xf.scale;
    pos    = in->xf.apply_R(offset) + in->xf.translation;
    orient = quat_from_euler_zyx_degrees(in->xf.rotation);
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
    instance* in = get_body_instance(sc, scene_id);
    phys_body b{ scene_id, vec3(0,0,0), vec3(0,0,0), in->xf.scale };
    b.motion = motion;  b.mass = mass;
    b.shape  = COLLIDER_BOX;
    b.friction = friction;  b.restitution = restitution;
    // derive the box collider from the object
    quat orient;
    box_collider_of(in, b.pos, b.half, orient, b.offset);
    set_orientation(b, orient);
    return b;
}

// collect all vertices of a primitive
inline void collect_vertices(const primitive& p, std::vector<vec3>& out) {
    switch (p.type) {
    case QUAD: {
        const quad* q = static_cast<const quad*>(p.object);
        out.push_back(q->Q); out.push_back(q->Q + q->u); out.push_back(q->Q + q->v); out.push_back(q->Q + q->u + q->v);
        break;
    }
    case TRIANGLE: {
        const triangle* t = static_cast<const triangle*>(p.object);
        out.push_back(t->v0); out.push_back(t->v1); out.push_back(t->v2);
        break;
    }
    default:
        std::cerr << "make_hull_body: a sphere has no vertices.\n";
        std::exit(1);
    }
}

// collect all vertices of an instance
inline void collect_vertices(const instance* in, std::vector<vec3>& out) {
    if (in->leaf_type == LEAF_PRIMITIVE) {
        collect_vertices(*in->prim_ptr(), out);
    } else {
        const mesh* m = in->mesh_ptr();
        for (int i = 0; i < m->item_count; i++) collect_vertices(m->items[i], out);
    }
}

// the hull of the object's scaled vertices about its centre of mass
inline void hull_collider_of(const instance* in, hull_shape& h, vec3& pos, quat& orient, vec3& offset) {
    std::vector<vec3> pts;
    collect_vertices(in, pts);

    for (vec3& v : pts) v *= in->xf.scale;

    offset = build_hull(pts, h);
    pos    = in->xf.apply_R(offset) + in->xf.translation;
    orient = quat_from_euler_zyx_degrees(in->xf.rotation);
}

// make a hull body from a polygonal object
inline phys_body make_hull_body(scene& sc, int scene_id, motion_type motion = DYNAMIC,
                                real mass = real(1),
                                real friction = real(0.5), real restitution = real(0.7)) {
    instance* in = get_body_instance(sc, scene_id);
    phys_body b{ scene_id, vec3(0,0,0), vec3(0,0,0), in->xf.scale };
    b.motion = motion;  b.mass = mass;
    b.shape  = COLLIDER_HULL;
    b.friction = friction;  b.restitution = restitution;
    quat orient;
    hull_collider_of(in, b.hull, b.pos, orient, b.offset);
    set_orientation(b, orient);
    return b;
}

#endif // VIEWER_PHYSICS_UTILS_H
