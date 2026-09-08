#ifndef PHYSICS_BODY_H
#define PHYSICS_BODY_H

#include <cmath>
#include <vector>

#include "precision.h"   // real
#include "quat.h"        // orientation
#include "vec3.h"

enum motion_type { STATIC, KINEMATIC, DYNAMIC };

enum collider_type { COLLIDER_SPHERE, COLLIDER_BOX, COLLIDER_HULL };

static const int HULL_MAX_LOOP = 32;   // max vertices per face loop

struct hull_shape {
    struct face { 
        vec3 normal; 
        std::vector<int> loop;
    };

    std::vector<vec3> verts;
    std::vector<face> faces;
    real radius = real(0);          // farthest vertex from the centre
    vec3 inv_inertia[3];            // inverse inertia tensor about the centre of mass, per unit mass
};

inline vec3 hull_mass_properties(hull_shape& h) {
    real I[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };   // ∫1, ∫x, ∫y, ∫z, ∫x², ∫y², ∫z², ∫xy, ∫yz, ∫zx
    
    // triangle integration
    for (const hull_shape::face& f : h.faces)
        for (size_t k = 1; k + 1 < f.loop.size(); k++) {
            const vec3& p0 = h.verts[f.loop[0]]; 
            const vec3& p1 = h.verts[f.loop[k]]; 
            const vec3& p2 = h.verts[f.loop[k + 1]];

            const vec3 d = cross(p1 - p0, p2 - p0);
            real f1[3], f2[3], f3[3], g0[3], g1[3], g2[3];
            for (int a = 0; a < 3; a++) {
                const real w0 = p0[a], w1 = p1[a], w2 = p2[a];
                const real t0 = w0 + w1, t1 = w0 * w0, t2 = t1 + w1 * t0;
                f1[a] = t0 + w2; 
                f2[a] = t2 + w2 * f1[a]; 
                f3[a] = w0 * t1 + w1 * t2 + w2 * f2[a];
                g0[a] = f2[a] + w0 * (f1[a] + w0); 
                g1[a] = f2[a] + w1 * (f1[a] + w1); 
                g2[a] = f2[a] + w2 * (f1[a] + w2);
            }
            I[0] += d[0] * f1[0];
            I[1] += d[0] * f2[0]; I[2] += d[1] * f2[1]; I[3] += d[2] * f2[2];
            I[4] += d[0] * f3[0]; I[5] += d[1] * f3[1]; I[6] += d[2] * f3[2];
            I[7] += d[0] * (p0[1] * g0[0] + p1[1] * g1[0] + p2[1] * g2[0]);
            I[8] += d[1] * (p0[2] * g0[1] + p1[2] * g1[1] + p2[2] * g2[1]);
            I[9] += d[2] * (p0[0] * g0[2] + p1[0] * g1[2] + p2[0] * g2[2]);
        }
    const real scale[10] = { real(1) / 6, real(1) / 24, real(1) / 24, real(1) / 24, real(1) / 60, real(1) / 60, real(1) / 60, real(1) / 120, real(1) / 120, real(1) / 120 };
    for (int i = 0; i < 10; i++) I[i] *= scale[i];

    const real vol = I[0];

    const vec3 com(I[1] / vol, I[2] / vol, I[3] / vol);

    // inertia about the centre of mass, per unit mass 
    const real xx = (I[5] + I[6]) / vol - (com[1] * com[1] + com[2] * com[2]);
    const real yy = (I[4] + I[6]) / vol - (com[2] * com[2] + com[0] * com[0]);
    const real zz = (I[4] + I[5]) / vol - (com[0] * com[0] + com[1] * com[1]);
    const real xy = -I[7] / vol + com[0] * com[1];
    const real yz = -I[8] / vol + com[1] * com[2];
    const real zx = -I[9] / vol + com[2] * com[0];

    const real det = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * zx) + zx * (xy * yz - yy * zx);
    h.inv_inertia[0] = vec3(yy * zz - yz * yz, zx * yz - xy * zz, xy * yz - zx * yy) / det;
    h.inv_inertia[1] = vec3(zx * yz - xy * zz, xx * zz - zx * zx, xy * zx - xx * yz) / det;
    h.inv_inertia[2] = vec3(xy * yz - zx * yy, xy * zx - xx * yz, xx * yy - xy * xy) / det;
    
    // re-center around com
    h.radius = 0;
    for (vec3& v : h.verts) { 
        v -= com; 
        if (v.length() > h.radius) h.radius = v.length(); 
    }
    
    return com;
}

struct phys_body {
    int  scene_id;
    vec3 pos;                       // collider center from world's origin, in world's axes
    vec3 vel;
    vec3 scale;
    quat orient;
    vec3 offset = vec3(0, 0, 0);    // collider center from transform's origin, in transform's axes, scaled

    // orient in matrix form, a derived cache of a box's x/y/z axes in world space
    vec3          axes[3]    = { vec3(1,0,0), vec3(0,1,0), vec3(0,0,1) };
    
    motion_type   motion     = DYNAMIC;
    bool          collidable = true;
    real          mass       = real(1);
    real          restitution= real(0.7);
    real          friction   = real(0.5);
    real          rolling_friction= real(0.01);
    real          spinning_friction = real(0.002);

    vec3          omega      = vec3(0, 0, 0);

    collider_type shape      = COLLIDER_SPHERE;

    // -- COLLIDER_SPHERE --
    real          radius     = real(0);

    // -- COLLIDER_BOX --
    vec3          half       = vec3(0, 0, 0);   // the box's half extents

    // -- COLLIDER_HULL --
    const hull_shape* hull   = nullptr;
};

// set the orientation and derive its x/y/z axes in world space
inline void set_orientation(phys_body& b, const quat& q) {
    b.orient  = normalize(q);
    b.axes[0] = quat_axis(b.orient, 0);
    b.axes[1] = quat_axis(b.orient, 1);
    b.axes[2] = quat_axis(b.orient, 2);
}

#endif // PHYSICS_BODY_H
