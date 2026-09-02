#ifndef PHYSICS_DETECT_H
#define PHYSICS_DETECT_H

#include <cmath>

#include "physics/body.h"
#include "physics/gjk.h"
#include "physics/math.h"
#include "precision.h"
#include "vec3.h"

// ---- shape queries -------------------------------------------------------

inline bool sphere_sphere_contact(const phys_body& A, const phys_body& B, vec3& n, real& pen) {
    vec3 d = A.pos - B.pos;
    real dist2 = d.length_squared();
    real rsum = A.radius + B.radius;
    if (dist2 >= rsum * rsum || dist2 < real(1e-12)) return false;  // concentric spheres report NO contact
    real dist = std::sqrt(dist2);
    n = d / dist;
    pen = rsum - dist;
    return true;
}

inline bool sphere_aabb_contact(const vec3& c, real r, const vec3& bmin, const vec3& bmax,
                                vec3& n, real& pen) {
    vec3 q(clampr(c[0], bmin[0], bmax[0]),         // closest point on the box to the centre
           clampr(c[1], bmin[1], bmax[1]),
           clampr(c[2], bmin[2], bmax[2]));
    vec3 d = c - q;
    real dist2 = d.length_squared();
    if (dist2 > real(1e-12)) {                     // centre outside the box
        if (dist2 >= r * r) return false;
        real dist = std::sqrt(dist2);
        n = d / dist;                              // box surface -> centre
        pen = r - dist;
    } else {                                       // centre inside the box: eject along nearest face
        real ex = c[0]-bmin[0] < bmax[0]-c[0] ? -(c[0]-bmin[0]) : (bmax[0]-c[0]);
        real ey = c[1]-bmin[1] < bmax[1]-c[1] ? -(c[1]-bmin[1]) : (bmax[1]-c[1]);
        real ez = c[2]-bmin[2] < bmax[2]-c[2] ? -(c[2]-bmin[2]) : (bmax[2]-c[2]);
        real ax = ex<0?-ex:ex, ay = ey<0?-ey:ey, az = ez<0?-ez:ez;  // exit distances
        if (ax <= ay && ax <= az) { n = vec3(ex<0?real(-1):real(1), 0, 0); pen = r + ax; }
        else if (ay <= az)        { n = vec3(0, ey<0?real(-1):real(1), 0); pen = r + ay; }
        else                      { n = vec3(0, 0, ez<0?real(-1):real(1)); pen = r + az; }
    }
    return true;
}

inline bool sphere_box_contact(const phys_body& A, const phys_body& B, vec3& n, real& pen) {
    // transform sphere into box's frame
    vec3 d = A.pos - B.pos;
    vec3 c_local(dot(d, B.axes[0]), dot(d, B.axes[1]), dot(d, B.axes[2]));
    // compute sphere vs. aabb contact
    vec3 n_local;
    if (!sphere_aabb_contact(c_local, A.radius, -B.half, B.half, n_local, pen)) return false;
    // transform n back to world frame
    n = n_local.x() * B.axes[0] + n_local.y() * B.axes[1] + n_local.z() * B.axes[2];
    return true;
}

// ---- dispatch ------------------------------------------------------------

inline bool contact_detect(const phys_body& A, const phys_body& B, vec3& n, real& pen) {
    if (A.shape == COLLIDER_SPHERE && B.shape == COLLIDER_SPHERE)
        return sphere_sphere_contact(A, B, n, pen);
    if (A.shape == COLLIDER_SPHERE && B.shape == COLLIDER_BOX)
        return sphere_box_contact(A, B, n, pen);
    return gjk_epa_contact(A, B, n, pen);
}

#endif // PHYSICS_DETECT_H
