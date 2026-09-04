#ifndef PHYSICS_BODY_H
#define PHYSICS_BODY_H

#include <cmath>

#include "precision.h"   // real
#include "quat.h"        // orientation
#include "vec3.h"

enum motion_type { STATIC, KINEMATIC, DYNAMIC };

enum collider_type { COLLIDER_SPHERE, COLLIDER_BOX };

struct phys_body {
    int  scene_id;
    vec3 pos, vel;
    vec3 scale;
    quat orient;

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
};

// set the orientation and derive its x/y/z axes in world space
inline void set_orientation(phys_body& b, const quat& q) {
    b.orient  = normalize(q);
    b.axes[0] = quat_axis(b.orient, 0);
    b.axes[1] = quat_axis(b.orient, 1);
    b.axes[2] = quat_axis(b.orient, 2);
}

#endif // PHYSICS_BODY_H
