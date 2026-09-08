#ifndef PHYSICS_H
#define PHYSICS_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "physics/body.h"
#include "physics/contact.h"
#include "physics/detect.h"
#include "physics/dynamics.h"
#include "physics/gjk.h"
#include "physics/manifold.h"
#include "physics/math.h"
#include "physics/solver.h"
#include "precision.h"   // real
#include "quat.h"        // orientation
#include "vec3.h"

inline real physics_step(std::vector<phys_body>& bodies, const phys_params& p, real dt) {
    // apply gravity (on DYNAMIC bodies)
    for (phys_body& b : bodies) {
        // gate on inverse mass
        if (inv_mass(b) <= real(0)) continue;
        // integrate position
        b.vel[1] += p.gravity * dt;
        b.pos    += b.vel * dt;
        // integrate orientation
        if (b.omega.length_squared() > real(0))
            set_orientation(b, quat_integrate(b.orient, b.omega, dt));
    }

    solve_sequential(bodies, p);

    real maxv = 0;
    for (phys_body& b : bodies) {
        if (inv_mass(b) <= real(0)) continue;
        real v = b.vel.length();
        real r = b.shape == COLLIDER_BOX ? b.half.length() : b.shape == COLLIDER_HULL ? b.hull->radius : b.radius;
        real w = b.omega.length() * r;
        if (v > maxv) maxv = v;
        if (w > maxv) maxv = w;
    }
    return maxv;
}


#endif // PHYSICS_H
