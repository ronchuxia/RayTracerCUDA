#ifndef PHYSICS_CONTACT_H
#define PHYSICS_CONTACT_H

#include <cstddef>
#include <vector>

#include "physics/body.h"
#include "physics/detect.h"
#include "physics/dynamics.h"
#include "physics/manifold.h"
#include "precision.h"
#include "vec3.h"

// ---- contact abstraction -------------------------------------------------

struct contact {
    int  a, b;
    vec3 n; // points from b toward a
    real pen;
    vec3 p;
    int  mcount;
};

// ---- contact list --------------------------------------------------------

inline void build_contacts(const std::vector<phys_body>& bodies, std::vector<contact>& out) {
    out.clear();

    // broad phase: TODO

    // narrow phase: brute-force pair enumeration
    for (std::size_t i = 0; i < bodies.size(); i++)
        for (std::size_t j = i + 1; j < bodies.size(); j++) {
            // skip non-collidable and immovable bodies
            if (!bodies[i].collidable || !bodies[j].collidable) continue;
            if (inv_mass(bodies[i]) + inv_mass(bodies[j]) <= real(0)) continue;

            // canonicalize for the analytical path: set operand A to be COLLIDER_SPHERE
            int a = (int)i, b = (int)j;
            if (bodies[a].shape != COLLIDER_SPHERE && bodies[b].shape == COLLIDER_SPHERE) {
                a = (int)j;
                b = (int)i;
            }

            // detect contact, then expand it into a manifold
            vec3 n; real pen;
            if (contact_detect(bodies[a], bodies[b], n, pen)) {
                vec3 pts[4]; real pens[4];
                const int m = contact_manifold(bodies[a], bodies[b], n, pen, pts, pens);
                for (int q = 0; q < m; q++)
                    out.push_back({ a, b, n, pens[q], pts[q], m });
            }
        }
}

#endif // PHYSICS_CONTACT_H
