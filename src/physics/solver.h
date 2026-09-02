#ifndef PHYSICS_SOLVER_H
#define PHYSICS_SOLVER_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "physics/body.h"
#include "physics/contact.h"
#include "physics/dynamics.h"
#include "physics/math.h"
#include "precision.h"   // real
#include "vec3.h"

static constexpr int  SEQ_VEL_ITERS = 8;            // velocity iterations
static constexpr int  SEQ_POS_ITERS = 4;            // position correction iterations
static constexpr real SEQ_POS_BETA  = real(0.8);    // fraction of penetration corrected per position iteration
static constexpr real SEQ_POS_SLOP  = real(1e-4);   // penetration tolerance
static constexpr real SEQ_REST_VEL  = real(0.5);    // approach speed tolerance

inline void solve_sequential(std::vector<phys_body>& bodies, const phys_params& p) {
    // build contact list
    std::vector<contact> C;
    build_contacts(bodies, C);
    const std::size_t n = C.size();

    // cache
    std::vector<real> vbias(n);             // target separating velocity
    std::vector<real> jn(n, real(0));       // accumulated normal impulse
    std::vector<real> jt(n, real(0));       // accumulated sliding-friction impulse
    std::vector<real> jr(n, real(0));       // accumulated rolling-resistance angular impulse
    std::vector<real> js(n, real(0));       // accumulated spinning-resistance angular impulse
    std::vector<real> mu(n);                // combined sliding-friction coefficient
    std::vector<real> mur(n);               // combined rolling-friction coefficient
    std::vector<real> mus(n);               // combined spinning-friction coefficient
    std::vector<real> rad(n);               // rolling radius supplied by contacting spheres
    std::vector<vec3> tang(n);              // fixed sliding direction in the contact plane
    std::vector<vec3> roll(n);              // fixed rolling axis in the contact plane
    std::vector<vec3> lev_a(n);             // lever arm from body A's center to contact point
    std::vector<vec3> lev_b(n);             // lever arm from body B's center to contact point
    for (std::size_t c = 0; c < n; c++) {
        const contact& k = C[c];

        // lever arm from body's center to contact point
        lev_a[c] = contact_lever(bodies[k.a], k.n, k.p, true);
        lev_b[c] = contact_lever(bodies[k.b], k.n, k.p, false);

        // normal impulse
        vec3 vrel = velocity_at(bodies[k.a], lev_a[c]) - velocity_at(bodies[k.b], lev_b[c]);    // relative velocity at contact point
        real vn = dot(vrel, k.n);   // relative normal velocity
        real e   = combine(bodies[k.a].restitution, bodies[k.b].restitution, p.restitution_combine);
        vbias[c] = vn < -SEQ_REST_VEL ? -e * vn : real(0);  // target separating velocity

        // friction impulse
        mu[c]    = combine(bodies[k.a].friction, bodies[k.b].friction, p.friction_combine); // combined friction coefficient
        tang[c]  = tangent_from(vrel, k.n); // friction direction

        // rolling and spinning resistance for sphere-object contact
        mur[c]   = combine(bodies[k.a].rolling_friction, bodies[k.b].rolling_friction,
                           p.friction_combine);
        mus[c]   = combine(bodies[k.a].spinning_friction, bodies[k.b].spinning_friction,
                           p.friction_combine);
        // rolling radius
        rad[c]   = real(0);
        if (bodies[k.a].shape == COLLIDER_SPHERE) 
            rad[c] = bodies[k.a].radius;
        if (bodies[k.b].shape == COLLIDER_SPHERE && bodies[k.b].radius > rad[c])
            rad[c] = bodies[k.b].radius;
        // rolling axis
        roll[c]  = tangent_from(bodies[k.a].omega - bodies[k.b].omega, k.n);
    }

    // (1) velocity: accumulated normal + friction impulses
    for (int it = 0; it < SEQ_VEL_ITERS; it++)
        for (std::size_t c = 0; c < n; c++) {
            const contact& k = C[c];
            phys_body& A = bodies[k.a];
            phys_body& B = bodies[k.b];
            const real ima = inv_mass(A), imb = inv_mass(B);
            if (ima + imb <= real(0)) continue;         // neither can move
            const vec3& ra = lev_a[c];
            const vec3& rb = lev_b[c];

            // normal impulse
            const real kn = inv_effective_mass(A, B, ra, rb, k.n);
            if (kn > real(0)) {
                vec3 vrel = velocity_at(A, ra) - velocity_at(B, rb);    // current relative velocity at contact point
                real vn = dot(vrel, k.n);    // current relative normal velocity
                real jn_new = jn[c] + (vbias[c] - vn) / kn;
                if (jn_new < 0) jn_new = real(0);   // no sticking
                vec3 imp = k.n * (jn_new - jn[c]);  // impulse change to apply 
                jn[c] = jn_new;
                A.vel += imp * ima;   A.omega += delta_omega(A,  cross(ra, imp));
                B.vel -= imp * imb;   B.omega -= delta_omega(B,  cross(rb, imp));
            }

            // friction impulse
            const real kt = inv_effective_mass(A, B, ra, rb, tang[c]);
            if (mu[c] > real(0) && kt > real(0)) {
                vec3 vrel = velocity_at(A, ra) - velocity_at(B, rb);
                real vt   = dot(vrel, tang[c]);
                real jt_new = jt[c] + (0 - vt) / kt;
                real lim  = mu[c] * jn[c];  // maximum permitted friction impulse
                if (jt_new >  lim) jt_new =  lim;
                if (jt_new < -lim) jt_new = -lim;
                vec3 impt = tang[c] * (jt_new - jt[c]);
                jt[c] = jt_new;
                A.vel += impt * ima;  A.omega += delta_omega(A, cross(ra, impt));
                B.vel -= impt * imb;  B.omega -= delta_omega(B, cross(rb, impt));
            }

            // rolling and spinning resistance for sphere-object contact
            if (rad[c] > real(0)) {
                const vec3 wrel = A.omega - B.omega;
                if (mur[c] > real(0)) {                 // rolling: orthogonal to n
                    real kw = dot(roll[c], delta_omega(A, roll[c]))
                            + dot(roll[c], delta_omega(B, roll[c]));
                    if (kw > real(0)) {
                        real jr_new = jr[c] - dot(wrel, roll[c]) / kw;
                        real lim = mur[c] * jn[c] * rad[c];
                        if (jr_new >  lim) jr_new =  lim;
                        if (jr_new < -lim) jr_new = -lim;
                        vec3 impr = roll[c] * (jr_new - jr[c]);
                        jr[c] = jr_new;
                        A.omega += delta_omega(A, impr);
                        B.omega -= delta_omega(B, impr);
                    }
                }
                if (mus[c] > real(0)) {                 // spinning: paralle to n
                    real kw = dot(k.n, delta_omega(A, k.n))
                            + dot(k.n, delta_omega(B, k.n));
                    if (kw > real(0)) {
                        real js_new = js[c] - dot(wrel, k.n) / kw;
                        real lim = mus[c] * jn[c] * rad[c];
                        if (js_new >  lim) js_new =  lim;
                        if (js_new < -lim) js_new = -lim;
                        vec3 imps = k.n * (js_new - js[c]);
                        js[c] = js_new;
                        A.omega += delta_omega(A, imps);
                        B.omega -= delta_omega(B, imps);
                    }
                }
            }
        }

    // (2) position
    for (int it = 0; it < SEQ_POS_ITERS; it++) {
        build_contacts(bodies, C);
        for (std::size_t c = 0; c < C.size(); c++) {
            const contact& k = C[c];
            real corr = (k.pen - SEQ_POS_SLOP) * SEQ_POS_BETA / real(k.mcount);
            if (corr <= 0) continue;
            real ima = inv_mass(bodies[k.a]);
            real imb = inv_mass(bodies[k.b]);
            real invSum = ima + imb;
            if (invSum <= real(0)) continue;
            vec3 push = k.n * (corr / invSum);
            bodies[k.a].pos += push * ima;
            bodies[k.b].pos -= push * imb;
        }
    }
}

#endif // PHYSICS_SOLVER_H
