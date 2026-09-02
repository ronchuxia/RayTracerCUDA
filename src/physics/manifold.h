#ifndef PHYSICS_MANIFOLD_H
#define PHYSICS_MANIFOLD_H

#include <cmath>

#include "physics/body.h"
#include "physics/gjk.h"   // support(), for the fallback point
#include "precision.h"     // real
#include "vec3.h"

// check the box’s x/y/z axes in world space and chooses the one most parallel to dir
inline real box_best_face(const phys_body& b, const vec3& dir, int& axis, real& sign) {
    real best = real(-1);
    axis = 0; sign = real(1);
    for (int i = 0; i < 3; i++) {
        const real d = dot(b.axes[i], dir);
        const real a = d < real(0) ? -d : d;
        if (a > best) { best = a; axis = i; sign = d < real(0) ? real(-1) : real(1); }
    }
    return best;
}

// Sutherland-Hodgman algorithm.
// `in`: input polygon vertices.
// `n`: number of input vertices.
// `nrm`, `off`: clipping half-space is dot(p, nrm) <= off.
// `out`: ouput clipped polygon vertices.
// returns: number of output vertices.
inline int clip_polygon(const vec3* in, int n, const vec3& nrm, real off, vec3* out) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        const vec3& p = in[i];
        const vec3& q = in[(i + 1) % n];
        const real dp = dot(p, nrm) - off;
        const real dq = dot(q, nrm) - off;
        const bool pin = dp <= real(0), qin = dq <= real(0);
        if (pin) out[m++] = p;
        if (pin != qin) {
            const real t = dp / (dp - dq);
            out[m++] = p + (q - p) * t;
        }
    }
    return m;
}

inline int box_box_manifold(const phys_body& A, const phys_body& B, const vec3& n,
                            vec3 pts[4], real pens[4]) {
    // reference face: the face whose normal points most parallel with the collision normal
    int  ai, bi;
    real as, bs;
    const real align_a = box_best_face(A, -n, ai, as);
    const real align_b = box_best_face(B,  n, bi, bs);
    const bool ref_is_a = align_a >= align_b;   // the box with larger alignment has the face most square-on to the collision
    const phys_body& R = ref_is_a ? A : B;  // reference box
    const phys_body& I = ref_is_a ? B : A;  // incident box
    const int  raxis = ref_is_a ? ai : bi;  // reference face's axis
    const real rsign = ref_is_a ? as : bs;  // reference face's sign: positive or negative face on that axis

    const vec3 rn = R.axes[raxis] * rsign;                 // reference face's outward normal
    const vec3 rc = R.pos + rn * R.half[raxis];            // reference face's centre
    const int  rj = (raxis + 1) % 3, rk = (raxis + 2) % 3; // reference face's in-plane axes

    // incident face: the face of the other box whose normal points most opposite to rn
    int  iaxis;
    real isign;
    box_best_face(I, -rn, iaxis, isign);
    const vec3 ic = I.pos + I.axes[iaxis] * (isign * I.half[iaxis]);    // incident face's centre
    const int  ij = (iaxis + 1) % 3, ik = (iaxis + 2) % 3;  // incident face's in-plane axes
    const vec3 u = I.axes[ij] * I.half[ij], v = I.axes[ik] * I.half[ik];

    // a quad clipped by four planes can have at most 8 vertices
    vec3 buf0[8], buf1[8];
    // incident face's corners
    buf0[0] = ic + u + v;  buf0[1] = ic + u - v;
    buf0[2] = ic - u - v;  buf0[3] = ic - u + v;
    int np = 4;

    // clip against the reference face's four side planes
    const vec3 side[2] = { R.axes[rj], R.axes[rk] };
    const real ext[2]  = { R.half[rj], R.half[rk] };
    vec3* src = buf0; vec3* dst = buf1;
    vec3* t;
    for (int s = 0; s < 2 && np > 0; s++) {
        np = clip_polygon(src, np, side[s],  dot(rc, side[s]) + ext[s], dst);
        t = src; src = dst; dst = t;
        if (np == 0) break;
        np = clip_polygon(src, np, -side[s], -dot(rc, side[s]) + ext[s], dst);
        t = src; src = dst; dst = t;
    }

    // keep the 4 deepest points that are penetrating the reference face
    int cnt = 0;
    for (int i = 0; i < np; i++) {
        const real d = dot(src[i] - rc, rn);
        if (d > real(0)) continue;                 // not penetrating
        const real pen = -d;
        if (cnt < 4) { pts[cnt] = src[i]; pens[cnt] = pen; cnt++; continue; }
        int shallow = 0;
        for (int q = 1; q < 4; q++) if (pens[q] < pens[shallow]) shallow = q;
        if (pen > pens[shallow]) { pts[shallow] = src[i]; pens[shallow] = pen; }
    }
    return cnt;
}

inline int contact_manifold(const phys_body& A, const phys_body& B,
                            const vec3& n, real pen, vec3 pts[4], real pens[4]) {
    if (A.shape == COLLIDER_SPHERE) { pts[0] = A.pos - n * A.radius; pens[0] = pen; return 1; }
    if (B.shape == COLLIDER_SPHERE) { pts[0] = B.pos + n * B.radius; pens[0] = pen; return 1; }
    if (A.shape == COLLIDER_BOX && B.shape == COLLIDER_BOX) {
        const int c = box_box_manifold(A, B, n, pts, pens);
        if (c > 0) return c;
    }
    pts[0] = support(A, -n); pens[0] = pen; return 1;
}

#endif // PHYSICS_MANIFOLD_H
