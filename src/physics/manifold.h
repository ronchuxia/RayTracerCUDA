#ifndef PHYSICS_MANIFOLD_H
#define PHYSICS_MANIFOLD_H

#include <cmath>

#include "physics/body.h"
#include "physics/gjk.h"   // support(), for the fallback point
#include "precision.h"     // real
#include "vec3.h"

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

inline int face_count(const phys_body& b) { 
    return b.shape == COLLIDER_HULL ? (int)b.hull.faces.size() : 6;
}

inline vec3 face_normal(const phys_body& b, int f) {
    if (b.shape == COLLIDER_HULL) {
        const vec3& n = b.hull.faces[f].normal;
        return b.axes[0] * n[0] + b.axes[1] * n[1] + b.axes[2] * n[2];
    }
    // COLLIDER_BOX
    return b.axes[f / 2] * (f & 1 ? real(-1) : real(1));
}

// return vertices of a face in world space
inline int face_loop(const phys_body& b, int f, vec3* out) {
    if (b.shape == COLLIDER_HULL) {
        const hull_shape::face& fc = b.hull.faces[f];
        for (size_t i = 0; i < fc.loop.size(); i++) {
            const vec3& v = b.hull.verts[fc.loop[i]];
            out[i] = b.pos + b.axes[0] * v[0] + b.axes[1] * v[1] + b.axes[2] * v[2];
        }
        return (int)fc.loop.size();
    }
    // COLLIDER_BOX
    const int  a = f / 2, j = (a + 1) % 3, k = (a + 2) % 3;   // axes[j] x axes[k] = axes[a]
    const real s = f & 1 ? real(-1) : real(1);
    const vec3 c = b.pos + b.axes[a] * (s * b.half[a]), u = b.axes[j] * b.half[j], v = b.axes[k] * b.half[k];
    out[0] = c + u + v; out[2] = c - u - v;
    out[1] = s > real(0) ? c - u + v : c + u - v;
    out[3] = s > real(0) ? c + u - v : c - u + v;
    return 4;
}

// find the face whose outward normal is most aligned with dir
inline real best_face(const phys_body& b, const vec3& dir, int& face) {
    real best_alignment = real(-2);
    face = 0;
    for (int f = 0, nf = face_count(b); f < nf; f++) {
        const real alignment = dot(face_normal(b, f), dir);
        if (alignment > best_alignment) { best_alignment = alignment; face = f; }
    }
    return best_alignment;
}

inline int face_manifold(const phys_body& A, const phys_body& B, const vec3& n,
                         vec3 pts[4], real pens[4]) {
    // reference face: the face whose normal points most parallel with the collision normal
    int face_a, face_b;
    const real align_a = best_face(A, -n, face_a);
    const real align_b = best_face(B,  n, face_b);
    const bool ref_is_a = align_a >= align_b;   // the body with larger alignment has the face most square-on to the collision
    const phys_body& R = ref_is_a ? A : B;      // reference body
    const phys_body& I = ref_is_a ? B : A;      // incident body
    const int  ref_face = ref_is_a ? face_a : face_b;         // reference face
    const vec3 ref_normal = face_normal(R, ref_face);         // reference face's outward normal

    // incident face: the face of the other body whose normal points most opposite to the reference normal
    int inc_face;
    best_face(I, -ref_normal, inc_face);

    // clip the incident face against the reference face's side planes. each plane adds at most one vertex
    vec3 ref_loop[HULL_MAX_LOOP], buf0[2 * HULL_MAX_LOOP], buf1[2 * HULL_MAX_LOOP];
    const int ref_count = face_loop(R, ref_face, ref_loop);
    int poly_count = face_loop(I, inc_face, buf0);
    vec3* src = buf0; vec3* dst = buf1;
    for (int e = 0; e < ref_count && poly_count > 0; e++) {
        const vec3 side_normal = cross(ref_loop[(e + 1) % ref_count] - ref_loop[e], ref_normal);   // outward for a counter-clockwise loop
        poly_count = clip_polygon(src, poly_count, side_normal, dot(ref_loop[e], side_normal), dst);
        vec3* t = src; src = dst; dst = t;
    }

    // keep the points that are penetrating the reference face
    real depth[2 * HULL_MAX_LOOP];
    int cnt = 0;
    for (int i = 0; i < poly_count; i++) {
        const real pen = dot(ref_loop[0] - src[i], ref_normal);
        if (pen < real(0)) continue;               // not penetrating
        src[cnt] = src[i]; 
        depth[cnt] = pen; 
        cnt++;
    }
    if (cnt <= 4) {
        for (int i = 0; i < cnt; i++) { 
            pts[i] = src[i]; 
            pens[i] = depth[i]; 
        }
        return cnt;
    }

    // keep 4 points: for a resting body, the contact points must enclose the projection of the centre of mass
    int k[4];
    // 1st point: the deepest point
    k[0] = 0;
    for (int i = 1; i < cnt; i++) if (depth[i] > depth[k[0]]) k[0] = i;
    // 2nd point: the point farthest from the first point
    k[1] = 0;
    real best = real(-1);
    for (int i = 0; i < cnt; i++) {
        const real l = (src[i] - src[k[0]]).length_squared();
        if (l > best) { best = l; k[1] = i; }
    }
    // 3rd point: the point forming the largest triangle (projected onto the reference plane) with the first two points
    k[2] = 0; 
    best = real(-1);
    for (int i = 0; i < cnt; i++) {
        const real a = dot(cross(src[k[1]] - src[k[0]], src[i] - src[k[0]]), ref_normal);
        const real m = a < real(0) ? -a : a;
        if (m > best) { best = m; k[2] = i; }
    }
    // 4th point: the point that expands the triangle the most
    k[3] = 0; 
    best = real(-1);
    for (int t = 0; t < 3; t++) {
        const vec3& a = src[k[t]];
        const vec3  ab = src[k[(t + 1) % 3]] - a;
        const real  side = dot(cross(ab, src[k[(t + 2) % 3]] - a), ref_normal) < real(0) ? real(1) : real(-1);
        for (int i = 0; i < cnt; i++) {
            const real gain = side * dot(cross(ab, src[i] - a), ref_normal);   // area outside the edge
            if (gain > best) { best = gain; k[3] = i; }
        }
    }

    for (int i = 0; i < 4; i++) { 
        pts[i] = src[k[i]]; 
        pens[i] = depth[k[i]]; 
    }
    return 4;
}

inline int contact_manifold(const phys_body& A, const phys_body& B,
                            const vec3& n, real pen, vec3 pts[4], real pens[4]) {
    if (A.shape == COLLIDER_SPHERE) { pts[0] = A.pos - n * A.radius; pens[0] = pen; return 1; }
    if (B.shape == COLLIDER_SPHERE) { pts[0] = B.pos + n * B.radius; pens[0] = pen; return 1; }
    const int c = face_manifold(A, B, n, pts, pens);
    if (c > 0) return c;
    pts[0] = support(A, -n); pens[0] = pen; return 1;
}

#endif // PHYSICS_MANIFOLD_H
