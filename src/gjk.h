#ifndef GJK_H
#define GJK_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "precision.h"   // real
#include "vec3.h"

// CONVEX-CONVEX COLLISION DETECTION: support functions + GJK + EPA (workstream
// D, phase 3 B2). See docs/plans/physics.md.
//
// This file is included at the BOTTOM of physics.h, after `phys_body` is
// defined — the same wiring hittable.h uses for its composite shapes. physics.h
// forward-declares the two entry points it calls (`support` and
// `gjk_epa_contact`) near the top and the definitions land here, so the sim core
// stays readable and this algorithm stays separable. Nothing else includes it.
//
// WHY A SUPPORT FUNCTION. Writing one analytic test per shape pair does not
// scale: N shapes need N*(N+1)/2 tests, and each new shape rewrites the narrow
// phase. A support function collapses that to N. `support(dir)` answers one
// question — "which point of this shape is furthest along `dir`?" — and every
// convex shape can answer it in a few lines: a sphere by stepping one radius
// along the direction, a box by picking the extreme corner, a convex hull (B4)
// by scanning its vertices. GJK and EPA below then work for ANY pair, because
// they never look at a shape, only at what its support function returns.
//
// WHY THE MINKOWSKI DIFFERENCE. Both algorithms run on the set
// A (-) B = { a - b : a in A, b in B }, which is itself convex. Two shapes
// overlap exactly when that set contains the origin, and the distance from the
// origin to its boundary is the penetration depth. So a question about two
// shapes becomes a question about one set and one point. That set is never
// built: its support function is `support(A, d) - support(B, -d)`, which is why
// this costs two shape queries per iteration and no memory.
//
//   GJK (Gilbert-Johnson-Keerthi) answers the yes/no: it grows a simplex (up to
//     a tetrahedron) of points from that set, walking toward the origin, and
//     either encloses the origin (overlap) or finds a direction in which the set
//     cannot reach it (separated).
//   EPA (Expanding Polytope Algorithm) answers by how much: starting from GJK's
//     tetrahedron it repeatedly pushes the face nearest the origin outward until
//     it sits on the boundary. That face's normal and distance are the contact
//     normal and penetration depth.
//
// The analytic sphere-sphere and sphere-box tests in physics.h are kept as fast
// paths, on two measurements. EXACTNESS: on a curved surface this path is 0.36
// degrees off in the normal (see EPA_TOL below) where the closed form is exact.
// COST: on the identical sphere-box pair, 1354 ns through here against 8.1 ns
// analytic — 167x. On a BOX both paths are exact, so only the cost argument
// applies there; docs/plans/phase3-status.md records where that starts to matter.
// tests/test_physics.cu cross-checks the two against each other.

// ---- support functions ---------------------------------------------------

// The point of `b`'s collider furthest along `dir` (which need not be unit).
// This is the ONLY thing GJK and EPA know about a shape.
//
// A box is separable per axis: being furthest along `dir` overall means being
// furthest along each of its own three axes independently, so the answer is a
// corner picked one sign at a time. A sphere has the same answer in closed form
// — its centre plus one radius along the direction. A zero direction has no
// answer, so both fall back to the centre.
inline vec3 support(const phys_body& b, const vec3& dir) {
    if (b.shape == COLLIDER_SPHERE) {
        real len2 = dir.length_squared();
        if (len2 < real(1e-20)) return b.pos;
        return b.pos + dir * (b.radius / std::sqrt(len2));
    }
    vec3 p = b.pos;
    for (int i = 0; i < 3; i++)
        p += b.axes[i] * (dot(dir, b.axes[i]) >= real(0) ? b.half[i] : -b.half[i]);
    return p;
}

// Support of the Minkowski difference A (-) B. Note the NEGATED direction on B:
// the furthest point of `a - b` along `dir` pairs A's furthest point along `dir`
// with B's furthest point AGAINST it.
inline vec3 support_diff(const phys_body& A, const phys_body& B, const vec3& dir) {
    return support(A, dir) - support(B, -dir);
}

// ---- GJK -----------------------------------------------------------------

static constexpr int GJK_MAX_ITERS = 32;

// Up to four points of the Minkowski difference. p[0] is always the NEWEST — the
// case analysis below is written around "the point we just added", because that
// is the only one the origin can have moved relative to.
struct gjk_simplex {
    vec3 p[4];
    int  n;
};

static inline void gjk_set(gjk_simplex& s, const vec3& a) {
    s.p[0] = a; s.n = 1;
}
static inline void gjk_set(gjk_simplex& s, const vec3& a, const vec3& b) {
    s.p[0] = a; s.p[1] = b; s.n = 2;
}
static inline void gjk_set(gjk_simplex& s, const vec3& a, const vec3& b, const vec3& c) {
    s.p[0] = a; s.p[1] = b; s.p[2] = c; s.n = 3;
}

// Triple product cross(cross(a, b), a): the component of b perpendicular to a,
// scaled. Used to aim the next search direction sideways off an edge or face
// rather than along it.
static inline vec3 triple(const vec3& a, const vec3& b) {
    return cross(cross(a, b), a);
}

// Drop the parts of the simplex that cannot contain the closest point to the
// origin, and aim `dir` at the origin from what is left. Never returns true —
// only a tetrahedron can enclose the origin, and that is gjk_tetra's job.
static inline bool gjk_line(gjk_simplex& s, vec3& dir) {
    const vec3 a = s.p[0], b = s.p[1];      // a is newest
    const vec3 ab = b - a, ao = -a;
    if (dot(ab, ao) > real(0)) {            // origin beside the segment
        dir = triple(ab, ao);
        if (dir.length_squared() < real(1e-20))   // origin ON the segment
            dir = any_perpendicular(ab / ab.length());
    } else {                                // origin behind a: b is useless
        gjk_set(s, a);
        dir = ao;
    }
    return false;
}

static inline bool gjk_triangle(gjk_simplex& s, vec3& dir) {
    const vec3 a = s.p[0], b = s.p[1], c = s.p[2];
    const vec3 ab = b - a, ac = c - a, ao = -a;
    const vec3 abc = cross(ab, ac);         // the triangle's plane normal

    if (dot(cross(abc, ac), ao) > real(0)) {        // outside edge ac
        if (dot(ac, ao) > real(0)) { gjk_set(s, a, c); dir = triple(ac, ao); return false; }
        gjk_set(s, a, b);
        return gjk_line(s, dir);
    }
    if (dot(cross(ab, abc), ao) > real(0)) {        // outside edge ab
        gjk_set(s, a, b);
        return gjk_line(s, dir);
    }
    // Over the face. Search perpendicular to it, on the origin's side, keeping
    // the winding consistent so the tetrahedron case can trust it.
    if (dot(abc, ao) > real(0)) { dir =  abc; }
    else                        { gjk_set(s, a, c, b); dir = -abc; }
    return false;
}

// The only case that can succeed. The origin was already known to be on the
// inner side of the base face when this tetrahedron was formed, so three face
// tests decide it: outside any one of them, drop the vertex that face excludes
// and fall back to the triangle case.
static inline bool gjk_tetra(gjk_simplex& s, vec3& dir) {
    const vec3 a = s.p[0], b = s.p[1], c = s.p[2], d = s.p[3];
    const vec3 ao = -a;
    if (dot(cross(b - a, c - a), ao) > real(0)) { gjk_set(s, a, b, c); return gjk_triangle(s, dir); }
    if (dot(cross(c - a, d - a), ao) > real(0)) { gjk_set(s, a, c, d); return gjk_triangle(s, dir); }
    if (dot(cross(d - a, b - a), ao) > real(0)) { gjk_set(s, a, d, b); return gjk_triangle(s, dir); }
    return true;                                    // enclosed
}

static inline bool gjk_do_simplex(gjk_simplex& s, vec3& dir) {
    switch (s.n) {
        case 2:  return gjk_line(s, dir);
        case 3:  return gjk_triangle(s, dir);
        default: return gjk_tetra(s, dir);
    }
}

static inline void gjk_push(gjk_simplex& s, const vec3& p) {
    for (int i = 3; i > 0; i--) s.p[i] = s.p[i - 1];
    s.p[0] = p;
    if (s.n < 4) s.n++;
}

// Do A and B overlap? On true, `s` is a tetrahedron of Minkowski-difference
// points enclosing the origin, which is what EPA starts from.
//
// The loop is: aim at the origin, take the furthest point of the difference set
// in that direction, and ask whether it even reached the origin's side. If it
// did not, no point of the set can — `dir` is a separating direction and we are
// done in one test. If it did, keep it and let the case analysis discard
// whatever part of the simplex the origin has moved away from.
inline bool gjk_overlap(const phys_body& A, const phys_body& B, gjk_simplex& s) {
    vec3 dir = A.pos - B.pos;
    if (dir.length_squared() < real(1e-20)) dir = vec3(1, 0, 0);   // coincident centres

    gjk_set(s, support_diff(A, B, dir));
    dir = -s.p[0];

    for (int it = 0; it < GJK_MAX_ITERS; it++) {
        if (dir.length_squared() < real(1e-20)) return true;       // origin on the simplex
        const vec3 p = support_diff(A, B, dir);
        if (dot(p, dir) < real(0)) return false;                   // separating direction found
        // No progress: the extreme point in this direction is one we already
        // hold, so the simplex cannot get any closer to the origin. For a
        // polytope that means the origin is on or outside the boundary — a
        // grazing touch, which is not a contact worth resolving.
        for (int i = 0; i < s.n; i++)
            if ((p - s.p[i]).length_squared() < real(1e-20)) return false;
        gjk_push(s, p);
        if (gjk_do_simplex(s, dir)) return true;
    }
    return false;                                                  // no convergence
}

// ---- EPA -----------------------------------------------------------------

// These two are a MATCHED PAIR, measured together against the analytic
// sphere-box test over 1728 configurations (tests/test_physics.cu):
//
//   TOL     iters needed   worst depth error   worst normal error
//   1e-4    within 64      9.7e-5              0.0250 rad
//   1e-5    within 64      1.0e-5              0.0063 rad
//   1e-6    over 64        1.0e-6              0.0024 rad
//
// Depth error tracks TOL directly. The NORMAL error does not: a tolerance on
// DISTANCE bounds the angle only to sqrt(2*TOL/r) on a surface of radius r,
// because distance varies quadratically with angle near the closest point. That
// predicts 0.0063 at TOL = 1e-5 with r = 0.5, which is what the table measures,
// and identically in float and double — this is the algorithm's limit, not the
// precision's.
//
// 1e-5 is the tightest tolerance a curved shape still converges to within 64
// iterations; the fallback below never fires there (raising the cap to 512
// changes nothing). 1e-6 does need more, and capped at 64 it comes back WORSE
// (0.021 rad) than the looser setting, because the fallback returns the nearest
// face of an unfinished polytope. So the two must be raised together or not at
// all. This is also a concrete reason to keep the analytic sphere paths in
// physics.h beyond their speed: they are exact where this is 0.36 degrees off.
static constexpr int  EPA_MAX_ITERS = 64;
static constexpr real EPA_TOL       = real(1e-5);   // face-to-boundary distance

// A face of the expanding polytope, wound so `n` points AWAY from the origin.
// `dist` is the plane's distance from the origin, which is what "nearest face"
// is measured by.
struct epa_face {
    int  a, b, c;
    vec3 n;
    real dist;
};

struct epa_edge { int a, b; };

// Add a face, orienting it outward. Slivers with no usable normal are dropped.
//
// "Outward" is measured from `interior`, a point strictly inside the polytope,
// NOT from the origin. The origin is inside too and is the obvious candidate,
// but it can lie exactly ON a starting face's plane — GJK's tetrahedron tests
// accept the origin on a boundary plane as enclosed — and then it says nothing
// about which way that face points. Two boxes whose centres coincide EXACTLY hit
// this: the face got an arbitrary orientation and a distance of 0, which made it
// permanently the nearest face and let the convergence test pass on it
// immediately, reporting zero penetration between two boxes overlapping by 1.98.
// Offsetting the centres by 0.001 was enough to hide it. The tetrahedron's
// centroid cannot lie on its own face, and it stays interior as the polytope
// only ever grows, so it has no such blind spot.
static inline void epa_push_face(const std::vector<vec3>& v, std::vector<epa_face>& faces,
                                 const vec3& interior, int a, int b, int c) {
    vec3 n = cross(v[b] - v[a], v[c] - v[a]);
    const real len = n.length();
    if (len < real(1e-12)) return;
    n = n / len;
    if (dot(n, v[a] - interior) < real(0)) { n = -n; const int t = b; b = c; c = t; }
    real d = dot(n, v[a]);            // distance from the ORIGIN, which is what
    if (d < real(0)) d = real(0);     // "nearest face" ranks by; 0 when it is on the plane
    epa_face f; f.a = a; f.b = b; f.c = c; f.n = n; f.dist = d;
    faces.push_back(f);
}

// Collect the boundary of the visible region. An edge shared by two visible
// faces is interior and cancels; what survives is the horizon loop the new
// vertex cones back to.
static inline void epa_add_edge(std::vector<epa_edge>& h, int a, int b) {
    for (std::size_t i = 0; i < h.size(); i++)
        if (h[i].a == b && h[i].b == a) { h.erase(h.begin() + (std::ptrdiff_t)i); return; }
    epa_edge e; e.a = a; e.b = b; h.push_back(e);
}

// Penetration depth and normal, starting from GJK's enclosing tetrahedron.
//
// Repeatedly: take the face nearest the origin, ask the difference set for its
// furthest point in that face's direction, and stop once the face is already as
// far out as the set goes — that face lies ON the boundary, so its distance IS
// the penetration depth. Otherwise the new point proves the polytope is too
// small there: delete every face it can see, and cone it back to the horizon.
//
// `n` comes back in the CONTACT convention (from B toward A, the direction A
// must move to separate), which is the negative of the outward normal EPA finds:
// the outward normal points from the origin toward the boundary of A (-) B, and
// translating A by that much in that direction moves the set the same way, so it
// is A's exit direction that gets negated. Two overlapping spheres are the
// worked example in tests/test_physics.cu.
static inline bool epa_penetration(const phys_body& A, const phys_body& B,
                                   const gjk_simplex& s, vec3& n, real& pen) {
    if (s.n < 4) return false;

    std::vector<vec3> v;
    v.reserve(EPA_MAX_ITERS + 4);
    for (int i = 0; i < 4; i++) v.push_back(s.p[i]);

    // A flat starting tetrahedron has no interior for the polytope to grow from.
    // This is a grazing/degenerate configuration, not a penetration.
    if (std::fabs((double)dot(cross(v[1] - v[0], v[2] - v[0]), v[3] - v[0])) < 1e-12)
        return false;

    // A point that is strictly inside and stays inside, for orienting faces.
    const vec3 interior = (v[0] + v[1] + v[2] + v[3]) * real(0.25);

    std::vector<epa_face> faces;
    faces.reserve(4 * EPA_MAX_ITERS);
    epa_push_face(v, faces, interior, 0, 1, 2);
    epa_push_face(v, faces, interior, 0, 2, 3);
    epa_push_face(v, faces, interior, 0, 3, 1);
    epa_push_face(v, faces, interior, 1, 3, 2);
    if (faces.size() < 4) return false;

    std::vector<epa_edge> horizon;
    for (int it = 0; it < EPA_MAX_ITERS; it++) {
        std::size_t nearest = 0;
        for (std::size_t i = 1; i < faces.size(); i++)
            if (faces[i].dist < faces[nearest].dist) nearest = i;
        const vec3 fn = faces[nearest].n;
        const real fd = faces[nearest].dist;

        const vec3 p = support_diff(A, B, fn);
        if (dot(p, fn) - fd < EPA_TOL) {          // this face is already the boundary
            n   = -fn;
            pen = fd;
            return true;
        }

        horizon.clear();
        for (std::size_t i = faces.size(); i-- > 0; )
            if (dot(faces[i].n, p) > faces[i].dist) {          // p can see this face
                epa_add_edge(horizon, faces[i].a, faces[i].b);
                epa_add_edge(horizon, faces[i].b, faces[i].c);
                epa_add_edge(horizon, faces[i].c, faces[i].a);
                faces.erase(faces.begin() + (std::ptrdiff_t)i);
            }
        if (horizon.empty() || faces.empty()) return false;    // lost the polytope

        const int ip = (int)v.size();
        v.push_back(p);
        for (std::size_t i = 0; i < horizon.size(); i++)
            epa_push_face(v, faces, interior, horizon[i].a, horizon[i].b, ip);
        if (faces.empty()) return false;
    }

    // Out of iterations: report the best face we have. It is a lower bound on
    // the depth in very nearly the right direction, which the solver can act on;
    // returning nothing would drop a real contact.
    std::size_t nearest = 0;
    for (std::size_t i = 1; i < faces.size(); i++)
        if (faces[i].dist < faces[nearest].dist) nearest = i;
    n   = -faces[nearest].n;
    pen = faces[nearest].dist;
    return true;
}

// The convex-convex narrow phase, for any pair of colliders. Fills n (unit, from
// B toward A) and pen (>= 0) exactly as the analytic tests do, so contact_between
// can pick either path and the solver cannot tell which ran.
inline bool gjk_epa_contact(const phys_body& A, const phys_body& B, vec3& n, real& pen) {
    gjk_simplex s;
    if (!gjk_overlap(A, B, s)) return false;
    if (!epa_penetration(A, B, s, n, pen)) return false;
    return pen > real(0);
}

#endif // GJK_H
