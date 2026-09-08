#ifndef GJK_H
#define GJK_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "physics/body.h"
#include "physics/math.h"   // any_perpendicular
#include "precision.h"      // real
#include "vec3.h"

// ---- support functions ---------------------------------------------------

inline vec3 support(const phys_body& b, const vec3& dir) {
    if (b.shape == COLLIDER_SPHERE) {
        real len2 = dir.length_squared();
        if (len2 < real(1e-20)) return b.pos;
        return b.pos + dir * (b.radius / std::sqrt(len2));
    }
    if (b.shape == COLLIDER_HULL) {
        const vec3 d(dot(dir, b.axes[0]), dot(dir, b.axes[1]), dot(dir, b.axes[2]));
        const vec3* best = &b.hull.verts[0];
        for (const vec3& v : b.hull.verts)
            if (dot(v, d) > dot(*best, d))
                best = &v;
        return b.pos + b.axes[0] * (*best)[0] + b.axes[1] * (*best)[1] + b.axes[2] * (*best)[2];
    }
    // COLLIDER_BOX
    vec3 p = b.pos;
    for (int i = 0; i < 3; i++)
        p += b.axes[i] * (dot(dir, b.axes[i]) >= real(0) ? b.half[i] : -b.half[i]);
    return p;
}

// support function of the Minkowski difference A (-) B
inline vec3 support_diff(const phys_body& A, const phys_body& B, const vec3& dir) {
    return support(A, dir) - support(B, -dir);
}

// ---- GJK -----------------------------------------------------------------

static constexpr int GJK_MAX_ITERS = 32;

// up to four points of the Minkowski difference
// p[0] is always the NEWEST point
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

static inline void gjk_push(gjk_simplex& s, const vec3& p) {
    for (int i = 3; i > 0; i--) s.p[i] = s.p[i - 1];
    s.p[0] = p;
    if (s.n < 4) s.n++;
}

static inline vec3 triple(const vec3& a, const vec3& b) {
    return cross(cross(a, b), a);
}

static inline bool gjk_do_line(gjk_simplex& s, vec3& dir) {
    const vec3 a = s.p[0], b = s.p[1];      // a is newest
    const vec3 ab = b - a, ao = -a;
    if (dot(ab, ao) > real(0)) {            // some point on ab is closer to origin than a
        dir = triple(ab, ao);               // dir is perpendicular to ab and points from ab toward the origin
        if (dir.length_squared() < real(1e-20))   // origin on ab, does not immediately report overlap because epa needs a tetrahedron
            dir = any_perpendicular(ab / ab.length());
    } else {                                // no point on ab is closer to origin than a
        gjk_set(s, a);
        dir = ao;
    }
    return false;
}

static inline bool gjk_do_triangle(gjk_simplex& s, vec3& dir) {
    const vec3 a = s.p[0], b = s.p[1], c = s.p[2];
    const vec3 ab = b - a, ac = c - a, ao = -a;
    const vec3 abc = cross(ab, ac);         // the triangle's plane normal

    if (dot(cross(abc, ac), ao) > real(0)) {        // projected origin outside edge ac
        gjk_set(s, a, c);
        dir = triple(ac, ao);
        return false;
    }
    if (dot(cross(ab, abc), ao) > real(0)) {        // projected origin outside edge ab
        gjk_set(s, a, b);
        dir = triple(ab, ao);
        return false;
    }
    // projected origin inside triangle abc
    if (dot(abc, ao) > real(0)) { dir =  abc; }
    else                        { gjk_set(s, a, c, b); dir = -abc; }
    return false;
}

static inline bool gjk_do_tetra(gjk_simplex& s, vec3& dir) {
    const vec3 a = s.p[0], b = s.p[1], c = s.p[2], d = s.p[3];
    const vec3 ao = -a;
    if (dot(cross(b - a, c - a), ao) > real(0)) { gjk_set(s, a, b, c); return gjk_do_triangle(s, dir); }
    if (dot(cross(c - a, d - a), ao) > real(0)) { gjk_set(s, a, c, d); return gjk_do_triangle(s, dir); }
    if (dot(cross(d - a, b - a), ao) > real(0)) { gjk_set(s, a, d, b); return gjk_do_triangle(s, dir); }
    return true;
}

// prune simplex an re-aim dir
static inline bool gjk_do_simplex(gjk_simplex& s, vec3& dir) {
    switch (s.n) {
        case 2:  return gjk_do_line(s, dir);
        case 3:  return gjk_do_triangle(s, dir);
        default: return gjk_do_tetra(s, dir);
    }
}

inline bool gjk_overlap(const phys_body& A, const phys_body& B, gjk_simplex& s) {
    // seed the simplex
    vec3 dir = A.pos - B.pos;
    if (dir.length_squared() < real(1e-20)) dir = vec3(1, 0, 0);
    gjk_set(s, support_diff(A, B, dir));
    dir = -s.p[0];

    for (int it = 0; it < GJK_MAX_ITERS; it++) {
        if (dir.length_squared() < real(1e-20)) return true;       // origin on the simplex
        const vec3 p = support_diff(A, B, dir);
        if (dot(p, dir) < real(0)) return false;                   // separating direction found
        for (int i = 0; i < s.n; i++)
            if ((p - s.p[i]).length_squared() < real(1e-20)) return false;  // repeated support point
        gjk_push(s, p);
        if (gjk_do_simplex(s, dir)) return true;
    }
    return false;
}

// ---- EPA -----------------------------------------------------------------

static constexpr int  EPA_MAX_ITERS = 64;
static constexpr real EPA_TOL       = real(1e-5);   // face-to-boundary distance

struct epa_face {
    int  a, b, c;   // counterclockwise when viewed from outside
    vec3 n;         // outward normal
    real dist;      // distance from origin
};

struct epa_edge { int a, b; };

static inline void epa_push_face_to_polytope(const std::vector<vec3>& v, std::vector<epa_face>& faces,
                                             const vec3& interior, int a, int b, int c) {
    vec3 n = cross(v[b] - v[a], v[c] - v[a]);
    const real len = n.length();
    if (len < real(1e-12)) return;
    n = n / len;
    if (dot(n, v[a] - interior) < real(0)) { n = -n; const int t = b; b = c; c = t; }
    real d = dot(n, v[a]);
    if (d < real(0)) d = real(0);
    epa_face f; f.a = a; f.b = b; f.c = c; f.n = n; f.dist = d;
    faces.push_back(f);
}

static inline void epa_add_edge_to_horizon(std::vector<epa_edge>& horizon, int a, int b) {
    for (std::size_t i = 0; i < horizon.size(); i++)
        if (horizon[i].a == b && horizon[i].b == a) {   // edge lies inside the region of faces to be removed
            horizon.erase(horizon.begin() + (std::ptrdiff_t)i);
            return;
        }
    epa_edge e; e.a = a; e.b = b; horizon.push_back(e);
}

static inline bool epa_penetration(const phys_body& A, const phys_body& B,
                                   const gjk_simplex& s, vec3& n, real& pen) {
    if (s.n < 4) return false;

    std::vector<vec3> v;
    v.reserve(EPA_MAX_ITERS + 4);
    for (int i = 0; i < 4; i++) v.push_back(s.p[i]);

    // tetrahedron has nonzero volume
    if (std::fabs((double)dot(cross(v[1] - v[0], v[2] - v[0]), v[3] - v[0])) < 1e-12)
        return false;

    // a point strictly inside tetrahedron
    const vec3 interior = (v[0] + v[1] + v[2] + v[3]) * real(0.25);

    // initialize polytope with tetrahedron
    std::vector<epa_face> faces;
    faces.reserve(4 * EPA_MAX_ITERS);
    epa_push_face_to_polytope(v, faces, interior, 0, 1, 2);
    epa_push_face_to_polytope(v, faces, interior, 0, 2, 3);
    epa_push_face_to_polytope(v, faces, interior, 0, 3, 1);
    epa_push_face_to_polytope(v, faces, interior, 1, 3, 2);
    if (faces.size() < 4) return false;

    std::vector<epa_edge> horizon;  // boundary of region of faces to be removed
    for (int it = 0; it < EPA_MAX_ITERS; it++) {
        // find the closest face to the origin
        std::size_t nearest = 0;
        for (std::size_t i = 1; i < faces.size(); i++)
            if (faces[i].dist < faces[nearest].dist) nearest = i;
        const vec3 fn = faces[nearest].n;
        const real fd = faces[nearest].dist;

        const vec3 p = support_diff(A, B, fn);  // the support point of the Minkowski difference in the closest face's outward normal's direction
        if (dot(p, fn) - fd < EPA_TOL) {        // the closest face is already the boundary
            n   = -fn;
            pen = fd;
            return true;
        }

        // add p to polytope
        horizon.clear();
        for (std::size_t i = faces.size(); i-- > 0; )
            if (dot(faces[i].n, p) > faces[i].dist) {           // p lies outside the face
                epa_add_edge_to_horizon(horizon, faces[i].a, faces[i].b);
                epa_add_edge_to_horizon(horizon, faces[i].b, faces[i].c);
                epa_add_edge_to_horizon(horizon, faces[i].c, faces[i].a);
                faces.erase(faces.begin() + (std::ptrdiff_t)i); // remove the face
            }
        if (horizon.empty() || faces.empty()) return false;    // lost the polytope

        const int ip = (int)v.size();
        v.push_back(p);
        for (std::size_t i = 0; i < horizon.size(); i++)
            epa_push_face_to_polytope(v, faces, interior, horizon[i].a, horizon[i].b, ip);  // add new faces
    }

    // out of iterations: report the best face
    std::size_t nearest = 0;
    for (std::size_t i = 1; i < faces.size(); i++)
        if (faces[i].dist < faces[nearest].dist) nearest = i;
    n   = -faces[nearest].n;
    pen = faces[nearest].dist;
    return true;
}

inline bool gjk_epa_contact(const phys_body& A, const phys_body& B, vec3& n, real& pen) {
    gjk_simplex s;
    if (!gjk_overlap(A, B, s)) return false;
    if (!epa_penetration(A, B, s, n, pen)) return false;
    return pen > real(0);
}

#endif // GJK_H
