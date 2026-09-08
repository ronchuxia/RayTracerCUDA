#ifndef PHYSICS_HULL_H
#define PHYSICS_HULL_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "precision.h"
#include "vec3.h"

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

struct tri {
    int  v[3];
    vec3 n;
    real d;
    bool alive;
};

inline tri make_tri(const std::vector<vec3>& p, int a, int b, int c) {
    tri t;
    t.v[0] = a;
    t.v[1] = b; 
    t.v[2] = c;
    t.n = unit_vector(cross(p[b] - p[a], p[c] - p[a]));
    t.d = dot(t.n, p[a]);
    t.alive = true;
    return t;
}

// returns the centre of mass
inline vec3 build_hull(const std::vector<vec3>& p, hull_shape& h) {
    const int n = (int)p.size();
    vec3 lo = p[0], hi = p[0];
    for (const vec3& q : p) 
        for (int a = 0; a < 3; a++) { 
            lo[a] = fmin(lo[a], q[a]); 
            hi[a] = fmax(hi[a], q[a]); 
        }
    const real tol = real(1e-5) * (hi - lo).length();   // a point whose distance to a plane is within tol lies on that plane

    // initial tetrahedron
    int e[4] = { 0, 0, 0, 0 };
    // 1st point: the point with smallest x coordinate
    for (int i = 1; i < n; i++) if (p[i][0] < p[e[0]][0]) e[0] = i;
    // 2nd point: the point farthest from the first point
    real best = real(-1);
    for (int i = 0; i < n; i++) { const real l = (p[i] - p[e[0]]).length_squared(); if (l > best) { best = l; e[1] = i; } }
    // 3rd point: the point farthest from the line through the first two points
    const vec3 axis = p[e[1]] - p[e[0]];
    best = real(-1);
    for (int i = 0; i < n; i++) { const real l = cross(axis, p[i] - p[e[0]]).length_squared(); if (l > best) { best = l; e[2] = i; } }
    // 4th point: the point farthest from the plane through the first three points
    const vec3 nrm = unit_vector(cross(axis, p[e[2]] - p[e[0]]));
    best = real(-1);
    for (int i = 0; i < n; i++) { const real l = fabs(dot(nrm, p[i] - p[e[0]])); if (l > best) { best = l; e[3] = i; } }
    if (best <= tol) { 
        fprintf(stderr, "build_hull: the points are flat\n"); 
        exit(1); 
    }

    std::vector<tri> F;
    const int tet[4][4] = { { 0, 1, 2, 3 }, { 0, 1, 3, 2 }, { 0, 2, 3, 1 }, { 1, 2, 3, 0 } };
    for (int f = 0; f < 4; f++) {
        tri t = make_tri(p, e[tet[f][0]], e[tet[f][1]], e[tet[f][2]]);
        if (dot(t.n, p[e[tet[f][3]]]) > t.d) 
            t = make_tri(p, e[tet[f][0]], e[tet[f][2]], e[tet[f][1]]);
        F.push_back(t);
    }

    // insert every point
    for (int i = 0; i < n; i++) {
        std::vector<int> vis;
        for (int f = 0; f < (int)F.size(); f++)
            if (F[f].alive && dot(F[f].n, p[i]) - F[f].d > tol) 
                vis.push_back(f);
        if (vis.empty()) continue;
        
        std::set<std::pair<int, int> > edges;
        for (int f : vis) 
            for (int k = 0; k < 3; k++) 
                edges.insert(std::make_pair(F[f].v[k], F[f].v[(k + 1) % 3]));

        for (int f : vis) {
            const tri t = F[f];
            F[f].alive = false;
            for (int k = 0; k < 3; k++) {
                const int a = t.v[k], b = t.v[(k + 1) % 3];
                if (!edges.count(std::make_pair(b, a))) 
                    F.push_back(make_tri(p, a, b, i));
            }
        }
    }

    // merge coplanar triangles
    std::map<std::pair<int, int>, int> face_of_edge;
    for (int f = 0; f < (int)F.size(); f++)
        if (F[f].alive) 
            for (int k = 0; k < 3; k++) 
                face_of_edge[std::make_pair(F[f].v[k], F[f].v[(k + 1) % 3])] = f;
    
    std::vector<int> group(F.size(), -1);
    std::vector<int> remap(n, -1);
    h = hull_shape();
    for (int seed = 0; seed < (int)F.size(); seed++) {
        if (!F[seed].alive || group[seed] >= 0) continue;

        const int g = (int)h.faces.size();
        std::vector<int> members(1, seed);
        group[seed] = g;
        for (size_t m = 0; m < members.size(); m++)
            for (int k = 0; k < 3; k++) {
                const tri& t = F[members[m]];
                const int nb = face_of_edge[std::make_pair(t.v[(k + 1) % 3], t.v[k])];
                if (group[nb] >= 0) continue;

                bool on = true;
                for (int j = 0; j < 3; j++) 
                    on = on && fabs(dot(F[seed].n, p[F[nb].v[j]]) - F[seed].d) <= tol;
                if (on) { 
                    group[nb] = g; 
                    members.push_back(nb); 
                }
            }
        
        // find the group's boundary edges
        std::map<int, int> next;
        for (int m : members)
            for (int k = 0; k < 3; k++) {
                const int a = F[m].v[k], b = F[m].v[(k + 1) % 3];
                if (group[face_of_edge[std::make_pair(b, a)]] != g) next[a] = b;
            }
        
        // chain the boundary edges into a loop
        std::vector<int> loop;
        for (int v = next.begin()->first; loop.empty() || v != loop[0]; v = next[v]) loop.push_back(v);

        // remove collinear vertices
        std::vector<int> kept;
        for (size_t k = 0; k < loop.size(); k++) {
            const vec3& prev = p[kept.empty() ? loop.back() : kept.back()];
            const vec3& cur  = p[loop[k]];
            const vec3& nxt  = p[loop[(k + 1) % loop.size()]];
            if (cross(cur - prev, nxt - prev).length() > tol * (nxt - prev).length()) kept.push_back(loop[k]);
        }
        if ((int)kept.size() > HULL_MAX_LOOP) { fprintf(stderr, "build_hull: a face has more than %d vertices\n", (int)kept.size(), HULL_MAX_LOOP); exit(1); }
        
        hull_shape::face fc;
        fc.normal = F[seed].n;
        for (int v : kept) {
            if (remap[v] < 0) { // the vertex has not been added to the hull
                remap[v] = (int)h.verts.size(); 
                h.verts.push_back(p[v]); 
            }
            fc.loop.push_back(remap[v]);
        }
        h.faces.push_back(fc);
    }

    return hull_mass_properties(h);
}

#endif // PHYSICS_HULL_H
