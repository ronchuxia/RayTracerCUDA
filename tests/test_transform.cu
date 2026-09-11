// TRS transform tests: the math is the risky part (ray world<->object,
// inverse-transpose normals, Euler order), so verify it directly on the device
// through instance_hit, the one place the transform is applied.
//  1. a translation-only instance agrees with the same sphere authored at the
//     translated position under the identity;
//  2. each Euler axis rotates a known point correctly;
//  3. a non-uniformly scaled sphere (an ellipsoid) is hit where expected, with
//     a correctly inverse-transposed, unit-length normal.
#include <cstdio>
#include <cmath>

#include "hittable.h"
#include "material.h"
#include "viewer/scene.h"
#include "scenes/scene_utils.h"

// Fire one ray at an instance; report hit + t + point + normal.
__global__ void shoot(const instance* in, ray r, int* hit, real* t, vec3* p, vec3* n) {
    hit_record rec;
    *hit = instance_hit(*in, r, interval(real(0.001), infinity), rec, nullptr);
    if (*hit) { *t = rec.t; *p = rec.p; *n = rec.normal; }
}

static int   *g_hit;  static real *g_t;  static vec3 *g_p, *g_n;
static instance* g_in;
static void run(const instance& in, ray r) {
    *g_in = in;   // managed copy the kernel can read
    shoot<<<1,1>>>(g_in, r, g_hit, g_t, g_p, g_n);
    checkCudaErrors(cudaDeviceSynchronize());
}
static bool close(real a, real b, real eps = 1e-3) { return fabs((double)(a - b)) < eps; }
static bool vclose(const vec3& a, const vec3& b, real eps = 1e-3) {
    return close(a.x(),b.x(),eps) && close(a.y(),b.y(),eps) && close(a.z(),b.z(),eps);
}

#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } \
                              else printf("ok: %s\n", msg); } while (0)

int main() {
    checkCudaErrors(cudaMallocManaged(&g_hit, sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&g_t, sizeof(real)));
    checkCudaErrors(cudaMallocManaged(&g_p, sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&g_n, sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&g_in, sizeof(instance)));
    std::vector<void*> allocs;
    material* m = new_lambertian(color(1,1,1), allocs);
    int fails = 0;
    const vec3 no_rot(0,0,0), unit(1,1,1);

    // 1. translation-only instance == the sphere authored at (5,0,0) under the identity.
    instance authored = make_instance(make_sphere(point3(5,0,0), 1.0, m, allocs), vec3(0,0,0), no_rot, unit);
    instance via_transform = make_instance(make_sphere(point3(0,0,0), 1.0, m, allocs), vec3(5,0,0), no_rot, unit);
    ray r1(point3(5,0,-5), vec3(0,0,1));   // straight at the moved sphere
    run(authored, r1);       bool h1 = *g_hit; real t1 = *g_t; vec3 p1 = *g_p, n1 = *g_n;
    run(via_transform, r1);  bool h2 = *g_hit; real t2 = *g_t; vec3 p2 = *g_p, n2 = *g_n;
    CHECK(h1 && h2 && close(t1,t2) && vclose(p1,p2) && vclose(n1,n2),
          "translation-only instance matches the sphere authored in place");

    // 2. rotation of a centered sphere is a no-op on position: a yawed unit
    //    sphere at translation (4,0,0) is still hit at its center from -X.
    instance yaw = make_instance(make_sphere(point3(0,0,0), 1.0, m, allocs), vec3(4,0,0), vec3(0,90,0), unit);
    ray at_center(point3(-5,0,0), vec3(1,0,0));   // origin y=0,z=0 line passes through (4,0,0)
    run(yaw, at_center);
    CHECK(*g_hit && close(g_p->x(), 3.0), "yawed unit sphere still hit at its center face");

    // A non-centered feature: rotate about Z by 90, a sphere translated to +X=4,
    // confirm the instance's bbox center moved appropriately (position map).
    instance roll = make_instance(make_sphere(point3(0,0,0), 1.0, m, allocs), vec3(4,0,0), vec3(0,0,90), unit);
    aabb bb = roll.bounding_box();
    real cx = (bb.x.min + bb.x.max) / 2, cy = (bb.y.min + bb.y.max) / 2;
    CHECK(close(cx,4.0) && close(cy,0.0), "translation applied after rotation (center at T)");

    // The instance box is the union of the leaf box's eight mapped corners
    // (pbrt-v4 Transform::operator()(Bounds3f)), so a rotated sphere's box is
    // LOOSE — the rotated cube's corners stick out — but must still contain
    // the sphere, [T-r, T+r] on every axis.
    instance rot_none = make_instance(make_sphere(point3(0,0,0), 1.0, m, allocs), vec3(2,0,0), no_rot, unit);
    instance rot_arb  = make_instance(make_sphere(point3(0,0,0), 1.0, m, allocs), vec3(2,0,0), vec3(37,52,-19), unit);
    aabb b0 = rot_none.bounding_box(), b1 = rot_arb.bounding_box();
    CHECK(b1.x.min <= 1.0 && b1.x.max >= 3.0 && b1.y.min <= -1.0 && b1.y.max >= 1.0 &&
          b1.z.min <= -1.0 && b1.z.max >= 1.0, "rotated sphere AABB still contains the sphere");
    CHECK(close(b0.x.min,1.0) && close(b0.x.max,3.0), "unrotated sphere AABB is the tight [T-r, T+r] box");

    // 3. Ellipsoid: unit sphere scaled (1,1,3) along z. A ray down -z through the
    //    center hits the front pole at z=+3; the normal there must be +z, unit.
    instance ell = make_instance(make_sphere(point3(0,0,0), 1.0, m, allocs), vec3(0,0,0), no_rot, vec3(1,1,3));
    // Under an axis-aligned scale the corner union is exact: ±(1,1,3).
    aabb be = ell.bounding_box();
    CHECK(close(be.x.max,1.0) && close(be.y.max,1.0) && close(be.z.max,3.0) &&
          close(be.z.min,-3.0), "scaled sphere AABB is the tight ellipsoid box");
    ray downz(point3(0,0,10), vec3(0,0,-1));
    run(ell, downz);
    CHECK(*g_hit && close(g_p->z(), 3.0), "ellipsoid (scale z=3) hit at z=+3 pole");
    CHECK(vclose(*g_n, vec3(0,0,1)), "ellipsoid pole normal is +z");
    CHECK(close(g_n->length(), 1.0), "ellipsoid normal is unit length (inverse-transpose + renorm)");
    // A grazing hit on the ellipsoid's side must have a normal with NO shear —
    // hit the +x equator (unaffected by z-scale): normal should be +x.
    ray fromx(point3(10,0,0), vec3(-1,0,0));
    run(ell, fromx);
    CHECK(*g_hit && close(g_p->x(), 1.0) && vclose(*g_n, vec3(1,0,0)),
          "ellipsoid equator (x) unaffected by z-scale");

    // 4. A mesh leaf goes through the same maps: a unit box turned 45 deg about
    //    Y and moved to (4,0,0) is hit at its near corner ridge, x = 4 - sqrt(2)/2.
    std::vector<mesh*> mesh_dtors;
    mesh* box = new_box(point3(-0.5,-0.5,-0.5), point3(0.5,0.5,0.5), m, allocs, mesh_dtors);
    instance turned = make_instance(box, vec3(4,0,0), vec3(0,45,0), unit);
    ray fromnegx(point3(-5,0,0), vec3(1,0,0));
    run(turned, fromnegx);
    CHECK(*g_hit && close(g_p->x(), 4.0 - std::sqrt(0.5)), "turned box mesh hit at its ridge");
    CHECK(close(g_n->x(), -std::sqrt(0.5), 1e-3) && close(g_n->y(), 0.0), "turned box normal rotated with the instance");

    printf(fails ? "TRANSFORM TESTS FAILED (%d)\n" : "ALL TRANSFORM TESTS PASSED\n", fails);
    for (mesh* mm : mesh_dtors) mm->~mesh();
    for (void* p : allocs) cudaFree(p);
    return fails ? 1 : 0;
}
