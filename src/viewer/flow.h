#ifndef VIEWER_FLOW_H
#define VIEWER_FLOW_H

#include "camera.h"
#include "cuda_helper.h"

struct flow_field {
    float2* flow = nullptr;
    float*  trust = nullptr;       // 1 = history valid; 0 = off-frame, occluded, or no history yet
    real*   depth = nullptr;
    real*   depth_prev = nullptr;
    int*    id = nullptr;
    size_t  pixels = 0;
    bool    history = false;

    void allocate(int w, int h) {
        release();
        pixels = (size_t)w * h;
        checkCudaErrors(cudaMalloc(&flow,       pixels * sizeof(float2)));
        checkCudaErrors(cudaMalloc(&trust,      pixels * sizeof(float)));
        checkCudaErrors(cudaMalloc(&depth,      pixels * sizeof(real)));
        checkCudaErrors(cudaMalloc(&depth_prev, pixels * sizeof(real)));
        checkCudaErrors(cudaMalloc(&id,         pixels * sizeof(int)));
        history = false;
    }
    void release() {
        cudaFree(flow); cudaFree(trust); cudaFree(depth); cudaFree(depth_prev); cudaFree(id);
        flow = nullptr; trust = nullptr; depth = depth_prev = nullptr; id = nullptr; pixels = 0;
    }
    void reset_history() { history = false; }
    void advance() { real* t = depth; depth = depth_prev; depth_prev = t; history = true; }
};

__global__ void flow_frame(const camera& cam, const camera& prev_cam, const hittable& world,
                           const transform* const* tr, const transform* tr_prev,
                           curandState* rand_states, flow_field ff, int w, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;
    int p = j * w + i;

    ray r = cam.get_ray_through_pixel(i, j);
    hit_record rec;
    bool hit = world.hit(r, interval(real(0.001), infinity), rec, &rand_states[p]);

    point3 pt, pt_prev;
    if (hit) {
        pt = rec.p;
        ff.depth[p] = (pt - cam.center).length();
        ff.id[p]    = rec.id;
        const transform& t  = *tr[rec.id];
        const transform& tp = tr_prev[rec.id];
        vec3 q  = t.inv_scale * t.apply_Rt(pt - t.translation);             // transform hit point from world space to object space
        pt_prev = tp.apply_R(q * tp.scale) + tp.translation;                // transform hit point from object space to world space in previous frame
    } else {
        pt = cam.center + unit_vector(r.direction()) * real(1e4);
        ff.depth[p] = infinity;
        ff.id[p]    = -1;
        pt_prev = pt;
    }

    // transform hit point from world space to pixel space in previous frame
    real qx, qy;
    bool ahead = prev_cam.world_to_pixel(pt_prev, qx, qy);
    int qi = (int)(qx + real(0.5)), qj = (int)(qy + real(0.5));
    bool vis = ahead && qi >= 0 && qi < w && qj >= 0 && qj < h;
    ff.flow[p] = vis ? make_float2((float)(real(i) - qx), (float)(real(j) - qy)) : make_float2(0.f, 0.f);

    bool ok = vis && ff.history;
    bool occluded = false;
    if (ok) {
        real d_prev = ff.depth_prev[qj * w + qi];
        real d_pt   = hit ? (pt_prev - prev_cam.center).length() : infinity;
        occluded = d_prev < d_pt * real(0.99);
    }
    ff.trust[p] = ok && !occluded ? 1.f : 0.f;
}

#endif // VIEWER_FLOW_H
