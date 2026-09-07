#ifndef VIEWER_DENOISER_H
#define VIEWER_DENOISER_H

#include "viewer/gbuffer.h"
#include "viewer/flow.h"
#include "viewer/primary_hits.h"

struct denoiser {
    float3* output = nullptr;
    int out_w = 0, out_h = 0;

    virtual void setup(int in_w, int in_h, int out_w, int out_h) = 0;   // setup or re-setup the buffers after frame size changes, also resets history
    virtual void invoke(const color* accum, const gbuffer& gb, int samples, float blend, const flow_field& ff,
                        const primary_hits& ph, const camera& cam) = 0;
    virtual void reset_history() {}
    virtual ~denoiser() {}
};

#endif // VIEWER_DENOISER_H
