# RayTracerCUDA

A CUDA path tracer with an interactive SDL2/Vulkan viewer and rigid-body
physics.

## Features

- CUDA path tracing.
    - Meshes.
    - BVH.
- Interactive SDL2/Vulkan viewer.
- OptiX denoiser.
- DLSS ray reconstruction
- DLSS frame generation.
- Rigid-body physics.
    - Spheres.
    - Boxes.
    - Convex hulls.

## Build the Viewer

The viewer requires an NVIDIA CUDA toolkit, an NVIDIA driver, SDL2 development
files, Vulkan development files, `pkg-config`, and NVML.

Build the scene from the repository root. Select a scene with `SCENE=1` through `SCENE=7`:

```bash
SCENE=7 scripts/build_viewer.sh
```

## Run the Viewer

Use the same `SCENE` value used to build the viewer:

```bash
SCENE=7 scripts/run_viewer.sh
```

## UI

- **Switch view:** use the **view** dropdown.
- **Switch denoiser:** use the **denoiser** dropdown.
- **Turn on DLSS frame generation:** click the **frame generation** checkbox.
- **Play physics:** click the **Play**, **Pause**, or **Stop** button.
