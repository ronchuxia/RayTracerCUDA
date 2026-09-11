#!/usr/bin/env bash
# Build the interactive viewer.
#
#   SCENE=7 scripts/build_viewer.sh
#
#   SCENE   scene                                   binary
#   0       primitives                              build/viewer
#   1       ball pit, roomy 1.5, frictionless       build/viewer_pit
#   2       ball pit, tight 1.3, frictionless       build/viewer_tight
#   3       ball pit, roomy 1.5, friction 0.5       build/viewer_roll
#   4       spinning earth balls, friction 0.5      build/viewer_spin
#   5       room                                    build/viewer_room
#   6       dlss pit: mixed materials               build/viewer_dlss
#   7       hull pit: convex + balls                build/viewer_hull
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

# tee all stdout+stderr to a log
LOG="${LOG:-build/build_viewer.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

# detect GPU arch
if [ -z "${ARCH:-}" ]; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | tr -d '. ' || true)
    ARCH="sm_${CC:-86}"
fi
echo "building viewer with -arch=$ARCH"

# pick scene
SCENE="${SCENE:-0}"
case "$SCENE" in
    0) NAME=viewer ;;
    1) NAME=viewer_pit ;;
    2) NAME=viewer_tight ;;
    3) NAME=viewer_roll ;;
    4) NAME=viewer_spin ;;
    5) NAME=viewer_room ;;
    6) NAME=viewer_dlss ;;
    7) NAME=viewer_hull ;;
    *) echo "error: unknown SCENE=$SCENE (0 primitives, 1 pit, 2 tight, 3 roll, 4 spin, 5 room, 6 dlss pit, 7 hull pit)" >&2
       exit 1 ;;
esac

# pick precision
PRECISION="${PRECISION:-32}"
OUT="build/$NAME"
[ "$PRECISION" = 64 ] && OUT="${OUT}_fp64"

# OptiX denoiser
OPTIX_FLAGS=(-Isrc/external/optix -ldl)

# DLSS
DLSS_FLAGS=(-Isrc/external/dlss/include src/external/dlss/lib/libnvsdk_ngx.a -lcuda)

# Dear ImGui
IMGUI=src/external/imgui
IMGUI_FLAGS=(-I"$IMGUI" "$IMGUI"/imgui.cpp "$IMGUI"/imgui_draw.cpp "$IMGUI"/imgui_tables.cpp
             "$IMGUI"/imgui_widgets.cpp "$IMGUI"/imgui_impl_sdl2.cpp "$IMGUI"/imgui_impl_vulkan.cpp)

# SDL2
SDL_CFLAGS=$(pkg-config --cflags sdl2)

nvcc src/viewer/viewer.cu -o "$OUT" -std=c++14 -arch="$ARCH" -Isrc $SDL_CFLAGS \
    -DRT_PRECISION="$PRECISION" -DVIEWER_SCENE="$SCENE" \
    -lSDL2 -lvulkan -lnvidia-ml "${IMGUI_FLAGS[@]}" "${OPTIX_FLAGS[@]}" "${DLSS_FLAGS[@]}" "$@"

echo "built $OUT (SCENE=$SCENE, RT_PRECISION=$PRECISION, log: $LOG)"
