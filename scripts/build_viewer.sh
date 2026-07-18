#!/usr/bin/env bash
# Build the interactive viewer (roadmap B1) → build/viewer.
# Needs SDL2 + GLEW + OpenGL dev libraries; no display required to build.
# Full output (incl. nvcc/ptxas warnings) is teed to build/build_viewer.log
# and still shown on the terminal; override the path with LOG=...
#   scripts/build_viewer.sh                                       # auto-detect GPU arch
#   ARCH=sm_89 scripts/build_viewer.sh                            # override the arch
#   LOG=build/my.log scripts/build_viewer.sh                      # custom log path
#   scripts/build_viewer.sh -DRT_IMAGE_WIDTH=1280 -DRT_SAMPLES=8  # extra nvcc -D flags
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

# Tee all stdout+stderr to a log under build/ (and keep showing it on the terminal).
LOG="${LOG:-build/build_viewer.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

if [ -z "${ARCH:-}" ]; then
    # Query GPU 0 explicitly (one line, so no `head` — avoids a SIGPIPE race
    # under pipefail); `|| true` tolerates nvidia-smi being absent.
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | tr -d '. ' || true)
    ARCH="sm_${CC:-86}"
fi
echo "building viewer with -arch=$ARCH"

# PRECISION=64 builds the double-precision reference viewer as build/viewer_fp64
# (run it side by side with the default float build/viewer to compare trace ms).
PRECISION="${PRECISION:-32}"
OUT=build/viewer
[ "$PRECISION" = 64 ] && OUT=build/viewer_fp64

# Dear ImGui (vendored in src/external/imgui, pinned v1.92.8) is plain C++ —
# nvcc hands the .cpp files to the host compiler. Its SDL2 backend does
# `#include <SDL.h>`, so it needs SDL2's include dir from pkg-config.
SDL_CFLAGS=$(pkg-config --cflags sdl2)
IMGUI=src/external/imgui
nvcc src/viewer/viewer.cu \
    "$IMGUI"/imgui.cpp "$IMGUI"/imgui_draw.cpp "$IMGUI"/imgui_tables.cpp \
    "$IMGUI"/imgui_widgets.cpp "$IMGUI"/imgui_impl_sdl2.cpp "$IMGUI"/imgui_impl_opengl2.cpp \
    -o "$OUT" -std=c++14 -arch="$ARCH" -Isrc -I"$IMGUI" $SDL_CFLAGS \
    -DRT_PRECISION="$PRECISION" \
    -lSDL2 -lGLEW -lGL "$@"
echo "built $OUT (RT_PRECISION=$PRECISION, log: $LOG)"
