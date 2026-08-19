#!/usr/bin/env bash
# Build the interactive viewer (roadmap B1).
#
# EACH SCENE GETS ITS OWN BINARY, so they coexist: building one never clobbers
# another, and you can keep several open side by side. Pick with SCENE=, which
# sets both -DVIEWER_SCENE and the output name.
#
#   SCENE   scene                                   binary
#   0       primitives showcase (default)           build/viewer
#   1       ball pit, roomy 1.5, frictionless       build/viewer_pit
#   2       ball pit, tight 1.3, frictionless       build/viewer_tight
#   3       ball pit, roomy 1.5, friction 0.5       build/viewer_rolling
#
# Needs SDL2 + GLEW + OpenGL dev libraries; no display required to build.
# Full output (incl. nvcc/ptxas warnings) is teed to build/build_viewer.log
# and still shown on the terminal; override the path with LOG=...
#   scripts/build_viewer.sh                                       # scene 0, auto-detect GPU arch
#   SCENE=3 scripts/build_viewer.sh                               # -> build/viewer_rolling
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

# SCENE picks both the compiled-in scene and the binary name, so the two can
# never disagree — a binary called viewer_rolling always IS the rolling scene.
# Passing -DVIEWER_SCENE by hand would break that, hence the guard.
SCENE="${SCENE:-0}"
case " $* " in
    *" -DVIEWER_SCENE"*)
        echo "error: pass the scene as SCENE=$SCENE, not -DVIEWER_SCENE — the name" >&2
        echo "       of the output binary is derived from it." >&2
        exit 1 ;;
esac
case "$SCENE" in
    0) NAME=viewer ;;
    1) NAME=viewer_pit ;;
    2) NAME=viewer_tight ;;
    3) NAME=viewer_rolling ;;
    *) echo "error: unknown SCENE=$SCENE (0 primitives, 1 pit, 2 tight, 3 rolling)" >&2
       exit 1 ;;
esac

# PRECISION=64 builds the double-precision reference viewer, suffixed _fp64
# (run it side by side with the float build to compare trace ms).
PRECISION="${PRECISION:-32}"
OUT="build/$NAME"
[ "$PRECISION" = 64 ] && OUT="${OUT}_fp64"

# Dear ImGui (vendored in src/external/imgui, pinned v1.92.8) is plain C++ —
# nvcc hands the .cpp files to the host compiler. Its SDL2 backend does
# `#include <SDL.h>`, so it needs SDL2's include dir from pkg-config.
SDL_CFLAGS=$(pkg-config --cflags sdl2)
IMGUI=src/external/imgui
nvcc src/viewer/viewer.cu \
    "$IMGUI"/imgui.cpp "$IMGUI"/imgui_draw.cpp "$IMGUI"/imgui_tables.cpp \
    "$IMGUI"/imgui_widgets.cpp "$IMGUI"/imgui_impl_sdl2.cpp "$IMGUI"/imgui_impl_opengl2.cpp \
    -o "$OUT" -std=c++14 -arch="$ARCH" -rdc=true -Isrc -I"$IMGUI" $SDL_CFLAGS \
    -DRT_PRECISION="$PRECISION" -DVIEWER_SCENE="$SCENE" \
    -lSDL2 -lGLEW -lGL -lnvidia-ml "$@"
echo "built $OUT (SCENE=$SCENE, RT_PRECISION=$PRECISION, log: $LOG)"
