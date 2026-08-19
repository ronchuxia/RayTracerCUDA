#!/usr/bin/env bash
# Launch the interactive viewer (roadmap B1). Needs a display — local, VNC, or
# `ssh -X`. Builds once via scripts/build_viewer.sh if the binary is missing;
# it does NOT rebuild an existing binary, so re-run build_viewer.sh after edits.
#
# SCENE picks which one to launch, using the same names build_viewer.sh writes —
# each scene is its own binary, so they coexist and can run side by side:
#   SCENE=0  build/viewer          primitives showcase (default)
#   SCENE=1  build/viewer_pit      ball pit, roomy 1.5, frictionless
#   SCENE=2  build/viewer_tight    ball pit, tight 1.3, frictionless
#   SCENE=3  build/viewer_rolling  ball pit, roomy 1.5, friction 0.5
#
# PRECISION=64 runs the double-precision build (…_fp64) instead of the default
# float one — for comparing fp32 vs fp64 live.
# Full output is teed to build/run_viewer.log (override with LOG=...).
#   scripts/run_viewer.sh
#   SCENE=3 scripts/run_viewer.sh
#   PRECISION=64 scripts/run_viewer.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

# Tee all stdout+stderr to a log under build/ (and keep showing it on the terminal).
LOG="${LOG:-build/run_viewer.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

# Same SCENE -> name mapping as build_viewer.sh; keep the two in step.
SCENE="${SCENE:-0}"
case "$SCENE" in
    0) NAME=viewer ;;
    1) NAME=viewer_pit ;;
    2) NAME=viewer_tight ;;
    3) NAME=viewer_rolling ;;
    *) echo "error: unknown SCENE=$SCENE (0 primitives, 1 pit, 2 tight, 3 rolling)" >&2
       exit 1 ;;
esac
PRECISION="${PRECISION:-32}"
BIN="build/$NAME"
[ "$PRECISION" = 64 ] && BIN="${BIN}_fp64"

if [ ! -x "$BIN" ]; then
    echo "$BIN not found — building it first (SCENE=$SCENE, RT_PRECISION=$PRECISION)"
    SCENE="$SCENE" PRECISION="$PRECISION" scripts/build_viewer.sh
fi

echo "launching $BIN (SCENE=$SCENE, RT_PRECISION=$PRECISION)"
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "WARNING: no DISPLAY/WAYLAND_DISPLAY set — the window can't open here."
    echo "Run over VNC or 'ssh -X'."
fi

exec "./$BIN" "$@"
