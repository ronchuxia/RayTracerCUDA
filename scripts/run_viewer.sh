#!/usr/bin/env bash
# Launch the interactive viewer.
#
#   SCENE=3 scripts/run_viewer.sh
#
#   SCENE    binary
#   0        build/viewer           primitives
#   1        build/viewer_pit       ball pit, roomy 1.5, frictionless
#   2        build/viewer_tight     ball pit, tight 1.3, frictionless
#   3        build/viewer_roll      ball pit, roomy 1.5, friction 0.5
#   4        build/viewer_spin      spinning earth balls, friction 0.5
#   5        build/viewer_denoise   room
#   6        build/viewer_dlss      dlss pit: mixed materials 
#   7        build/viewer_hull      hull pit: convex + balls
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

# tee all stdout+stderr to a log
LOG="${LOG:-build/run_viewer.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

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
BIN="build/$NAME"
[ "$PRECISION" = 64 ] && BIN="${BIN}_fp64"

if [ ! -x "$BIN" ]; then
    echo "$BIN not found — building it first (SCENE=$SCENE, RT_PRECISION=$PRECISION)"
    SCENE="$SCENE" PRECISION="$PRECISION" scripts/build_viewer.sh
fi

echo "launching $BIN (SCENE=$SCENE, RT_PRECISION=$PRECISION)"
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "WARNING: no DISPLAY/WAYLAND_DISPLAY set — the window can't open."
fi

exec "./$BIN"
