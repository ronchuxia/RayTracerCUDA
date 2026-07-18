#!/usr/bin/env bash
# Launch the interactive viewer (roadmap B1). Needs a display — local, VNC, or
# `ssh -X`. Builds once via scripts/build_viewer.sh if build/viewer is missing;
# it does NOT rebuild an existing binary, so re-run build_viewer.sh after edits.
# PRECISION=64 runs the double-precision build (build/viewer_fp64) instead of
# the default float build (build/viewer) — for comparing fp32 vs fp64 live.
# Full output is teed to build/run_viewer.log (override with LOG=...).
#   scripts/run_viewer.sh
#   PRECISION=64 scripts/run_viewer.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

# Tee all stdout+stderr to a log under build/ (and keep showing it on the terminal).
LOG="${LOG:-build/run_viewer.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

PRECISION="${PRECISION:-32}"
BIN=build/viewer
[ "$PRECISION" = 64 ] && BIN=build/viewer_fp64

if [ ! -x "$BIN" ]; then
    echo "$BIN not found — building it first (RT_PRECISION=$PRECISION)"
    PRECISION="$PRECISION" scripts/build_viewer.sh
fi

echo "launching $BIN (RT_PRECISION=$PRECISION)"
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "WARNING: no DISPLAY/WAYLAND_DISPLAY set — the window can't open here."
    echo "Run over VNC or 'ssh -X'."
fi

exec "./$BIN" "$@"
