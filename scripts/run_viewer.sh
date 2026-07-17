#!/usr/bin/env bash
# Launch the interactive viewer (roadmap B1). Needs a display — local, VNC, or
# `ssh -X`. Builds once via scripts/build_viewer.sh if build/viewer is missing;
# it does NOT rebuild an existing binary, so re-run build_viewer.sh after edits.
# Full output is teed to build/run_viewer.log (override with LOG=...).
#   scripts/run_viewer.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

# Tee all stdout+stderr to a log under build/ (and keep showing it on the terminal).
LOG="${LOG:-build/run_viewer.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

if [ ! -x build/viewer ]; then
    echo "build/viewer not found — building it first"
    scripts/build_viewer.sh
fi

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "WARNING: no DISPLAY/WAYLAND_DISPLAY set — the window can't open here."
    echo "Run over VNC or 'ssh -X'."
fi

exec ./build/viewer "$@"
