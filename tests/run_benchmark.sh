#!/usr/bin/env bash
# Dense-scene render benchmark: flat hittable_list vs. flattened BVH.
# Timings print to stderr; the PPM output is discarded.
#   ARCH=sm_75 tests/run_benchmark.sh   # override the auto-detected GPU arch
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

if [ -z "${ARCH:-}" ]; then
    # Query GPU 0 explicitly (one line, so no `head` — avoids a SIGPIPE race
    # under pipefail); `|| true` tolerates nvidia-smi being absent.
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | tr -d '. ' || true)
    ARCH="sm_${CC:-86}"
fi
echo "using -arch=$ARCH"

nvcc tests/bench_bvh.cu -o build/bench_bvh -std=c++14 -arch=$ARCH -Isrc
./build/bench_bvh > /dev/null
