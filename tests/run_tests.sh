#!/usr/bin/env bash
# BVH test suite: device unit tests + byte-identical end-to-end render check.
# Run from anywhere; binaries and images land in build/.
#   ARCH=sm_75 tests/run_tests.sh   # override the auto-detected GPU arch
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

if [ -z "${ARCH:-}" ]; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')
    ARCH="sm_${CC:-86}"
fi
NVCC_FLAGS="-std=c++14 -arch=$ARCH -Isrc"
echo "using -arch=$ARCH"

echo "== [1/2] unit tests: flat vs BVH hit equivalence, invariants, refit =="
nvcc tests/test_bvh.cu -o build/test_bvh $NVCC_FLAGS
./build/test_bvh

echo "== [2/2] end-to-end: byte-identical render, flat vs BVH, fixed seed =="
nvcc src/main.cu -o build/rt_flat $NVCC_FLAGS -DUSE_BVH=0 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh  $NVCC_FLAGS -DUSE_BVH=1 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat > build/flat.ppm 2>/dev/null
./build/rt_bvh  > build/bvh.ppm  2>/dev/null
cmp build/flat.ppm build/bvh.ppm
echo "PASS: flat and BVH renders are byte-identical"

echo "ALL TESTS PASSED"
