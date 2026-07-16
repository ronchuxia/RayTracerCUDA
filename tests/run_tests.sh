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

nonblack() {  # fail if a PPM has no nonzero sample (catches all-black renders)
    awk 'NR>3 { for (i=1;i<=NF;i++) if ($i+0 > 0) { found=1; exit } } END { exit !found }' "$1"
}

echo "== [1/6] unit tests: flat vs BVH hit equivalence, invariants, refit =="
nvcc tests/test_bvh.cu -o build/test_bvh $NVCC_FLAGS
./build/test_bvh

echo "== [2/6] end-to-end: byte-identical render, flat vs BVH, fixed seed =="
nvcc src/main.cu -o build/rt_flat $NVCC_FLAGS -DUSE_BVH=0 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh  $NVCC_FLAGS -DUSE_BVH=1 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat > build/flat.ppm 2>/dev/null
./build/rt_bvh  > build/bvh.ppm  2>/dev/null
cmp build/flat.ppm build/bvh.ppm
nonblack build/flat.ppm
echo "PASS: flat and BVH renders are byte-identical"

echo "== [3/6] Cornell scene (quads/triangle/box/transforms): byte-identical render, flat vs BVH =="
nvcc src/main.cu -o build/rt_flat_cornell $NVCC_FLAGS -DRT_SCENE=1 -DUSE_BVH=0 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh_cornell  $NVCC_FLAGS -DRT_SCENE=1 -DUSE_BVH=1 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat_cornell > build/flat_cornell.ppm 2>/dev/null
./build/rt_bvh_cornell  > build/bvh_cornell.ppm  2>/dev/null
cmp build/flat_cornell.ppm build/bvh_cornell.ppm
nonblack build/flat_cornell.ppm
echo "PASS: Cornell flat and BVH renders are byte-identical"

echo "== [4/6] badge scene (STL mesh, nested BVH + transforms): byte-identical render, flat vs BVH =="
# RT_BADGE_FIELD=0 skips the 3.8k-sphere field so the flat-list path stays fast;
# the mesh's own BVH is present in BOTH paths (USE_BVH only toggles the world level).
nvcc src/main.cu -o build/rt_flat_badge $NVCC_FLAGS -DRT_SCENE=2 -DRT_BADGE_FIELD=0 -DUSE_BVH=0 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh_badge  $NVCC_FLAGS -DRT_SCENE=2 -DRT_BADGE_FIELD=0 -DUSE_BVH=1 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat_badge > build/flat_badge.ppm 2>/dev/null
./build/rt_bvh_badge  > build/bvh_badge.ppm  2>/dev/null
cmp build/flat_badge.ppm build/bvh_badge.ppm
nonblack build/flat_badge.ppm
echo "PASS: badge flat and BVH renders are byte-identical"

echo "== [5/6] smoke scene (constant media): sanity render, no byte-compare =="
# constant_medium::hit is STOCHASTIC (samples a scatter distance), so flat and
# BVH traversal orders consume the per-pixel RNG stream differently — the two
# renders are equally-correct Monte Carlo estimates but NOT byte-identical.
# We render both paths and only require valid non-black output.
nvcc src/main.cu -o build/rt_flat_smoke $NVCC_FLAGS -DRT_SCENE=3 -DUSE_BVH=0 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh_smoke  $NVCC_FLAGS -DRT_SCENE=3 -DUSE_BVH=1 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat_smoke > build/flat_smoke.ppm 2>/dev/null
./build/rt_bvh_smoke  > build/bvh_smoke.ppm  2>/dev/null
nonblack build/flat_smoke.ppm
nonblack build/bvh_smoke.ppm
echo "PASS: smoke scene renders on both paths (byte-compare not applicable to stochastic media)"

echo "== [6/6] tinted-glass scene (Beer-Lambert absorbing dielectric): byte-identical render, flat vs BVH =="
# The absorbing dielectric is deterministic (no RNG in the tint), so flat and BVH
# renders must be byte-identical — unlike the stochastic smoke scene above.
nvcc src/main.cu -o build/rt_flat_glass $NVCC_FLAGS -DRT_SCENE=4 -DUSE_BVH=0 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh_glass  $NVCC_FLAGS -DRT_SCENE=4 -DUSE_BVH=1 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat_glass > build/flat_glass.ppm 2>/dev/null
./build/rt_bvh_glass  > build/bvh_glass.ppm  2>/dev/null
cmp build/flat_glass.ppm build/bvh_glass.ppm
nonblack build/flat_glass.ppm
echo "PASS: tinted-glass flat and BVH renders are byte-identical"

echo "ALL TESTS PASSED"
