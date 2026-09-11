#!/usr/bin/env bash
# BVH test suite: device unit tests + byte-identical end-to-end render check.
# Run from anywhere; binaries and images land in build/.
#   ARCH=sm_75 tests/run_tests.sh   # override the auto-detected GPU arch
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

if [ -z "${ARCH:-}" ]; then
    # Query GPU 0 explicitly (one line, so no `head` — avoids a SIGPIPE race
    # under pipefail); `|| true` tolerates nvidia-smi being absent.
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | tr -d '. ' || true)
    ARCH="sm_${CC:-86}"
fi
# The byte-identical flat-vs-BVH stages run at RT_PRECISION=64: float (32,
# the app default) is still deterministic per build, but rays that graze a
# shared edge between adjacent faces can tie in t within a float ulp, making
# the winning surface traversal-order-dependent — so exact flat-vs-BVH
# equality is a 64-bit property. Stage [7/12] checks the float build with a
# small pixel tolerance instead. PRECISION=32 forces the whole suite to float.
PRECISION="${PRECISION:-64}"
# Whole-program device compilation (the -rdc=false default). Until 2026-09-09
# every build needed -rdc=true because whole-program compilation miscompiled
# the recursive hittable::hit dispatch; the two-level scene has no recursion
# (docs/finished-plans/two-level-scene.md), -rdc=false renders byte-identically to
# -rdc=true at fp64 and is 7-20 % faster on the trace kernel. Stage [10/12]
# guards both flags. See docs/issues/rdc-recursive-dispatch-corruption.md.
NVCC_FLAGS="-std=c++14 -arch=$ARCH -Isrc -DRT_PRECISION=$PRECISION"
echo "using -arch=$ARCH, RT_PRECISION=$PRECISION"

nonblack() {  # fail if a PPM has no nonzero sample (catches all-black renders)
    awk 'NR>3 { for (i=1;i<=NF;i++) if ($i+0 > 0) { found=1; exit } } END { exit !found }' "$1"
}

pixdiff() {  # print number of differing pixels between two same-size P3 PPMs (one pixel per line)
    diff <(tail -n +4 "$1") <(tail -n +4 "$2") | grep -c '^<' || true
}

echo "== [1/12] unit tests: flat vs BVH hit equivalence, invariants, refit =="
nvcc tests/test_bvh.cu -o build/test_bvh $NVCC_FLAGS
./build/test_bvh

echo "== [2/12] end-to-end: byte-identical render, flat vs BVH, fixed seed =="
# The reference build compiles with -DBVH_LEAF_SIZE=1000000: every BVH is one
# leaf whose traversal visits the items in insertion order, i.e. the brute-force
# walk, through the same code. The other build uses the default leaf size.
nvcc src/main.cu -o build/rt_flat $NVCC_FLAGS -DBVH_LEAF_SIZE=1000000 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh  $NVCC_FLAGS -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat > build/flat.ppm 2>/dev/null
./build/rt_bvh  > build/bvh.ppm  2>/dev/null
cmp build/flat.ppm build/bvh.ppm
nonblack build/flat.ppm
echo "PASS: flat and BVH renders are byte-identical"

echo "== [3/12] Cornell scene (quads/triangle/box/transforms): byte-identical render, flat vs BVH =="
nvcc src/main.cu -o build/rt_flat_cornell $NVCC_FLAGS -DRT_SCENE=1 -DBVH_LEAF_SIZE=1000000 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh_cornell  $NVCC_FLAGS -DRT_SCENE=1 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat_cornell > build/flat_cornell.ppm 2>/dev/null
./build/rt_bvh_cornell  > build/bvh_cornell.ppm  2>/dev/null
cmp build/flat_cornell.ppm build/bvh_cornell.ppm
nonblack build/flat_cornell.ppm
echo "PASS: Cornell flat and BVH renders are byte-identical"

echo "== [4/12] badge scene (STL mesh, nested BVH + transforms): byte-identical render, flat vs BVH =="
# RT_BADGE_FIELD=0 skips the 3.8k-sphere field so the flat-list path stays fast;
# BVH_LEAF_SIZE=1000000 makes every tree (world AND mesh) one leaf walked in insertion order.
nvcc src/main.cu -o build/rt_flat_badge $NVCC_FLAGS -DRT_SCENE=2 -DRT_BADGE_FIELD=0 -DBVH_LEAF_SIZE=1000000 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh_badge  $NVCC_FLAGS -DRT_SCENE=2 -DRT_BADGE_FIELD=0 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat_badge > build/flat_badge.ppm 2>/dev/null
./build/rt_bvh_badge  > build/bvh_badge.ppm  2>/dev/null
cmp build/flat_badge.ppm build/bvh_badge.ppm
nonblack build/flat_badge.ppm
echo "PASS: badge flat and BVH renders are byte-identical"

echo "== [5/12] smoke scene (constant media): sanity render, no byte-compare =="
# medium_hit is STOCHASTIC (samples a scatter distance), so flat and
# BVH traversal orders consume the per-pixel RNG stream differently — the two
# renders are equally-correct Monte Carlo estimates but NOT byte-identical.
# We render both paths and only require valid non-black output.
nvcc src/main.cu -o build/rt_flat_smoke $NVCC_FLAGS -DRT_SCENE=3 -DBVH_LEAF_SIZE=1000000 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh_smoke  $NVCC_FLAGS -DRT_SCENE=3 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat_smoke > build/flat_smoke.ppm 2>/dev/null
./build/rt_bvh_smoke  > build/bvh_smoke.ppm  2>/dev/null
nonblack build/flat_smoke.ppm
nonblack build/bvh_smoke.ppm
echo "PASS: smoke scene renders on both paths (byte-compare not applicable to stochastic media)"

echo "== [6/12] tinted-glass scene (Beer-Lambert absorbing dielectric): byte-identical render, flat vs BVH =="
# The absorbing dielectric is deterministic (no RNG in the tint), so flat and BVH
# renders must be byte-identical — unlike the stochastic smoke scene above.
nvcc src/main.cu -o build/rt_flat_glass $NVCC_FLAGS -DRT_SCENE=4 -DBVH_LEAF_SIZE=1000000 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh_glass  $NVCC_FLAGS -DRT_SCENE=4 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat_glass > build/flat_glass.ppm 2>/dev/null
./build/rt_bvh_glass  > build/bvh_glass.ppm  2>/dev/null
cmp build/flat_glass.ppm build/bvh_glass.ppm
nonblack build/flat_glass.ppm
echo "PASS: tinted-glass flat and BVH renders are byte-identical"

echo "== [7/12] float build (RT_PRECISION=32): flat vs BVH within edge-graze tolerance =="
# Float keeps per-build determinism but loses order-invariance for rays that
# graze a shared edge between adjacent faces (hit t ties within one ulp, so
# the winning surface depends on traversal order). Allow a handful of such
# pixels on the box-heavy Cornell scene; anything systematic still fails.
nvcc src/main.cu -o build/rt_flat_f32 -std=c++14 -arch=$ARCH -Isrc -DRT_PRECISION=32 -DRT_SCENE=1 -DBVH_LEAF_SIZE=1000000 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
nvcc src/main.cu -o build/rt_bvh_f32  -std=c++14 -arch=$ARCH -Isrc -DRT_PRECISION=32 -DRT_SCENE=1 -DRT_SEED=42 -DRT_IMAGE_WIDTH=200 -DRT_SAMPLES=16
./build/rt_flat_f32 > build/flat_f32.ppm 2>/dev/null
./build/rt_bvh_f32  > build/bvh_f32.ppm  2>/dev/null
nonblack build/flat_f32.ppm
NDIFF=$(pixdiff build/flat_f32.ppm build/bvh_f32.ppm)
echo "float flat-vs-BVH differing pixels: $NDIFF / 40000 (tolerance 20)"
[ "$NDIFF" -le 20 ]
echo "PASS: float flat and BVH renders agree within edge-graze tolerance"

echo "== [8/12] scene ids: stamping, outermost-wrapper semantics, mutate->refit->re-pick =="
# The mutable-scene foundation: scene::add() ids land in hit_record.id (the
# OUTERMOST tagged wrapper wins, so a box face reports the transform chain's
# id), flat and BVH traversals agree, and moving an object through its id +
# refit() relocates it for subsequent picks.
nvcc tests/test_ids.cu -o build/test_ids $NVCC_FLAGS
./build/test_ids

echo "== [9/12] TRS transform node: ray world<->object, Euler rotation, inverse-transpose normals =="
# The transform math is subtle (unnormalized-direction ray map, non-uniform-
# scale normals via inverse-transpose): translate-only agrees with the translate
# wrapper, each Euler axis rotates correctly, and a scaled sphere (ellipsoid) is
# hit where expected with a unit-length, unsheared normal.
nvcc tests/test_transform.cu -o build/test_transform $NVCC_FLAGS
./build/test_transform

echo "== [10/12] regression guard: the dispatch kernel runs every iteration under both -rdc flags =="
# docs/issues/rdc-recursive-dispatch-corruption.md. The reproducer's kernel runs
# a loop bounded by a uniform spp and cross-checks a register trip counter, a
# global-memory counter, and the observed bound. With the recursive dispatch
# (until 2026-09-09) whole-program compilation made a warp report incoherent
# counts; the two-level scene has no recursion and both builds must be clean.
# Built at BOTH precisions (the old fault was precision-dependent) and with
# both flags, so a returning miscompile fails the suite whichever way it leans.
for P in 32 64; do
    REPRO_FLAGS="-std=c++14 -arch=$ARCH -Isrc -DRT_PRECISION=$P"
    nvcc tests/repro_rdc_corruption.cu -o build/repro_rdc_false_$P $REPRO_FLAGS
    nvcc tests/repro_rdc_corruption.cu -o build/repro_rdc_true_$P  $REPRO_FLAGS -rdc=true
    ./build/repro_rdc_false_$P >/dev/null
    ./build/repro_rdc_true_$P  >/dev/null
    echo "PASS: fp$P reproducer is clean under -rdc=false and -rdc=true"
done

echo "== [11/12] physics module (src/physics.h): collision impulse + drop-settle =="
# Calls the REAL physics_step / solve_sequential (the shipping header), not a
# reimplementation — the payoff of extracting physics out of viewer.cu. Verifies
# head-on restitution and that dropped spheres settle on the ground without
# interpenetrating. Host-only, but built with nvcc (physics.h -> vec3.h).
nvcc tests/test_physics.cu -o build/test_physics $NVCC_FLAGS
./build/test_physics

echo "== [12/12] scene physics: authored bodies vs. the geometry they were built from =="
# Stage 12 tests physics.h over hand-built bodies. This tests the layer above:
# physics_utils.h's factories, the viewer scenes that call them, and the
# colliders derived from live hittables — none of which can be exercised without
# building a scene. Needs CUDA (scenes live in unified memory) but no display
# and no SDL2/GLEW, so unlike stage 7 it is never skipped.
nvcc tests/test_physics_scene.cu -o build/test_physics_scene $NVCC_FLAGS
./build/test_physics_scene

echo "ALL TESTS PASSED"
