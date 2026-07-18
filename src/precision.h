#ifndef PRECISION_H
#define PRECISION_H

// RT_PRECISION selects the floating-point width of the entire render path
// (vec3/point3/color components, intervals, rays, cuRAND draws):
//   32 (default) — float. ~32-64x the FP64 ALU rate on workstation/consumer
//                  GPUs (measured 1/75 on the RTX A4000), so this is the fast
//                  path. Renders are NOT byte-comparable to the 64-bit build.
//   64           — double. Bit-identical to the original all-double renderer;
//                  kept as the precision reference ("oracle") for verifying
//                  the float path statistically.
// Every header in the render path uses `real`; keep new code on `real` too,
// and wrap floating-point literals in hot __device__ code as real(...) so
// they don't silently promote the math back to double.
#ifndef RT_PRECISION
#define RT_PRECISION 32
#endif

#if RT_PRECISION == 64
using real = double;
#elif RT_PRECISION == 32
using real = float;
#else
#error "RT_PRECISION must be 32 or 64"
#endif

// Compile-time +inf at the render precision. Use this instead of the 1.0/0.0
// idiom: nvcc emits a *runtime double division* (MUFU.RCP64H) for that
// expression in device code, dragging FP64 into the float build.
#include <limits>
constexpr real infinity = std::numeric_limits<real>::infinity();

#endif // PRECISION_H
