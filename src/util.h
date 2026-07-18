#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "precision.h"

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;
using std::string;

// Constants

const real pi = real(3.1415926535897932385);

// Utility Functions

__host__ __device__ inline real degrees_to_radians(real degrees) {
    return degrees * real(3.1415926535897932385) / real(180.0);
}

// Uniform draw in (0,1], at the render path's precision (RT_PRECISION):
// curand_uniform is the float generator, curand_uniform_double the double one.
__device__ inline real random_real(curandState *state) {
#if RT_PRECISION == 64
    return curand_uniform_double(state);
#else
    return curand_uniform(state);
#endif
}

__device__ inline real random_real(real min, real max, curandState *state) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_real(state);
}

__device__ inline int random_int(int min, int max, curandState *state) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_real(real(min), real(max+1), state));
}

#endif
