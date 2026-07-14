#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <curand_kernel.h>
#include <cuda_runtime.h>

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;
using std::string;

// Utility Functions

__host__ __device__ inline double degrees_to_radians(double degrees) {
    return degrees * 3.1415926535897932385 / 180.0;
}

__device__ inline double random_double(double min, double max, curandState *state) {
    // Returns a random real in [min,max).
    return min + (max-min)*curand_uniform_double(state);
}

__device__ inline int random_int(int min, int max, curandState *state) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_double(min, max+1, state));
}

#endif