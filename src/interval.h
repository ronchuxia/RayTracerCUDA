#ifndef INTERVAL_H
#define INTERVAL_H

#include "precision.h"

struct interval {
    real min, max;

    __host__ __device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    __host__ __device__ interval(real _min, real _max) : min(_min), max(_max) {}

    __host__ __device__ interval(const interval& a, const interval& b)
      : min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}

    __host__ __device__ real size() const {
        return max - min;
    }

    __host__ __device__ interval expand(real delta) const {
        auto padding = delta/2;
        return interval(min - padding, max + padding);
    }

    __host__ __device__ bool contains(real x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(real x) const {
        return min < x && x < max;
    }

    __host__ __device__ real clamp(real x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }
};

__host__ __device__ interval operator+(const interval& ival, real displacement) {
    return interval(ival.min + displacement, ival.max + displacement);
}

__host__ __device__ interval operator+(real displacement, const interval& ival) {
    return ival + displacement;
}

__host__ __device__ interval operator*(const interval& ival, real scale) {
    return interval(ival.min * scale, ival.max * scale);
}

__host__ __device__ interval operator*(real scale, const interval& ival) {
    return ival * scale;
}

#endif