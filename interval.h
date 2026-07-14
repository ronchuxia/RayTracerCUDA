#ifndef INTERVAL_H
#define INTERVAL_H

class interval {
  public:
    double min, max;

    __host__ __device__ interval() : min(+1.0/0.0), max(-1.0/0.0) {} // Default interval is empty

    __host__ __device__ interval(double _min, double _max) : min(_min), max(_max) {}

    __host__ __device__ interval(const interval& a, const interval& b)
      : min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}

    __host__ __device__ double size() const {
        return max - min;
    }

    __host__ __device__ interval expand(double delta) const {
        auto padding = delta/2;
        return interval(min - padding, max + padding);
    }

    __host__ __device__ bool contains(double x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(double x) const {
        return min < x && x < max;
    }

    __host__ __device__ double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }
};

__host__ __device__ interval operator+(const interval& ival, double displacement) {
    return interval(ival.min + displacement, ival.max + displacement);
}

__host__ __device__ interval operator+(double displacement, const interval& ival) {
    return ival + displacement;
}

__host__ __device__ interval operator*(const interval& ival, double scale) {
    return interval(ival.min * scale, ival.max * scale);
}

__host__ __device__ interval operator*(double scale, const interval& ival) {
    return ival * scale;
}

#endif