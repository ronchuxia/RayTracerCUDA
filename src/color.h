#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include "interval.h"

#include <iostream>

using color = vec3;

__host__ __device__ inline real linear_to_gamma(real linear_component)
{
    return sqrt(linear_component);
}

// Convert one accumulated pixel (the SUM of `samples_per_pixel` samples, in
// linear space) to its three 8-bit display channels: average, gamma-correct,
// clamp to [0,0.999], scale by 256. This is the single source of truth for the
// tonemap, shared by the offline PPM writer (write_pixel) and the viewer's
// GPU tonemap kernel so the two present the byte-identical image.
__host__ __device__ inline void tonemap_pixel(color pixel_color, int samples_per_pixel,
                                              unsigned char& r, unsigned char& g, unsigned char& b) {
    real scale = real(1.0) / samples_per_pixel;
    const interval intensity(0.000, 0.999);
    r = static_cast<unsigned char>(256 * intensity.clamp(linear_to_gamma(pixel_color.x() * scale)));
    g = static_cast<unsigned char>(256 * intensity.clamp(linear_to_gamma(pixel_color.y() * scale)));
    b = static_cast<unsigned char>(256 * intensity.clamp(linear_to_gamma(pixel_color.z() * scale)));
}

// Offline path: write one pixel as a PPM ASCII triplet.
void write_pixel(std::ostream &out, color pixel_color, int samples_per_pixel) {
    unsigned char r, g, b;
    tonemap_pixel(pixel_color, samples_per_pixel, r, g, b);
    out << static_cast<int>(r) << ' '
        << static_cast<int>(g) << ' '
        << static_cast<int>(b) << '\n';
}

#endif // COLOR_H