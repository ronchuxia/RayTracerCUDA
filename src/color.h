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

// color -> RGB8
__host__ __device__ inline void tonemap_pixel(color pixel_color, int samples_per_pixel,
                                              unsigned char& r, unsigned char& g, unsigned char& b,
                                              bool gamma = true) {
    real scale = real(1.0) / samples_per_pixel;
    const interval intensity(0.000, 0.999);
    color c = pixel_color * scale;
    if (gamma) c = color(linear_to_gamma(c.x()), linear_to_gamma(c.y()), linear_to_gamma(c.z()));
    r = static_cast<unsigned char>(256 * intensity.clamp(c.x()));
    g = static_cast<unsigned char>(256 * intensity.clamp(c.y()));
    b = static_cast<unsigned char>(256 * intensity.clamp(c.z()));
}

// write one pixel as a PPM ASCII triplet
void write_pixel(std::ostream &out, color pixel_color, int samples_per_pixel) {
    unsigned char r, g, b;
    tonemap_pixel(pixel_color, samples_per_pixel, r, g, b);
    out << static_cast<int>(r) << ' '
        << static_cast<int>(g) << ' '
        << static_cast<int>(b) << '\n';
}

#endif // COLOR_H