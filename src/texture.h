#ifndef TEXTURE_H
#define TEXTURE_H

#include "color.h"
#include "interval.h"
#include "vec3.h"

// Textures as a tagged union, mirroring material.h's model: a TextureType tag
// plus a union of the concrete variants, dispatched by a switch in value().
// A texture is a small self-contained value (the image variant just points at
// pixel data in managed memory), so materials embed it BY VALUE — no texture*
// indirection or extra allocations.

enum TextureType {
    SOLID_COLOR,
    CHECKER,
    IMAGE
};

struct solid_color {
    color albedo;

    __device__ color value(real u, real v, const point3& p) const {
        return albedo;
    }
};

struct checker_texture {
    real inv_scale;
    color even, odd;   // solid colors (no nested textures, unlike the book)

    __device__ color value(real u, real v, const point3& p) const {
        // The book's 3-D spatial checker: parity of the hit point's integer
        // lattice cell (independent of the surface's UV).
        auto x = static_cast<int>(floor(inv_scale * p.x()));
        auto y = static_cast<int>(floor(inv_scale * p.y()));
        auto z = static_cast<int>(floor(inv_scale * p.z()));
        return (x + y + z) % 2 == 0 ? even : odd;
    }
};

struct image_texture {
    unsigned char* data;   // RGB8, row-major; managed memory owned by the scene's allocs
    int width, height;

    __device__ color value(real u, real v, const point3& p) const {
        // Debug aid: solid cyan if there is no image data.
        if (data == nullptr || height <= 0) return color(0, 1, 1);

        u = interval(0, 1).clamp(u);
        v = 1.0 - interval(0, 1).clamp(v);  // flip V to image coordinates

        auto i = static_cast<int>(u * width);
        auto j = static_cast<int>(v * height);
        if (i > width - 1)  i = width - 1;
        if (j > height - 1) j = height - 1;

        const unsigned char* pixel = data + 3 * (j * width + i);
        real color_scale = real(1.0 / 255.0);
        return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
    }
};

struct texture {
    TextureType type;
    union {
        solid_color     solid;
        checker_texture checker;
        image_texture   image;
    };

    __host__ __device__ texture() : type(SOLID_COLOR) { solid.albedo = color(0, 0, 0); }

    // Implicit color → solid texture, so lambertian(color(...)) keeps working.
    __host__ __device__ texture(const color& c) : type(SOLID_COLOR) { solid.albedo = c; }

    __device__ color value(real u, real v, const point3& p) const {
        switch (type) {
            case SOLID_COLOR:
                return solid.value(u, v, p);
            case CHECKER:
                return checker.value(u, v, p);
            case IMAGE:
                return image.value(u, v, p);
            default:
                return color(0, 0, 0);
        }
    }
};

// Host-side factories, one per variant. Prefer these in new scene code for
// consistency; make_solid_color is the explicit spelling of the implicit
// color → texture conversion (which stays for back-compat with existing sites).

inline texture make_solid_color(const color& albedo) {
    texture t;
    t.type = SOLID_COLOR;
    t.solid.albedo = albedo;
    return t;
}

inline texture make_checker(real scale, const color& even, const color& odd) {
    texture t;
    t.type = CHECKER;
    t.checker.inv_scale = 1.0 / scale;
    t.checker.even = even;
    t.checker.odd = odd;
    return t;
}

inline texture make_image(unsigned char* data, int width, int height) {
    texture t;
    t.type = IMAGE;
    t.image.data = data;
    t.image.width = width;
    t.image.height = height;
    return t;
}

#endif // TEXTURE_H
