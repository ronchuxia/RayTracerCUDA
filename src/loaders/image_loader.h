#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "cuda_helper.h"
#include "texture.h"

// stb_image implementation lives in this header; the include guard keeps it
// single-definition within each translation unit (this repo is single-TU).
#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"

// Host-side image loader: reads an image file (jpg/png/... via stb_image),
// copies the RGB8 pixels into managed memory (recorded in `allocs`), and
// returns an IMAGE texture pointing at them — embeddable by value in a
// material like any other texture.
inline texture load_image_texture(const char* path, std::vector<void*>& allocs) {
    int width, height, components;
    unsigned char* pixels = stbi_load(path, &width, &height, &components, 3);
    if (!pixels) {
        std::cerr << "load_image_texture: cannot load '" << path << "'\n";
        std::exit(1);
    }
    std::clog << "Loading image texture: " << path << " (" << width << "x" << height << ")\n" << std::flush;

    unsigned char* data;
    checkCudaErrors(cudaMallocManaged((void**)&data, (size_t)width * height * 3));
    memcpy(data, pixels, (size_t)width * height * 3);
    stbi_image_free(pixels);
    allocs.push_back(data);

    return make_image(data, width, height);
}

#endif // IMAGE_LOADER_H
