#ifndef CAMERA_H
#define CAMERA_H

#include "util.h"
#include "cuda_helper.h"
#include "curand_kernel.h"
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "hittable.h"
#include "material.h"

// Background on ray miss: 0 (default) = black, so scenes are lit only by
// diffuse_light objects; 1 = the book's sky gradient (the variant the fork
// later commented out, including its 0.7 dimming factor) — build with
// -DRT_SKY=1 to reproduce the fork's older sky-lit renders.
#ifndef RT_SKY
#define RT_SKY 0
#endif

struct camera;
__global__ void initialize_rand(const camera& cam, curandState* state, unsigned long seed);
__global__ void render_pixel(const camera& cam, int max_depth, const hittable& world, color* pixel_colors, curandState* rand_states);

struct camera {
        double aspect_ratio      = 1.0;  // Ratio of image width over height
        int    image_width       = 100;  // Rendered image width in pixel count
        int    image_height;             // Rendered image height in pixel count
        int    samples_per_pixel = 10;   // Count of random samples for each pixel
        int    max_depth         = 10;   // Maximum number of ray bounces into scene

        double vfov     = 90;              // Vertical view angle (field of view)
        point3 lookfrom = point3(0,0,-1);  // Point camera is looking from
        point3 lookat   = point3(0,0,0);   // Point camera is looking at
        vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

        double defocus_angle = 0;  // Variation angle of rays through each pixel
        double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

        long long seed = -1;       // RNG seed; negative → seed from time(0). Fix it for reproducible renders (tests).

        __host__ void render(const hittable& world) {
            initialize();

            // initialize random states
            curandState* rand_states;
            checkCudaErrors(cudaMallocManaged(&rand_states, image_width * image_height * sizeof(curandState)));
            unsigned long rng_seed = (seed < 0) ? (unsigned long)time(0) : (unsigned long)seed;
            initialize_rand<<<(image_width * image_height + 255) / 256, 256>>>(*this, rand_states, rng_seed);
            
            checkCudaErrors(cudaDeviceSynchronize());
            
            std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

            // allocate memory for pixel colors
            color* pixel_colors;
            checkCudaErrors(cudaMallocManaged(&pixel_colors, image_width * image_height * sizeof(color)));

            // render the pixel colors
            dim3 threads(16, 16);
            dim3 blocks((image_width + threads.x - 1) / threads.x, (image_height + threads.y - 1) / threads.y);
            render_pixel<<<blocks, threads>>>(*this, max_depth, world, pixel_colors, rand_states);

            checkCudaErrors(cudaDeviceSynchronize());

            // output the pixel colors
            for (int j = 0; j < image_height; ++j) {
                for (int i = 0; i < image_width; ++i) {
                    write_color(std::cout, pixel_colors[j * image_width + i], samples_per_pixel);
                }
            }

            // clean up
            checkCudaErrors(cudaFree(rand_states));
            checkCudaErrors(cudaFree(pixel_colors));
        }

        __device__ ray get_ray(int i, int j, curandState* rand_state) const {
            // Get a randomly-sampled camera ray for the pixel at location i,j, originating from
            // the camera defocus disk.
            auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto pixel_sample = pixel_center + pixel_sample_square(rand_state);

            auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(rand_state);
            auto ray_direction = pixel_sample - ray_origin;

            return ray(ray_origin, ray_direction);
        }

        __device__ color ray_color(ray r, const hittable& world, int max_depth, curandState* state) const {
            ray current_ray = r;
            color current_color = color(0,0,0); // Total color until now
            color throughput = color(1,1,1);    // Total attenuation until now
            hit_record rec;

            for (int i = 0; i < max_depth; i++) {
                if (world.hit(current_ray, interval(0.001, 1.0/0.0), rec, state)) {
                    ray scattered;
                    color attenuation;
                    color emit = rec.mat->emitted();

                    if (rec.mat->scatter(current_ray, rec, attenuation, scattered, state)) {
                        // If material scatters, accumulate attenuation, add emitted light and continue
                        throughput *= attenuation;
                        current_color += throughput * emit;
                        current_ray = scattered;
                    }
                    else {
                        // If material doesn't scatter, add emitted light and terminate
                        current_color += throughput * emit;
                        break;
                    }  
                }
                else {
                    // If no hit, add background color and terminate
#if RT_SKY
                    vec3 unit_direction = unit_vector(current_ray.direction());
                    auto a = 0.5 * (unit_direction.y() + 1.0);
                    color background = ((1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0)) * 0.7;
#else
                    color background = color(0,0,0);
#endif
                    current_color += throughput * background;
                    break;
                }
            }
            return current_color;
        }

    private:
        point3 center;          // Camera center
        point3 pixel00_loc;     // Location of pixel 0, 0
        vec3   pixel_delta_u;   // Offset to pixel to the right
        vec3   pixel_delta_v;   // Offset to pixel below
        vec3   u, v, w;         // Camera frame basis vectors
        vec3   defocus_disk_u;  // Defocus disk horizontal radius
        vec3   defocus_disk_v;  // Defocus disk vertical radius

        __host__ void initialize() {
            image_height = static_cast<int>(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            center = lookfrom;

            // Determine viewport dimensions.
            auto theta = degrees_to_radians(vfov);
            auto h = tan(theta/2);
            auto viewport_height = 2 * h * focus_dist;
            auto viewport_width = viewport_height * (static_cast<double>(image_width)/image_height);

            // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            // Calculate the vectors across the horizontal and down the vertical viewport edges.
            vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
            vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

            // Calculate the horizontal and vertical delta vectors to the next pixel.
            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            // Calculate the location of the upper left pixel.
            auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

            // Calculate the camera defocus disk basis vectors.
            auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
        }

        __device__ vec3 pixel_sample_square(curandState* state) const {
            // Returns a random point in the square surrounding a pixel at the origin.
            auto px = -0.5 + curand_uniform_double(state);
            auto py = -0.5 + curand_uniform_double(state);
            return (px * pixel_delta_u) + (py * pixel_delta_v);
        }

        __device__ point3 defocus_disk_sample(curandState* state) const {
            // Returns a random point in the camera defocus disk.
            auto p = random_in_unit_disk(state);
            return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
        }
};

__global__ void initialize_rand(const camera& cam, curandState* state, unsigned long seed) {
    // same seed, different sequence
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cam.image_width * cam.image_height) return;
    curand_init(seed, idx, 0, &state[idx]);
}

// The default translation of the color function results in a stack overflow since it can call 
// itself many times. But, the code can be simply be translated into a loop instead of recursion.
__global__ void render_pixel(const camera& cam, int max_depth, const hittable& world, color* pixel_colors, curandState* rand_states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= cam.image_width || j >= cam.image_height) return;

    // retrieve the random state for this pixel
    int pixel_index = j * cam.image_width + i;
    curandState* rand_state = &rand_states[pixel_index];

    // initialize black color for this pixel
    pixel_colors[pixel_index] = color(0,0,0);

    for (int sample = 0; sample < cam.samples_per_pixel; ++sample) {
        // sample a ray for this pixel
        ray r = cam.get_ray(i, j, rand_state);

        // render along the ray
        pixel_colors[pixel_index] += cam.ray_color(r, world, max_depth, rand_state);
    }
}

#endif // CAMERA_H