/*
    nvcc -o median_denoise_cuda median_denoise_cuda.cu -lstdc++ -lm
    ./median_denoise_cuda ./data/noisy/color_saltpepper.png ./data/denoised/color_median_cuda.png
*/


#include <stdio.h>
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <cuda_runtime.h>

#define FILETER_SIZE 3
#define TILE_SIZE 32

__global__ void apply_median_filter_kernel_tile_halo(uint8_t *src, uint8_t *dst, int width, int height, int channels, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    int half_filter = FILETER_SIZE / 2;
    int tile_width = blockDim.x + FILETER_SIZE - 1;
    int const tile_size = TILE_SIZE + FILETER_SIZE - 1;

    __shared__ uint8_t tile[3][tile_size*tile_size]; // Assuming maximum block size 32x32 and maximum filter size 3x3

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x_tile = tx + half_filter;
    int y_tile = ty + half_filter;

    // Load main tile data
    if (x < width && y < height) {
        tile[c][y_tile * tile_width + x_tile] = src[(y * width + x) * channels + c];
    }

    // Load left and right halo data
    if (tx < half_filter) {
        if (x >= half_filter && y < height) {
            tile[c][y_tile * tile_width + tx] = src[(y * width + x - half_filter) * channels + c];
        }
        if (x + blockDim.x < width && y < height) {
            tile[c][y_tile * tile_width + (tx + blockDim.x + half_filter)] = src[(y * width + x + blockDim.x - half_filter) * channels + c];
        }
    }

    // Load top and bottom halo data
    if (ty < half_filter) {
        if (y >= half_filter && x < width) {
            tile[c][(ty) * tile_width + x_tile] = src[((y - half_filter) * width + x) * channels + c];
        }
        if (y + blockDim.y < height && x < width) {
            tile[c][(ty + blockDim.y + half_filter) * tile_width + x_tile] = src[((y + blockDim.y - half_filter) * width + x) * channels + c];
        }
    }
    __syncthreads();

    if (x < width && y < height) {
        int count = 0;
        uint8_t window[FILETER_SIZE*FILETER_SIZE]; // Assuming maximum filter size is 5x5

        for (int dy = -half_filter; dy <= half_filter; ++dy) {
            for (int dx = -half_filter; dx <= half_filter; ++dx) {
                int new_y = y_tile + dy;
                int new_x = x_tile + dx;

                if (new_y >= 0 && new_y < height && new_x >= 0 && new_x < width) {
                    window[count++] = tile[c][(new_y * tile_width) + new_x];
                }
            }
        }

        // Sort the window array
        for (int i = 0; i < count; ++i) {
            for (int j = i + 1; j < count; ++j) {
                if (window[i] > window[j]) {
                    uint8_t temp = window[i];
                    window[i] = window[j];
                    window[j] = temp;
                }
            }
        }

        dst[(y * width + x) * channels + c] = window[count / 2];
    }

}


__global__ void apply_median_filter_kernel_tile(uint8_t *src, uint8_t *dst, int width, int height, int channels, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    int half_filter = FILETER_SIZE / 2;
    int tile_width = blockDim.x + FILETER_SIZE - 1;
    int const tile_size = TILE_SIZE + FILETER_SIZE - 1;

    __shared__ uint8_t tile[3][tile_size*tile_size]; // Assuming maximum block size 32x32 and maximum filter size 3x3

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x_tile = tx + half_filter;
    int y_tile = ty + half_filter;

    // Load main tile data
    if (x < width && y < height) {
        tile[c][y_tile * tile_width + x_tile] = src[(y * width + x) * channels + c];
    }

    // Load halo data
    if (tx < half_filter && x >= half_filter) {
        tile[c][(y_tile) * tile_width + tx] = src[(y * width + x - half_filter) * channels + c];
    }
    if (ty < half_filter && y >= half_filter) {
        tile[c][(ty) * tile_width + x_tile] = src[((y - half_filter) * width + x) * channels + c];
    }

    __syncthreads();

    if (x < width && y < height) {
        int count = 0;
        uint8_t window[FILETER_SIZE*FILETER_SIZE]; // Assuming maximum filter size is 5x5

        for (int dy = -half_filter; dy <= half_filter; ++dy) {
            for (int dx = -half_filter; dx <= half_filter; ++dx) {
                int new_y = y_tile + dy;
                int new_x = x_tile + dx;

                if (new_y >= 0 && new_y < height && new_x >= 0 && new_x < width) {
                    window[count++] = tile[c][(new_y * tile_width) + new_x];
                }
            }
        }

        // Sort the window array
        for (int i = 0; i < count; ++i) {
            for (int j = i + 1; j < count; ++j) {
                if (window[i] > window[j]) {
                    uint8_t temp = window[i];
                    window[i] = window[j];
                    window[j] = temp;
                }
            }
        }

        dst[(y * width + x) * channels + c] = window[count / 2];
    }

}

__global__ void apply_median_filter_kernel_base(uint8_t *src, uint8_t *dst, int width, int height, int channels, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x < width && y < height) {
        int half_filter = filter_size / 2;
        int count = 0;
        uint8_t window[FILETER_SIZE * FILETER_SIZE]; // assuming maximum filter size is 5x5
        //uint8_t *window = new uint8_t[filter_size * filter_size];

        for (int dy = -half_filter; dy <= half_filter; ++dy) {
            for (int dx = -half_filter; dx <= half_filter; ++dx) {
                int new_y = y + dy;
                int new_x = x + dx;

                if (new_y >= 0 && new_y < height && new_x >= 0 && new_x < width) {
                    window[count++] = src[(new_y * width + new_x) * channels + c];
                }
            }
        }

        // Sort the window array
        for (int i = 0; i < count; ++i) {
            for (int j = i + 1; j < count; ++j) {
                if (window[i] > window[j]) {
                    uint8_t temp = window[i];
                    window[i] = window[j];
                    window[j] = temp;
                }
            }
        }

        dst[(y * width + x) * channels + c] = window[count / 2];
    }
}

void apply_median_filter(uint8_t *src, uint8_t *dst, int width, int height, int channels, int filter_size) {
    uint8_t *src_gpu, *dst_gpu;

    cudaMalloc((void **)&src_gpu, width * height * channels * sizeof(uint8_t));
    cudaMalloc((void **)&dst_gpu, width * height * channels * sizeof(uint8_t));

    cudaMemcpy(src_gpu, src, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, channels);

    apply_median_filter_kernel_tile_halo<<<gridDim, blockDim>>>(src_gpu, dst_gpu, width, height, channels, filter_size);
    
     // Wait for the kernel to finish execution
    cudaDeviceSynchronize();
    cudaMemcpy(dst, dst_gpu, width * height * channels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(src_gpu);
    cudaFree(dst_gpu);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    // const char *input_file = "input_image.png";
    // const char *output_file = "output_image.png";
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int filter_size = 3;

    int width, height, channels;
    uint8_t *input_image = stbi_load(input_file, &width, &height, &channels, 0);
    if (!input_image) {
        printf("Error loading image file: %s\n", input_file);
        return 1;
    }
    else {
        printf("Loaded image file: %s\n", input_file);
        printf("Image dimensions: %dpx x %dpx\n", width, height);
        printf("Number of channels: %d\n", channels);
    }

    uint8_t *output_image = (uint8_t *)malloc(width * height * channels * sizeof(uint8_t));
/*
    uint8_t* src = input_image;
    uint8_t* dst = output_image;

    uint8_t *src_gpu;
    uint8_t *dst_gpu;

    // Allocate memory on the GPU
    cudaMalloc((void**)&src_gpu, width * height * channels * sizeof(uint8_t));
    cudaMalloc((void**)&dst_gpu, width * height * channels * sizeof(uint8_t));

    // Copy the input image data from the host to the GPU
    cudaMemcpy(src_gpu, src, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Set up the execution configuration for the CUDA kernel
    dim3 blockDim(8, 8);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);
    printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);

    // Launch the CUDA kernel
    apply_median_filter_kernel_base<<<gridDim, blockDim>>>(src_gpu, dst_gpu, width, height, channels, filter_size);

    //apply_median_filter_kernel_tile<<<gridDim, blockDim, channels * blockDim.x * blockDim.y * sizeof(uint8_t)>>>(src_gpu, dst_gpu, width, height, channels, filter_size);

    // Copy the output image data from the GPU back to the host
    cudaMemcpy(dst, dst_gpu, width * height * channels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Release GPU memory and other resources
    cudaFree(src_gpu);
    cudaFree(dst_gpu);
    */

    apply_median_filter(input_image, output_image, width, height, channels, filter_size);

    if (!stbi_write_png(output_file, width, height, channels, output_image, width * channels)) {
        printf("Error writing output image file: %s\n", output_file);
    }

    stbi_image_free(input_image);
    free(output_image);
    return 0;
}