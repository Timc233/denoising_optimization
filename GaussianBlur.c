#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../dependency/stb/stb_image.h"
#include "../dependency/stb/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_KERNEL_SIZE 21

void gaussian_blur(unsigned char* image, int width, int height, int channels, int kernel_size, float sigma) {
    int radius = kernel_size / 2;
    float* kernel = (float*) malloc(kernel_size * sizeof(float));

    // Generate Gaussian kernel
    float sum = 0;
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = exp(-((float) (i - radius) * (i - radius)) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    // Apply Gaussian blur filter
    unsigned char* temp = (unsigned char*) malloc(width * height * channels * sizeof(unsigned char));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float acc = 0;
                for (int i = 0; i < kernel_size; i++) {
                    int x2 = x + i - radius;
                    if (x2 < 0) {
                        x2 = -x2;
                    }
                    if (x2 >= width) {
                        x2 = 2 * width - x2 - 1;
                    }
                    acc += kernel[i] * image[(y * width + x2) * channels + c];
                }
                temp[(y * width + x) * channels + c] = (unsigned char) round(acc);
            }
        }
    }

    // Copy blurred image back to input image
    for (int i = 0; i < width * height * channels; i++) {
        image[i] = temp[i];
    }

    // Free memory
    free(kernel);
    free(temp);
}

int main(int argc, char** argv) {
    // Check command line arguments
    if (argc != 4) {
        printf("Usage: %s <input image> <output image> <kernel size>\n", argv[0]);
        return -1;
    }

    // Load input image
    int width, height, channels;
    unsigned char* image = stbi_load(argv[1], &width, &height, &channels, 0);

    // Check if image was loaded successfully
    if (!image) {
        printf("Could not open or find the image: %s\n", argv[1]);
        return -1;
    }

    // Convert kernel size argument to integer
    int kernel_size = atoi(argv[3]);
    if (kernel_size < 1 || kernel_size > MAX_KERNEL_SIZE || kernel_size % 2 == 0) {
        printf("Invalid kernel size. Must be odd and between 1 and %d.\n", MAX_KERNEL_SIZE);
        return -1;
    }

    // Apply Gaussian blur filter
    gaussian_blur(image, width, height, channels, kernel_size, 1.0);

    // Save output image
    stbi_write_jpg(argv[2], width, height, channels, image, 100);

    // Free memory
    stbi_image_free(image);

		return 0;
}
   
