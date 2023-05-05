#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../dependency/stb/stb_image.h"
#include "../dependency/stb/stb_image_write.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_KERNEL_SIZE 21
#define SIGMA 3f
void gaussian_blur(unsigned char* image, int width, int height, int channels, int kernel_size, float* kernel) {
    
    int radius = kernel_size / 2;
    
    // Apply 2D Gaussian blur filter
    unsigned char* temp = (unsigned char*) malloc(width * height * channels * sizeof(unsigned char));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float acc = 0;
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int x2 = x + kx;
                        int y2 = y + ky;
                        if (x2 < 0) {
                            x2 = -x2;
                        }
                        if (x2 >= width) {
                            x2 = 2 * width - x2 - 1;
                        }
                        if (y2 < 0) {
                            y2 = -y2;
                        }
                        if (y2 >= height) {
                            y2 = 2 * height - y2 - 1;
                        }
                        acc += kernel[(ky+radius) * kernel_size + (kx+radius)] * image[(y2 * width + x2) * channels + c];
                    }
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
double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
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

    // Parse kernel size argument
    int kernel_size = atoi(argv[3]);
    if (kernel_size % 2 == 0 || kernel_size < 1 || kernel_size > MAX_KERNEL_SIZE) {
        printf("Kernel size must be an odd integer between 1 and %d\n", MAX_KERNEL_SIZE);
        return -1;
    }
    float* kernel = (float*) malloc(kernel_size * kernel_size * sizeof(float));
    int radius = kernel_size / 2;
    // Generate 2D Gaussian kernel
    float sum = 0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float value = exp(-(x*x + y*y) / (2 * SIGMA * SIGMA))/ (2 * M_PI * SIGMA * SIGMA);;
            kernel[(y+radius) * kernel_size + (x+radius)] = value;
            sum += value;
        }
    }
    float total=0;
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] /= sum;
        printf("%f\n", kernel[i]);
        total+= kernel[i];
    }
    printf("total=%f\n",total);
    
    struct timespec time_start, time_stop;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    // Apply Gaussian blur filter
    gaussian_blur(image, width, height, channels, kernel_size, kernel);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    double meas = interval(time_start, time_stop);
    printf("\nAll times are in seconds\n");
    printf("input_size, kernel_size, regular\n");
    printf("%ld, %ld, %10.4g\n", width * height, kernel_size * kernel_size, meas);
    // Save output image
    stbi_write_jpg(argv[2], width, height, channels, image, 100);

    // Free memory
    stbi_image_free(image);

    return 0;
}

   
