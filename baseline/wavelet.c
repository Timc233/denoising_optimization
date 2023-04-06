#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../util/stb_image.h"
#include "../util/stb_image_write.h"
#include "haar_wavelet.h"

void wavelet_denoise(const unsigned char *input, int width, int height, int channels, unsigned char *output, int levels, double threshold) {
    int size = width * height;
    double *data = (double *)malloc(size * sizeof(double));

    // Convert input image to grayscale and store in the data array
    for (int i = 0; i < size; i++) {
        double gray_value = 0.299 * input[i * channels] + 0.587 * input[i * channels + 1] + 0.114 * input[i * channels + 2];
        data[i] = gray_value;
    }

    // Perform Haar wavelet transform
    haar_2d(data, width, height, levels);

    // Apply soft thresholding
    for (int i = 0; i < size; i++) {
        double value = data[i];
        data[i] = (value > threshold) ? value - threshold : ((value < -threshold) ? value + threshold : 0);
    }

    // Perform inverse Haar wavelet transform
    inverse_haar_2d(data, width, height, levels);

    // Copy denoised data to output image and convert it back to RGB
    for (int i = 0; i < size; i++) {
        unsigned char gray_value = (unsigned char)data[i];
        for (int j = 0; j < channels; j++) {
            output[i * channels + j] = gray_value;
        }
    }

    free(data);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s input_image output_image levels threshold\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char *input = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!input) {
        printf("Error: Could not load input image.\n");
        return 1;
    }

    unsigned char *output = (unsigned char *)malloc(width * height * channels);
    int levels = atoi(argv[3]);
    double threshold = atof(argv[4]);

    // Perform wavelet denoising
    wavelet_denoise(input, width, height, channels, output, levels, threshold);

    // Save denoised image
    int success = stbi_write_png(argv[2], width, height, channels, output, width * channels);
    if (!success) {
        printf("Error: Could not save output image.\n");
    }

    stbi_image_free(input);
    free(output);

    return 0;
}
