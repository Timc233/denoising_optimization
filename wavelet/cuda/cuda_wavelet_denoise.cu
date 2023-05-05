# include "cuda_wavelet_denoise.h"
# include "cuda_haar_wavelet.h"

# include "../util/set_data_type.h"
# include <math.h>
# include <stdlib.h>
# include <stdio.h>

void threshold_cut(data_t *data, int len, data_t threshold) {
    for (int i = 0; i < len; i++) {
        if (fabs(data[i]) < threshold) {
            data[i] = 0;
        }
    }
}

void cuda_wavelet_denoise(const data_t *input, int width, int height, int channels, data_t *output) {
    int size = width * height;
    data_t *data = (data_t *)malloc(size * sizeof(data_t));

    for (int ch = 0; ch < channels; ch++) {
        // Copy input channel to the data array
        for (int i = 0; i < size; i++) {
            data[i] = input[i * channels + ch];
        }

        // Perform Haar wavelet transform
        // haar_2d(data, width, height, levels);
        cuda_haar_wavelet_transform_2d(data, width, height);

        // Threshold the wavelet coefficients
        threshold_cut(data, size, 5);


        // Perform inverse Haar wavelet transform
        // inverse_haar_2d(data, width, height, levels);
        cuda_inverse_haar_wavelet_transform_2d(data, width, height);

        // Copy denoised data back to the output image
        for (int i = 0; i < size; i++) {
            output[i * channels + ch] = (data_t)data[i];
        }
    }

    free(data);
}

void looped_cuda_wavelet_denoise(const data_t *input, int width, int height, int channels, data_t *output, int loops, float *time) {

    // GPU time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start GPU time measurement
    cudaEventRecord(start, 0);

    for (int i = 0; i < loops; i++) {
        cuda_wavelet_denoise(input, width, height, channels, output);
    }

    // Stop GPU time measurement
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float measured_time = 0;
    cudaEventElapsedTime(&measured_time, start, stop);

    *time = measured_time;

    // printf("Measured time: %f ms\n", measured_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

  

}