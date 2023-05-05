# include "omp_wavelet_denoise.h"
# include "omp_haar_wavelet.h"
# include "../util/set_data_type.h"
# include <math.h>
# include <omp.h>

void threshold_cut(data_t *data, int len, data_t threshold) {
    #pragma omp for
    for (int i = 0; i < len; i++) {
        if (fabs(data[i]) < threshold) {
            data[i] = 0;
        }
    }
}



void omp_wavelet_denoise(const data_t *input, int width, int height, int channels, data_t *output) {
    int size = width * height;
    data_t *data = (data_t *)malloc(size * sizeof(data_t));


    for (int ch = 0; ch < channels; ch++) {
        // Copy input channel to the data array
        #pragma omp for
        for (int i = 0; i < size; i++) {
            data[i] = input[i * channels + ch];
        }

        // Perform Haar wavelet transform
        // haar_2d(data, width, height, levels);
        omp_haar_wavelet_transform_2d(data, width, height);

        // Threshold the wavelet coefficients
        threshold_cut(data, size, 5);


        // Perform inverse Haar wavelet transform
        // inverse_haar_2d(data, width, height, levels);
        omp_inverse_haar_wavelet_transform_2d(data, width, height);

        // Copy denoised data back to the output image
        #pragma omp for
        for (int i = 0; i < size; i++) {
            output[i * channels + ch] = (data_t)data[i];
        }
    }

    free(data);
}