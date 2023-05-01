// Define the header file
#ifndef OMP_WAVELET_DENOISE_H
#define OMP_WAVELET_DENOISE_H
#include "../util/set_data_type.h"

void omp_wavelet_denoise(const data_t *input, int width, int height, int channels, data_t *output);

#endif // OMP_WAVELET_DENOISE_H