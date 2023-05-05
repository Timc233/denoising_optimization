// Define the header file
#ifndef WAVELET_DENOISE_H
#define WAVELET_DENOISE_H
# include "../util/set_data_type.h"


void wavelet_denoise(const data_t *input, int width, int height, int channels, data_t *output);

#endif // WAVELET_DENOISE_H