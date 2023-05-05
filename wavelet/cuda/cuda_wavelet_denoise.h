// Define the header file
#ifndef CUDA_WAVELET_DENOISE_H
#define CUDA_WAVELET_DENOISE_H

# include "../util/set_data_type.h"




#ifdef __cplusplus
extern "C" {
#endif

// Declare the function prototypes
void cuda_wavelet_denoise(const data_t *input, int width, int height, int channels, data_t *output);
void looped_cuda_wavelet_denoise(const data_t *input, int width, int height, int channels, data_t *output, int loops, float *time);

#ifdef __cplusplus
}
#endif

#endif // CUDA_HAAR_WAVELET_H