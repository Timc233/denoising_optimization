#ifndef CUDA_HAAR_WAVELET_H
#define CUDA_HAAR_WAVELET_H

#include "../util/set_data_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// Declare the function prototypes
void cuda_haar_wavelet_transform_1d(data_t *data, int len);
void cuda_haar_wavelet_transform_2d(data_t *data, int width, int height);

void cuda_inverse_haar_wavelet_transform_1d(data_t* wavelet_coeffs, int n);
void cuda_inverse_haar_wavelet_transform_2d(data_t *data, int width, int height);


#ifdef __cplusplus
}
#endif

#endif // CUDA_HAAR_WAVELET_H
