

#ifndef OMP_HAAR_WAVELET_H
#define OMP_HAAR_WAVELET_H

#include <stdlib.h>
#include <math.h>
#include "../util/set_data_type.h"

void haar_wavelet_transform_1d(data_t *data, int len);
void omp_haar_wavelet_transform_2d(data_t *data, int width, int height);
void inverse_haar_wavelet_transform_1d(data_t* wavelet_coeffs, int n);
void omp_inverse_haar_wavelet_transform_2d(data_t *data, int width, int height);

void omp_inverse_haar_wavelet_transform_1d(data_t* wavelet_coeffs, int n);
void omp_haar_wavelet_transform_1d(data_t *data, int len);

#endif // OMP_HAAR_WAVELET_H
