

#ifndef HAAR_WAVELET_H
#define HAAR_WAVELET_H

#include <stdlib.h>
#include <math.h>
#include "../util/set_data_type.h"

void haar_wavelet_transform_1d(data_t *data, int len);
void haar_wavelet_transform_2d(data_t *data, int width, int height);
void inverse_haar_wavelet_transform_1d(data_t* wavelet_coeffs, int n);
void inverse_haar_wavelet_transform_2d(data_t *data, int width, int height);

#endif // HAAR_WAVELET_H
