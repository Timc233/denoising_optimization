#ifndef HAAR_WAVELET_H
#define HAAR_WAVELET_H

#include <stdlib.h>
#include <math.h>

void haar_1d(double* data, int length);
void inverse_haar_1d(double* data, int length);
void haar_2d(double* data, int width, int height, int levels);
void inverse_haar_2d(double* data, int width, int height, int levels);

#endif // HAAR_WAVELET_H
