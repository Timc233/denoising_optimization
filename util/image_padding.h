// header file for image padding
#ifndef IMAGE_PADDING_H
#define IMAGE_PADDING_H

#include "set_data_type.h"

int next_power_of_2(int n);
void pad_image(unsigned char *input, int width, int height, int channels, data_t *output);
void unpad_image(data_t *input, int width, int height, int channels, unsigned char *output);

#endif // IMAGE_PADDING_H
