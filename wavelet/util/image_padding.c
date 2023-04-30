// Given a 2D image, this function pads the image with zeros til the size becomes a power of 2.
# include "image_padding.h"
#include "../util/set_data_type.h"

int next_power_of_2(int n) {
    int m = 1;
    while (m < n) {
        m *= 2;
    }
    return m;
}


void pad_image(unsigned char *input, int width, int height, int channels, data_t *output) {
    int padded_width = next_power_of_2(width);
    int padded_height = next_power_of_2(height);

    for (int ch = 0; ch < channels; ch++) {
        for (int i = 0; i < padded_height; i++) {
            for (int j = 0; j < padded_width; j++) {
                if (i < height && j < width) {
                    output[(i * padded_width + j) * channels + ch] = input[(i * width + j) * channels + ch];
                } else {
                    output[(i * padded_width + j) * channels + ch] = 0;
                }
            }
        }
    }
}

void unpad_image(data_t *input, int width, int height, int channels, unsigned char *output) {
    int padded_width = next_power_of_2(width);
    int padded_height = next_power_of_2(height);

    for (int ch = 0; ch < channels; ch++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                output[(i * width + j) * channels + ch] = input[(i * padded_width + j) * channels + ch];
            }
        }
    }
}

