#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

void apply_median_filter(uint8_t *src, uint8_t *dst, int width, int height, int channels, int filter_size) {
    int half_filter = filter_size / 2;
    int offset = filter_size * filter_size;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t window[offset];
            int count = 0;

            for (int dy = -half_filter; dy <= half_filter; ++dy) {
                for (int dx = -half_filter; dx <= half_filter; ++dx) {
                    int new_y = y + dy;
                    int new_x = x + dx;

                    if (new_y >= 0 && new_y < height && new_x >= 0 && new_x < width) {
                        window[count++] = src[new_y * width + new_x];
                    }
                }
            }

            // Sort the window array
            for (int i = 0; i < count; ++i) {
                for (int j = i + 1; j < count; ++j) {
                    if (window[i] > window[j]) {
                        uint8_t temp = window[i];
                        window[i] = window[j];
                        window[j] = temp;
                    }
                }
            }

            dst[y * width + x] = window[count / 2];
        }
    }
}

int main() {
    const char *input_file = "input_image.png";
    const char *output_file = "output_image.png";
    int filter_size = 3;

    int width, height, channels;
    uint8_t *input_image = stbi_load(input_file, &width, &height, &channels, STBI_grey);
    if (!input_image) {
        printf("Error loading image file: %s\n", input_file);
        return 1;
    }

    uint8_t *output_image = (uint8_t *)malloc(width * height * sizeof(uint8_t));

    apply_median_filter(input_image, output_image, width, height, channels, filter_size);

    if (!stbi_write_png(output_file, width, height, 1, output_image, width)) {
        printf("Error writing output image file: %s\n", output_file);
    }

    stbi_image_free(input_image);
    free(output_image);
    return 0;
}
