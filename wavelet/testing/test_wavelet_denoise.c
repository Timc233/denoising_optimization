// gcc -o test_wavelet_denoise test_wavelet_denoise.c ../baseline/*.c ../util/*.c -lm


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../baseline/wavelet_denoise.h"
#include "../util/stb_image.h"
#include "../util/stb_image_write.h"
#include "../util/image_padding.h"
#include "../util/set_data_type.h"




int main(int argc, char *argv[]) {

    // if (argc != 3) {
    //     printf("Usage: %s input_image output_image\n", argv[0]);
    //     return 1;
    // }

    char* input_image = "../images/noisy_lenna.png";
    char* output_image = "../images/lenna_denoised.png";
    
    int width, height, channels;
    unsigned char *input = stbi_load(input_image, &width, &height, &channels, 0);
    if (!input) {
        printf("Error: Could not load input image.\n");
        return 1;
    }

    // Image padding
    int padded_width = next_power_of_2(width);
    int padded_height = next_power_of_2(height);
    data_t *padded_image = (data_t *)malloc(padded_width * padded_height * channels * sizeof(data_t));
    pad_image(input, width, height, channels, padded_image);

    wavelet_denoise(padded_image, padded_width, padded_height, channels, padded_image);

    // Image unpadding
    unsigned char *output = (unsigned char *)malloc(width * height * channels);
    unpad_image(padded_image, width, height, channels, output);    

    // Save denoised image
    int success = stbi_write_png(output_image, width, height, channels, output, width * channels);
    if (!success) {
        printf("Error: Could not save output image.\n");
    }

    stbi_image_free(input);
    free(output);

    return 0;
}
