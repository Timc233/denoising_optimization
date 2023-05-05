// gcc -o test_timed_baseline test_timed_baseline.c ../baseline/*.c ../util/*.c -lm

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../baseline/wavelet_denoise.h"
#include "../util/stb_image.h"
#include "../util/stb_image_write.h"
#include "../util/image_padding.h"
#include "../util/set_data_type.h"

#define NUM_RUNS 10 //power of 2


int main(int argc, char *argv[]) {

    char* input_image = "../images/highres.jpg";
    char* output_image = "../images/highres_denoised.png";
    
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

    printf("run_no, loop_length, elapsed_time\n");

    // Reexecute the looped operation by NUM_RUNS times

    for(int i = 0; i < NUM_RUNS; i++){

        // loop length is 2 to the power of NUM_RUNS
        int loop_length = 1 << i;

        clock_t start = clock();

        for (int i = 0; i < loop_length; i++) {
            wavelet_denoise(padded_image, padded_width, padded_height, channels, padded_image);
        }

        clock_t end = clock();
        double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;

        printf("%d, %d, %f\n", i, loop_length, elapsed_time);
    }  
    

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