
// Compile:
// nvcc -c ../cuda/cuda_haar_wavelet.cu ../cuda/cuda_wavelet_denoise.cu -lcuda -lcudart -lcublas
// gcc -c ../util/*.c
// gcc -o test_cuda test_cuda.c *.o -lm -L/usr/local/cuda/lib64 -lcudart -lstdc++



#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <time.h>
#include <stdio.h>

#include "../util/stb_image.h"
#include "../util/stb_image_write.h"
#include "../util/image_padding.h"
#include "../util/set_data_type.h"

#include "../cuda/cuda_wavelet_denoise.h"



#define NUM_RUNS 10

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
    float time = 0;

    printf("run_no, loop_length, elapsed_time\n");

    for(int i = 0; i < NUM_RUNS; i++){

        // loop length is 2 to the power of NUM_RUNS
        int loop_length = 1 << i;

        clock_t start = clock();

        looped_cuda_wavelet_denoise(padded_image, padded_width, padded_height, channels, padded_image, loop_length, &time);

        clock_t end = clock();
        double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;

        printf("%d, %d, %f\n", i, loop_length, elapsed_time);
    }  


    // // Measure cpu time
    // clock_t start = clock();

    // looped_cuda_wavelet_denoise(padded_image, padded_width, padded_height, channels, padded_image, NUM_RUNS, &time);

    // clock_t end = clock();

    // // calculate duration
    // time = (double)(end - start) / CLOCKS_PER_SEC;

    // // print time in seconds
    // printf("Elapsed time: %f\n", time);

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