
// gcc -o test_timed_openmp test_timed_openmp.c ../openmp/*.c ../util/*.c -lm -fopenmp -lrt

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define _POSIX_C_SOURCE 199309L
// includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <math.h>


#include "../openmp/omp_wavelet_denoise.h"
#include "../util/stb_image.h"
#include "../util/stb_image_write.h"
#include "../util/image_padding.h"
#include "../util/set_data_type.h"

#define NUM_RUNS 10


int clock_gettime(clockid_t clk_id, struct timespec *tp);

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:
 
        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      measurement = interval(time_start, time_stop);

 */


/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

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

    struct timespec time_start, time_stop;

    // Reexecute the looped operation by NUM_RUNS times
    // time_t start_time, end_time;
    // double duration;
    for(int i = 0; i < NUM_RUNS; i++){
        
        double meas = 0;
        // loop length is 2 to the power of NUM_RUNS
        int loop_length = 1 << i;

        // start_time = time(NULL);
        clock_gettime(CLOCK_REALTIME, &time_start);

        for (int i = 0; i < loop_length; i++) {
            omp_wavelet_denoise(padded_image, padded_width, padded_height, channels, padded_image);
        }

        // end_time = time(NULL);
        // duration = difftime(end_time, start_time);

        clock_gettime(CLOCK_REALTIME, &time_stop);
        meas = interval(time_start, time_stop);

        printf("%d, %d, %f\n", i, loop_length, meas);
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