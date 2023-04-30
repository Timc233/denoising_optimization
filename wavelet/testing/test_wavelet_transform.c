// gcc -o test_wavelet_transform test_wavelet_transform.c ../baseline/haar_wavelet.c ../util/copy_array.c -lm

#include "../baseline/haar_wavelet.h"

#include <stdio.h>
#include <stdlib.h>

void main(){
    float test_array[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    // print original array
    printf("Original array: ");
    for (int i = 0; i < 8; i++){
        printf("%f ", test_array[i]);
    }

    // perform 1d haar wavelet transform
    haar_wavelet_transform_1d(test_array, 8);

    // print transformed array
    printf("\nTransformed array: ");
    for (int i = 0; i < 8; i++){
        printf("%f ", test_array[i]);
    }

    printf("\n");

    // perform 2d haar wavelet transform
    float test_array_2d[16] = {1, 2, 3, 4, 5, 6, 7, 8, 
                               9, 10, 11, 12, 13, 14, 15, 16};
    haar_wavelet_transform_2d(test_array_2d, 4, 4);

    // print transformed array
    printf("\nTransformed array: ");
    for (int i = 0; i < 16; i++){
        printf("%f ", test_array_2d[i]);
    }

    printf("\n");

    // perform 1d inverse haar wavelet transform
    inverse_haar_wavelet_transform_1d(test_array, 8);

    // print transformed array
    printf("\nInverse transformed array: ");
    for (int i = 0; i < 8; i++){
        printf("%f ", test_array[i]);
    }

    printf("\n");

    // perform 2d inverse haar wavelet transform
    inverse_haar_wavelet_transform_2d(test_array_2d, 4, 4);

    // print transformed array
    printf("\nInverse transformed array: ");
    for (int i = 0; i < 16; i++){
        printf("%f ", test_array_2d[i]);
    }

    printf("\n");
}