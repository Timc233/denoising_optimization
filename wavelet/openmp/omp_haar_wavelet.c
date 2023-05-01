#include "omp_haar_wavelet.h"
#include <math.h>
#include "../util/basic_util.h"
#include "../util/set_data_type.h"
#include <omp.h>

/**
 * Performs the Haar wavelet transform on the given array
 *
 * @param {data_t*} data - the array to transform
 * @param {int} n - the length of the array
 * 
 * The resulted transformed array will be stored in the data array.
 * The order of the output array would be [A_n, D_n, D_n-1, ..., D_1]
 * 
 */
void haar_wavelet_transform_1d(data_t *data, int len) {
    data_t *output = (data_t *)malloc(len * sizeof(data_t));
    copy_array(data, output, len);
    int level = log2(len);
    int iterations = level - 1;
    int n = len;
    for(int i = 0; i < iterations; i++){
        n = n / 2;
        data_t *A = (data_t *)malloc(n * sizeof(data_t));
        data_t *D = (data_t *)malloc(n * sizeof(data_t));
        for (int i = 0; i < n; i++) {
            A[i] = (output[i * 2] + output[i * 2 + 1]) / sqrt(2);
            D[i] = (output[i * 2] - output[i * 2 + 1]) / sqrt(2);
        }

        for (int i = 0; i < n; i++) {
            output[i] = A[i];
            output[n + i] = D[i];
        }

        free(A);
        free(D);     
    }

    copy_array(output, data, len);

    free(output);
}

void omp_haar_wavelet_transform_1d(data_t *data, int len) {
    data_t *output = (data_t *)malloc(len * sizeof(data_t));
    copy_array(data, output, len);
    int level = log2(len);
    int iterations = level - 1;
    int n = len;
    for(int i = 0; i < iterations; i++){
        n = n / 2;
        data_t *A = (data_t *)malloc(n * sizeof(data_t));
        data_t *D = (data_t *)malloc(n * sizeof(data_t));

        #pragma omp parallel shared(A, D, output, n) private(i)
        {
            #pragma omp for
            for (int i = 0; i < n; i++) {
                A[i] = (output[i * 2] + output[i * 2 + 1]) / sqrt(2);
                D[i] = (output[i * 2] - output[i * 2 + 1]) / sqrt(2);
            }
        }
        
        #pragma omp parallel shared(A, D, output, n) private(i)
        {
            #pragma omp for
            for (int i = 0; i < n; i++) {
                output[i] = A[i];
                output[n + i] = D[i];
            }
        }

        free(A);
        free(D);     
    }

    copy_array(output, data, len);

    free(output);
}

/**
 * Performs the inverse Haar wavelet transform on the given wavelet coefficients
 *
 * @param {data_t*} wavelet_coeffs - the wavelet coefficients to transform
 * @param {int} n - the length of the wavelet_coeffs array
 * @returns {data_t*} - the resulting output array
 * 
 * The order of the input coefficients array would be [A_n, D_n, D_n-1, ..., D_1]
 */
void inverse_haar_wavelet_transform_1d(data_t* wavelet_coeffs, int n) {
    int level = (int) log2(n);

    int a_length = 2;

    int iterations = level - 1;

    for (int i = 0; i < iterations; i++) {
        int new_a_length = a_length * 2;
        data_t* new_a = (data_t*) malloc(new_a_length * sizeof(data_t));

        for (int j = 0; j < a_length; j++) {
            new_a[2*j] = (wavelet_coeffs[j] + wavelet_coeffs[a_length+j]) / sqrt(2);
            new_a[2*j+1] = (wavelet_coeffs[j] - wavelet_coeffs[a_length+j]) / sqrt(2);
        }

        for (int j = 0; j < new_a_length; j++) {
            wavelet_coeffs[j] = new_a[j];
        }

        a_length = new_a_length;
        free(new_a);
    }

}

void omp_inverse_haar_wavelet_transform_1d(data_t* wavelet_coeffs, int n) {
    int level = (int) log2(n);

    int a_length = 2;

    int iterations = level - 1;

    for (int i = 0; i < iterations; i++) {
        int new_a_length = a_length * 2;
        data_t* new_a = (data_t*) malloc(new_a_length * sizeof(data_t));

        #pragma omp parallel shared(new_a, wavelet_coeffs, a_length) private(i)
        {
            #pragma omp for
            for (int j = 0; j < a_length; j++) {
                new_a[2*j] = (wavelet_coeffs[j] + wavelet_coeffs[a_length+j]) / sqrt(2);
                new_a[2*j+1] = (wavelet_coeffs[j] - wavelet_coeffs[a_length+j]) / sqrt(2);
            }
        }

        #pragma omp parallel shared(new_a, wavelet_coeffs, a_length) private(i)
        {
            #pragma omp for
            for (int j = 0; j < new_a_length; j++) {
                wavelet_coeffs[j] = new_a[j];
            }
        }

        a_length = new_a_length;
        free(new_a);
    }

}

void omp_haar_wavelet_transform_2d(data_t *data, int width, int height) {
    // Row-wise transformation
    for (int i = 0; i < height; i++) {
        omp_haar_wavelet_transform_1d(data + i * width, width);
    }

    // Column-wise transformation
    data_t *column = (data_t *)malloc(height * sizeof(data_t));
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
            column[i] = data[i * width + j];
        }
        omp_haar_wavelet_transform_1d(column, height);
        for (int i = 0; i < height; i++) {
            data[i * width + j] = column[i];
        }
    }

    free(column);
}

void omp_inverse_haar_wavelet_transform_2d(data_t *data, int width, int height) {
    // Inverse column-wise transformation
    data_t *column = (data_t *)malloc(height * sizeof(data_t));
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
            column[i] = data[i * width + j];
        }
        omp_inverse_haar_wavelet_transform_1d(column, height);
        for (int i = 0; i < height; i++) {
            data[i * width + j] = column[i];
        }
    }

    free(column);

    // Inverse row-wise transformation
    for (int i = 0; i < height; i++) {
        omp_inverse_haar_wavelet_transform_1d(data + i * width, width);
    }
}


