// nvcc test_cuda_haar_wavelet.cu -o test_cuda_haar_wavelet
// nvcc test_cuda_haar_wavelet.cu -o test_cuda_haar_wavelet

#include <stdio.h>
#include <math.h>
#include "../util/set_data_type.h"
#include "cuda_haar_wavelet.h"



__global__ void haar_wavelet_transform_1d_kernel(data_t *data, data_t *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data_t A = (output[i * 2] + output[i * 2 + 1]) / sqrtf(2);
        data_t D = (output[i * 2] - output[i * 2 + 1]) / sqrtf(2);
        output[i] = A;
        output[n + i] = D;
    }
}

void cuda_haar_wavelet_transform_1d(data_t *data, int len) {
    data_t *output;
    cudaMallocManaged(&output, len * sizeof(data_t));
    cudaMemcpy(output, data, len * sizeof(data_t), cudaMemcpyHostToDevice);

    int level = log2(len);
    int iterations = level - 1;
    int n = len;

    for (int i = 0; i < iterations; i++) {
        n = n / 2;

        haar_wavelet_transform_1d_kernel<<<(n + 31) / 32, 32>>>(data, output, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(data, output, len * sizeof(data_t), cudaMemcpyDeviceToHost);

    cudaFree(output);
}

__global__ void copy_data_to_column_kernel(data_t *data, data_t *column, int width, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < blockDim.y) {
        column[i] = data[i * width + j];
    }
}

__global__ void copy_column_to_data_kernel(data_t *data, data_t *column, int width, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < blockDim.y) {
        data[i * width + j] = column[i];
    }
}

void cuda_haar_wavelet_transform_2d(data_t *data, int width, int height) {
    // Row-wise transformation
    for (int i = 0; i < height; i++) {
        cuda_haar_wavelet_transform_1d(data + i * width, width);
    }

    // Column-wise transformation
    data_t *column;
    cudaMallocManaged(&column, height * sizeof(data_t));

    for (int j = 0; j < width; j++) {
        // copy_data_to_column_kernel<<<(height + 31) / 32, 32>>>(data, column, width, height, j);
        copy_data_to_column_kernel<<<(height + 31) / 32, 32>>>(data, column, width, j);
        cudaDeviceSynchronize();

        cuda_haar_wavelet_transform_1d(column, height);

        // copy_column_to_data_kernel<<<(height + 31) / 32, 32>>>(data, column, width, height, j);
        copy_column_to_data_kernel<<<(height + 31) / 32, 32>>>(data, column, width, j);
        cudaDeviceSynchronize();
    }

    cudaFree(column);
}

__global__ void inverse_haar_wavelet_transform_1d_kernel(data_t *wavelet_coeffs, int n, int a_length) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < a_length) {
        wavelet_coeffs[2 * j] = (wavelet_coeffs[j] + wavelet_coeffs[a_length + j]) / sqrtf(2);
        wavelet_coeffs[2 * j + 1] = (wavelet_coeffs[j] - wavelet_coeffs[a_length + j]) / sqrtf(2);
    }
}

void cuda_inverse_haar_wavelet_transform_1d(data_t* wavelet_coeffs, int n) {
    int level = (int) log2(n);

    int a_length = 2;
    int iterations = level - 1;

    for (int i = 0; i < iterations; i++) {
        inverse_haar_wavelet_transform_1d_kernel<<<(a_length + 31) / 32, 32>>>(wavelet_coeffs, n, a_length);
        cudaDeviceSynchronize();

        a_length *= 2;
    }
}

void cuda_inverse_haar_wavelet_transform_2d(data_t *data, int width, int height) {
    // Inverse column-wise transformation
    data_t *column;
    cudaMallocManaged(&column, height * sizeof(data_t));

    for (int j = 0; j < width; j++) {
        // copy_data_to_column_kernel<<<(height + 31) / 32, 32>>>(data, column, width, height, j);
        copy_data_to_column_kernel<<<(height + 31) / 32, 32>>>(data, column, width, j);
        cudaDeviceSynchronize();

        cuda_inverse_haar_wavelet_transform_1d(column, height);

        // copy_column_to_data_kernel<<<(height + 31) / 32, 32>>>(data, column, width, height, j);
        copy_column_to_data_kernel<<<(height + 31) / 32, 32>>>(data, column, width, j);
        cudaDeviceSynchronize();
    }

    cudaFree(column);

    // Inverse row-wise transformation
    for (int i = 0; i < height; i++) {
        cuda_inverse_haar_wavelet_transform_1d(data + i * width, width);
    }
}



