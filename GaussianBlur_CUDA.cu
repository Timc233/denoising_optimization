#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>

/*
   Minimal CUDA program, intended just to test ability
   to compile and run a CUDA program

     nvcc GaussianBlur_CUDA.cu -o GaussianBlur_CUDA

   You need to follow instructions provided elsewhere, such as in the
   "CUDA_and-SCC-for-EC527,pdf" file, to setup your environment where you can
   compile and run this.

   To understand the program, of course you should read the lecture notes
   (slides) that have "GPU" in the name.
*/

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include "../dependency/stb/stb_image.h"
#include "../dependency/stb/stb_image_write.h"
// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}



__global__ void gaussian_blur(const unsigned char* input, unsigned char* output, int width, int height, int channels, float* kernel, int kernelSize)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;
            for (int j = -kernelSize/2; j <= kernelSize/2; j++)
            {
                for (int i = -kernelSize/2; i <= kernelSize/2; i++)
                {
                    int x = min(max(col + i, 0), width - 1);
                    int y = min(max(row + j, 0), height - 1);

                    float value = static_cast<float>(input[(y * width + x) * channels + c]);
                    float weight = kernel[(j + kernelSize/2) * kernelSize + (i + kernelSize/2)];
                    sum += value * weight;
                }
            }
            output[(row * width + col) * channels + c] = static_cast<unsigned char>(sum);
        }
    }
}


int main(int argc, char **argv){

  int width, height, channels;
  // get width and height of the picture
  unsigned char* image = stbi_load("noise.jpg", &width, &height, &channels, 0);
  if (!image) {
        fprintf(stderr, "Failed to load image.\n");
        exit(1);
  }
  // GPU Timing variables
  cudaEvent_t start, stop;
  float elapsed_gpu;

  // blurring level
  float sigma = 1.0;
  //create a conv kernel
  int kernel_size = 21;
  float *kernel = (float*) malloc(sizeof(float)* kernel_size * kernel_size);

  int half = kernel_size / 2;
  float sum = 0.0;
  //setup the kernel
  for (int i =-half; i<= half;++i){
    for (int j =-half; j<=half;++j){
      char value = exp(-(i*i+j*j)/(2* sigma * sigma))/(2*M_PI*sigma*sigma);
      kernel[(i+ half) * kernel_size + j+ half ] = value;
      sum+= value;
    }
  }
  // normalize the kernel
  for(int i =0;i< kernel_size*kernel_size; ++i){
    kernel[i]/=sum;
  }

  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  unsigned char *d_in, *d_out;
  float *d_kernel;

  // Allocate GPU memory
  size_t size = sizeof(float)*width*height*channels;
  unsigned char *out = (unsigned char*)malloc(size);
  CUDA_SAFE_CALL(cudaMalloc(&d_in, size));
  CUDA_SAFE_CALL(cudaMalloc(&d_out, size));
  CUDA_SAFE_CALL(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));

#if PRINT_TIME
  // Create the cuda events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record event on the default stream
  cudaEventRecord(start, 0);
#endif

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_in, image, size, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_kernel, kernel, sizeof(float)*kernel_size*kernel_size, cudaMemcpyHostToDevice));

  // setup cuda block and grid
  dim3 block_size(16,16);
  dim3 grid_size((width+block_size.x-1)/block_size.x,(height + block_size.y -1)/block_size.y);
  

  
  // Launch the kernel
  gaussian_blur<<<grid_size, block_size>>>(d_in, d_out, width, height, channels, d_kernel, kernel_size);

  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(out, d_out, size , cudaMemcpyDeviceToHost));
  
  
#if PRINT_TIME
  // Stop and destroy the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu, start, stop);
  printf("\nGPU time: %f (msec)\n", elapsed_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#endif

  stbi_write_jpg("output.jpg", width, height, channels, out, 100);
  stbi_image_free(image);

 

  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_in));
  CUDA_SAFE_CALL(cudaFree(d_out));
  CUDA_SAFE_CALL(cudaFree(d_kernel));

  free(out);
  
  free(kernel);

  return 0;
}


