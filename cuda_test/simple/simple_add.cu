#include <iostream>
#include <math.h>
#include "simple.h"

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
    // printf("Index: %d, Stride: %d\n", index, stride);
}

int printGPUInfo()
{
    int device_num = 0;
    HANDLE_ERROR(cudaGetDeviceCount(&device_num));
    for (int i = 0; i < device_num; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf(
            "  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  MaxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    }
    return device_num;
}

int main(void)
{
    if (printGPUInfo() < 1)
    {
        std::cout << "No GPU device!" << std::endl;
        return -1;
    }

    int N = 1 << 20;
    float *x, *y;
    std::cout << "#Numbers: " << N << std::endl;

    // Measure GPU running time
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaEventRecord(start, 0);

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;  // round up in case N is not a multiple of blockSize
    add<<<numBlocks, blockSize>>>(N, x, y);

    cudaEventRecord(stop, 0);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the kernel: %f ms\n", time);

    return 0;
}
