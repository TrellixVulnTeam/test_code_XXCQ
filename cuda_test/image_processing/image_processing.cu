#include <iostream>
#include <memory>
#include <math.h>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "simple.h"
#include "image_processing.h"

float gpu_time;  // Recond gpu running time
cudaEvent_t start_time, stop_time;
const int kBlockSize = 32;

texture<uchar4, 2, cudaReadModeNormalizedFloat> rgb_tex;  // 2D GPU texture

/// Image convolution kernels
__constant__ float tex_kernel[9];  // kernel in constant memory

//! Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
    // printf("Index: %d, Stride: %d\n", index, stride);
}

//! 1D vector addition
void testVectorAddition()
{
    int N = 1 << 20;
    float *x, *y;
    std::cout << "#Numbers: " << N << std::endl;

    // Measure GPU running time
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaEventRecord(start_time, 0);

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;  // round up in case N is not a multiple of blockSize
    add<<<numBlocks, blockSize>>>(N, x, y);

    cudaEventRecord(stop_time, 0);
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

    cudaEventElapsedTime(&gpu_time, start_time, stop_time);
    printf("Time for the kernel: %f ms\n", gpu_time);
}

//! Print GPU info
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
        printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  MaxThreadsPerBlock: %d\n\n", prop.maxThreadsPerBlock);
    }
    return device_num;
}

//! Kernel function: Convert RGB (src) data to Gray (dst).
/**
 * Input:
 * @param src input array with size width * height * 4 in BGRA order
 * @param dst output array with size width * height
 */
__global__ void d_bgra2Gray(const unsigned char *src, int width, int height, unsigned char *dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)  // Note to add this condition
    {
        dst[y * width + x] = (unsigned char)(float(src[y * width * 4 + x * 4]) * 0.0722 +
                                             float(src[y * width * 4 + x * 4 + 1]) * 0.7152 +
                                             float(src[y * width * 4 + x * 4 + 2]) * 0.2126);
    }
}

//! Convert BGRA image to grayscale using GPU
void convertBGRA2GrayGPU(cv::Mat &src_bgra, cv::Mat &dst_grey)
{
    // Measure GPU running time
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    int width = src_bgra.cols, height = src_bgra.rows;
    unsigned char *dev_src{nullptr}, *dev_dst{nullptr};
    cudaMalloc(&dev_src, width * height * 4 * sizeof(unsigned char));  // 4 channels BGRA
    cudaMalloc(&dev_dst, width * height * sizeof(unsigned char));      // grayscale image has only 1 channel

    cudaMemcpy(dev_src, src_bgra.data, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(dev_dst, 0, width * height * sizeof(unsigned char));

    cudaEventRecord(start_time, 0);

    dim3 block_dim(kBlockSize, kBlockSize);
    dim3 grid_dim((width + kBlockSize - 1) / kBlockSize, (height + kBlockSize - 1) / kBlockSize);
    d_bgra2Gray<<<grid_dim, block_dim>>>(dev_src, width, height, dev_dst);

    cudaEventRecord(stop_time, 0);

    cudaDeviceSynchronize();

    unsigned char *dst_ptr = new unsigned char[width * height];
    cudaMemcpy(dst_ptr, dev_dst, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    dst_grey = cv::Mat(height, width, CV_8UC1, dst_ptr).clone();  // If not clone, dst_grey is only a reference to mat
    delete dst_ptr;

    cudaFree(dev_src);
    cudaFree(dev_dst);

    cudaEventElapsedTime(&gpu_time, start_time, stop_time);
    printf("Time for the kernel: %f ms\n", gpu_time);
}

//! Kernel function: Image convolution.
/**
 * Input:
 * @param src input image, 1D array with size width * height * 4 in BGRA order
 * @param conv_kernel input kernel, 1D array with size 'kernel_size'
 * @param dst output image, 1D array with size width * height * 3
 */
__global__ void d_imageConvolution(
    const unsigned char *src, int width, int height, const float *conv_kernel, int kernel_width, unsigned char *dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)  // Note to add this condition
    {
        int pad = kernel_width / 2;
        float rgb[3] = {0, 0, 0};
        int idx = -1;
        // Add 0s for all borders.
        for (int j = y - pad; j <= y + pad; j++)
        {
            for (int i = x - pad; i <= x + pad; ++i)
            {
                idx++;
                if (i < 0 || i > width - 1 || j < 0 || j > height - 1)
                    continue;
                for (int k = 0; k < 3; ++k)
                    rgb[k] += float(src[j * width * 4 + i * 4 + k]) * conv_kernel[idx];
            }
        }
        for (int k = 0; k < 3; ++k)
            dst[y * width * 4 + x * 4 + k] = (unsigned char)(max(min(rgb[k], 255.0), 0.0));
        dst[y * width * 4 + x * 4 + 3] = src[y * width * 4 + x * 4 + 3];  // simply copy the alpha channel
    }
}

//! Kernel function: Image convolution using 2D GPU texture.
/**
 * Input:
 * @param src input image, 1D array with size width * height * 3 in BGR order
 * @param conv_kernel input kernel, 1D array with size 'kernel_size'
 * @param dst output image, 1D array with size width * height * 3
 */
__global__ void d_imageConvolution2DTex(
    int width, int height, int pitch, const float *conv_kernel, int kernel_width, unsigned char *dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)  // Note to add this condition
    {
        int pad = kernel_width / 2;
        float bgr[3] = {0, 0, 0};
        float4 src_bgra;
        int idx = -1;
        // Add 0s for all borders.
        for (int j = y - pad; j <= y + pad; j++)
        {
            for (int i = x - pad; i <= x + pad; ++i)
            {
                idx++;
                if (i < 0 || i > width - 1 || j < 0 || j > height - 1)
                    continue;

                src_bgra = tex2D(rgb_tex, i, j);
                bgr[0] += src_bgra.x * conv_kernel[idx];
                bgr[1] += src_bgra.y * conv_kernel[idx];
                bgr[2] += src_bgra.z * conv_kernel[idx];
            }
        }
        for (int i = 0; i < 3; ++i)
            dst[y * pitch + x * 4 + i] = uchar(bgr[i] * 0xff); // Note to use pitch (bytes of width in the gpu array)
        src_bgra = tex2D(rgb_tex, x, y);
        dst[y * pitch + x * 4 + 3] = uchar(src_bgra.w * 0xff);  // simply copy the alpha channel
    }
}

//! 2D Gaussian with mean 0 and standard deviation 1
float computeGaussian(float x, float y)
{
    return exp((-x * x - y * y) / 2) / (M_PI * 2.0);
}

//! Compute gaussian kernel matrix
void computeGaussianKernel(float *kernel, int kernel_width)
{
    if (kernel_width <= 0)
        return;
    float sum = 0;
    int pad = kernel_width / 2;
    for (int i = -pad; i <= pad; ++i)
    {
        for (int j = -pad; j <= pad; j++)
        {
            float val = computeGaussian(i, j);
            kernel[(j + pad) * kernel_width + (i + pad)] = val;
            sum += val;
        }
    }
    assert(sum != 0);
    for (int i = 0; i < kernel_width * kernel_width; ++i)
        kernel[i] /= sum;
}

//! Test image convolution on a kernel
/**
 * Input:
 * @param src_bgra input image, 2D array with 4 channels in BGRA order
 * @param dst_rgb output image, 2D array with same size and type as src_bgra
 */
void imageConvolution(cv::Mat &src_bgra, cv::Mat &dst_rgb, KernelType kernel_type, bool use_2Dtex)
{
    cudaEventCreate(&start_time);  // Measure GPU running time
    cudaEventCreate(&stop_time);

    /// Initialize GPU array
    int width = src_bgra.cols, height = src_bgra.rows;
    unsigned char *dev_src{nullptr}, *dev_dst{nullptr};
    const int kImageDataSize = width * height * 4 * sizeof(uchar);
    size_t pitch;
    const int kSourceImagePitch = sizeof(uchar) * width * 4;  // bytes of width
    if (use_2Dtex)
    {
        // Here destination image is of the same type and size as input image, so
        // their pitches are exactly the same too, which is equal to the size of width in bytes.
        HANDLE_ERROR(cudaMallocPitch(&dev_src, &pitch, kSourceImagePitch, height));
        HANDLE_ERROR(cudaMallocPitch(&dev_dst, &pitch, kSourceImagePitch, height));

        cudaMemcpy2D(
            dev_src, pitch, src_bgra.data, kSourceImagePitch, kSourceImagePitch, height, cudaMemcpyHostToDevice);
        cudaMemset2D(dev_dst, pitch, 0, kSourceImagePitch, height);

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        HANDLE_ERROR(cudaBindTexture2D(0, rgb_tex, dev_src, desc, width, height, pitch));
    }
    else
    {
        cudaMalloc(&dev_src, kImageDataSize);
        cudaMalloc(&dev_dst, kImageDataSize);

        cudaMemcpy(dev_src, src_bgra.data, kImageDataSize, cudaMemcpyHostToDevice);
        cudaMemset(dev_dst, 0, kImageDataSize);

        dim3 block_dim(kBlockSize, kBlockSize);
        dim3 grid_dim((width + kBlockSize - 1) / kBlockSize, (height + kBlockSize - 1) / kBlockSize);
    }

    /// Create GPU convolution kernel
    const int kKernelWidth = 7;  // NOTE that the kernel width must be odd
    const int kKernelSize = kKernelWidth * kKernelWidth;
    float *gpu_kernel{nullptr};
    float kernel[kKernelSize] = {0};
    if (kernel_type == LAPLACIAN)
    {
        // Use a fixed 3x3 Laplacian kernel
        const float laplacian_kernel[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
        memcpy(kernel, laplacian_kernel, kKernelSize * sizeof(float));
    }
    else if (kernel_type == GAUSSIAN_FILTER)
    {
        computeGaussianKernel(kernel, kKernelWidth);
    }
    else if (kernel_type == BOX_BLUR)
    {
        for (int i = 0; i < kKernelSize; ++i)
            kernel[i] = 1.0 / kKernelSize;
    }
    cudaMalloc(&gpu_kernel, kKernelSize * sizeof(float));
    cudaMemcpy(gpu_kernel, kernel, kKernelSize * sizeof(float), cudaMemcpyHostToDevice);

    /// Start to call kernel function
    cudaEventRecord(start_time, 0);
    dim3 block_dim(kBlockSize, kBlockSize);
    dim3 grid_dim((width + kBlockSize - 1) / kBlockSize, (height + kBlockSize - 1) / kBlockSize);
    if (use_2Dtex)
    {
        d_imageConvolution2DTex<<<grid_dim, block_dim>>>(width, height, pitch, gpu_kernel, kKernelWidth, dev_dst);
    }
    else
    {
        d_imageConvolution<<<grid_dim, block_dim>>>(dev_src, width, height, gpu_kernel, kKernelWidth, dev_dst);
    }
    cudaEventRecord(stop_time, 0);

    // Use cudaEventSynchronize() to replace cudaDeviceSynchronize(), since the former seems
    // more light-weight and doesn't stall the GPU process.
    // Ref: https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
    // cudaDeviceSynchronize();
    cudaEventSynchronize(stop_time);

    /// Copy result from GPU to CPU
    unsigned char *dst_ptr = new unsigned char[kImageDataSize];
    if (use_2Dtex)
    {
        cudaMemcpy2D(dst_ptr, kSourceImagePitch, dev_dst, pitch, kSourceImagePitch, height, cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMemcpy(dst_ptr, dev_dst, kImageDataSize, cudaMemcpyDeviceToHost);
    }

    dst_rgb = cv::Mat(height, width, CV_8UC4, dst_ptr).clone();  // If not clone, dst is only be a reference
    delete dst_ptr;

    cudaFree(dev_src);
    cudaFree(dev_dst);
    cudaFree(gpu_kernel);
    cudaUnbindTexture(rgb_tex);

    cudaEventElapsedTime(&gpu_time, start_time, stop_time);
    printf("Time for the kernel: %f ms\n", gpu_time);
}
