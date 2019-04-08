#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

enum KernelType {LAPLACIAN = 0, GAUSSIAN_FILTER, BOX_BLUR, ERODE, NUM_KERNEL_TYPES};

void testVectorAddition();

int printGPUInfo();

void convertBGRA2GrayGPU(cv::Mat& src_rgb, cv::Mat& dst_grey);

void imageConvolution(cv::Mat &src_rgb, cv::Mat &dst_rgb, KernelType kernel_type, bool use_2Dtex = false);


#endif
