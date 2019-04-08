#include <iostream>
#include <string>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "image_processing.h"

int main(int argc, char const* argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: program <input_image>" << std::endl;
        return -1;
    }
    std::string filename(argv[1]);
    cv::Mat src = cv::imread(filename, cv::IMREAD_COLOR);
    if (!src.data)
    {
        std::cout << "ERROR: Cannot read image " << filename << std::endl;
        return -1;
    }
    if (printGPUInfo() < 1)
    {
        std::cout << "ERROR: No GPU device!" << std::endl;
        return -1;
    }
    // Convert the image to 4 channels BGRA to be smoothly used in GPU later
    cv::cvtColor(src, src, cv::COLOR_BGR2BGRA);
    cv::Mat dst = src;

    // convertBGRA2GrayGPU(src, dst);

    std::string src_wname("src"), dst_wname("Dst");
    cv::namedWindow(src_wname);
    cv::imshow(src_wname, src);
    cv::namedWindow(dst_wname);
    cv::imshow(dst_wname, dst);
    // cv::waitKey();

    bool use_2Dtex = false;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 27)  // 'ESC'
        {
            printf("Quit program.\n");
            break;
        }
        else if (key == 98)  // 'b'
        {
            use_2Dtex = !use_2Dtex;
            if (use_2Dtex)
                printf("Use 2D texture Memory\n");
            else
                printf("Use 1D Memory\n");
        }
        else
        {
            if (key >= 48 && key <= 57)  // '0' to '9'
            {
                if (key == 48) // '0'
                    printf("Run Laplacian Filter.\n");
                else if (key == 49) // '1'
                    printf("Run Gaussian Filter.\n");
                else if (key == 50) // '2'
                    printf("Run Box Blur.\n");
                else
                    printf("Run Image Erode.\n");
                KernelType kernel_type = KernelType(std::min(key - 48, 47 + NUM_KERNEL_TYPES));
                imageConvolution(src, dst, kernel_type, use_2Dtex);
                cv::imshow(dst_wname, dst);
            }
        }
    }
    return 0;
}
