
#include <iostream>
#include <string>
#include <vector>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

int main(int argc, char const *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: 3d_mat input_image" << std::endl;
        return -1;
    }

    std::string image_name(argv[1]);
    cv::Mat src = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
    if (!src.data)
    {
        std::cout << "Cannot read image " << image_name << "!" << std::endl;
        return -1;
    }

    src.convertTo(src, CV_32F);
    cv::Mat1f dev;  // 1 channel float mat
    int n = 10, m = 2;
    int h = src.rows, w = src.cols;
    const int dims[4] = {n, m, h, w};
    dev.create(4, dims);  // 3D mat with n images, each with size h x w. Seems it has default value 0.
    for (int i = 0; i < dev.dims; ++i)
    {
        // '.step' stores number of bytes for each element in its dimension. Here
        // dev.step[0] saves h * w * sizeof(float) * 1 (#channels = 1 here),
        // dev.step[1] saves w * sizeof(float) * 1, and step[2] is sizeof(float) * 1.
        std::cout << "Dim" << i << ": " << dev.size[i] << ", Step" << i << ": " << dev.step[i] << std::endl;

        // '.step.p' is exactly equal to '.step'.
        // std::cout << dev.step.p[0] << " " << dev.step.p[1] << " " << dev.step.p[2] << std::endl;
    }

    for (int i = 0; i < n; ++i)
    {
        // Try to copy some data into the high dimensional matrix
        // memcpy(dev.ptr<float>(i), (float *)src_new.data, h * w * sizeof(float));
        // cv::Mat mat(h, w, CV_32FC1, dev.ptr<float>(i));
        // mat.convertTo(mat, CV_8U);
        // cv::imwrite("mat" + std::to_string(i) + ".png", mat);

        // Seems a cv::Mat created from a pointer is only a reference, so copying
        // data to the matrix will also put the data into the pointer address, which
        // can be used to transfering data in high dimensional matrix easily.
        for (size_t j = 0; j < m; j++)
        {
            cv::Mat src_new = src + i * 10 + j * 20;
            float* dev_ptr = dev.ptr<float>(i, j);
            cv::Mat mat(h, w, CV_32FC1, dev_ptr);
            src_new.copyTo(mat, src < 200);  // copy with a mask for testing
            cv::Mat mat_dev(h, w, CV_32FC1);
            memcpy(mat_dev.data, dev.ptr<float>(i, j), h * w * sizeof(float));
            mat_dev.convertTo(mat_dev, CV_8U);
            cv::imwrite("mat" + std::to_string(i) + "_" + std::to_string(j) + ".png", mat_dev);
        }
    }
    return 0;
}
