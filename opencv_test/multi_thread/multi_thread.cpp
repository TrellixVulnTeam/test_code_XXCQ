#include <iostream>
#include <string>
#include <vector>
#include <thread>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

void threadFunc(cv::Mat& image, float enhanced_factor, const std::string& outfile)
{
    cv::Mat enhanced_image;
    image.convertTo(enhanced_image, -1, enhanced_factor, 0);
    cv::imwrite(outfile, enhanced_image);

    std::cout << "Thread is done" << std::endl;
}

// void testMultiThread(cv::Mat& image)
// {
//     std::thread t1(threadFunc, image);
// }

int main(int argc, char const* argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: multi_thread input_image" << std::endl;
        return -1;
    }
    std::string image_name(argv[1]);
    cv::Mat image = cv::imread(image_name, cv::IMREAD_COLOR);
    if (!image.data)
    {
        std::cout << "Cannot read image " << image_name << "!" << std::endl;
        return -1;
    }

    float enhanced_factor = 1.3;
    std::string outfile = "out1.png";
    std::thread t1(threadFunc, std::ref(image), enhanced_factor, outfile);

    std::cout << "Continue on main function. " << std::endl;
    int count = 0;
    for (int i = 0; i < 1e9; ++i)
    {
        count++;
    }

    enhanced_factor = 2.5;
    cv::Mat enhanced_image;
    image.convertTo(enhanced_image, -1, enhanced_factor, 0);
    outfile = "out2.png";
    cv::imwrite(outfile, enhanced_image);
    std::cout << "Main func is done." << std::endl;

    t1.join();

    return 0;
}
