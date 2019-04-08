/*************************************/
/* This demo file is to generate squared (texture) images with random pure colors.
*/
/*************************************/

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <cstdlib>
#include <time.h>

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: gen_pure_teximages <--random_color> prefix_string image_number image_resolution" << std::endl;
        return -1;
    }

    std::string filename(argv[1]);
    int image_num = atoi(argv[2]);
    int resolution = atoi(argv[3]);
    bool flag_random_color = (std::string(argv[1]) == "--random_color") ? true : false;
    if (flag_random_color)
    {
        filename = std::string(argv[2]);
        image_num = atoi(argv[3]);
        resolution = atoi(argv[4]);
        srand(time(0));
    }
    for (int i = 0; i <= image_num; ++i)
    {
        std::string id_str = std::to_string(i);
        std::string image_fname = filename + "_" + std::string(3 - id_str.length(), '0') + id_str + ".jpg";
        int r = rand() % 256, g = rand() % 256, b = rand() % 256;
        cv::Mat mat(resolution, resolution, CV_8UC3, cv::Scalar(r, g, b));
        cv::imwrite(image_fname, mat);
    }
    return 0;
}