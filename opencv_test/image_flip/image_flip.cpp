#include <iostream>
#include <string>
#include <vector>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

namespace ImageUtil {

void flipVert()
{

}

bool inPlaceFlip(cv::Mat& image, int flip_mode)
{
    if (image.dims() > 2)
        return false;
    cv::Size size = image.size();
    if (flip_mode < 0)
    {
        if (size.width == 1)
            flip_mode = 0;
        if (size.height == 1)
            flip_mode = 1;
    }
    if ((size.width == 1 && flip_mode > 0) || (size.height == 1 && flip_mode == 0) ||
        (size.height == 1 && size.width == 1 && flip_mode < 0))
    {
        // Image only has one line or one column. No need to flip it.
        return true;
    }
    int type = image.type();
    size_t esz = CV_ELEM_SIZE(type);

    if (flip_mode <= 0)
        flipVert(image.ptr(), image.step, image.ptr(), image.step, image.size(), esz);
    return true;
}

}  // namespace ImageUtil

int main(int argc, char const* argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: image_flip <input_image>" << std::endl;
        return -1;
    }
    std::string image_name(argv[1]);
    cv::Mat src = cv::imread(image_name, cv::IMREAD_ANYDEPTH);
    if (!src.data)
    {
        std::cout << "Cannot read image " << image_name << "!" << std::endl;
        return -1;
    }

    return 0;
}

