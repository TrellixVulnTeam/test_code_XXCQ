
#include <iostream>
#include <string>
#include <vector>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

//! Convert the type of a cv::Mat object to a readable string
/*
* Correspondence between type number and string is in this table:
*
+--------+----+----+----+----+------+------+------+------+
|        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
+--------+----+----+----+----+------+------+------+------+
| CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
| CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
| CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
| CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
| CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
| CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
| CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
+--------+----+----+----+----+------+------+------+------+
*/
std::string cvMatType2Str(int type)
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
        case CV_8U:
            r = "8U";
            break;
        case CV_8S:
            r = "8S";
            break;
        case CV_16U:
            r = "16U";
            break;
        case CV_16S:
            r = "16S";
            break;
        case CV_32S:
            r = "32S";
            break;
        case CV_32F:
            r = "32F";
            break;
        case CV_64F:
            r = "64F";
            break;
        default:
            r = "User";
            break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

cv::Mat convertUshortToGrey(const cv::Mat& image)
{
    double min_val, max_val;
    cv::minMaxLoc(image, &min_val, &max_val);

    float scale = 255 / (max_val - min_val);
    float offset = -min_val * scale;

    cv::Mat image_out;
    cv::convertScaleAbs(image, image_out, scale, offset);
    return image_out;
}

int main(int argc, char const* argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: image_basic <input_image>" << std::endl;
        return -1;
    }
    std::string image_name(argv[1]);
    cv::Mat src = cv::imread(image_name, cv::IMREAD_UNCHANGED);
    if (!src.data)
    {
        std::cout << "Cannot read image " << image_name << "!" << std::endl;
        return -1;
    }
    std::cout << "Image Type:" << cvMatType2Str(src.type()) << std::endl;

    cv::Mat gray = convertUshortToGrey(src);
    std::string outputFile(argv[1]);
    cv::imwrite(outputFile, gray);

    //----------------------------------------------------
    // Test cv::Rect
    // cv::Rect rect1(50, 30, 200, 200);
    // cv::Mat temp = gray(rect1); // Now temp is a reference/pointer to the image rectangle
    // temp.setTo(cv::Scalar(255)); // changing reference image also changes the original image 'gray'

    // // Test mask and common operators
    // cv::Mat mask = (gray < 100) | (gray > 200);  // mask matrix supports and (&), or (|) operations
    // cv::Mat mask1 = gray < 100;
    // cv::Mat mask2 = gray > 200;
    // std::cout << "Mask type: " << cvMatType2Str(mask.type())
    //           << std::endl;  // Mask matrix is 8UC1 type (i.e., cv::Mat1b)
    // gray.setTo(0, 1 - (mask1 | mask2));

    // // cv::Mat out = gray - 0.8 * gray; // matrix also support + and -

    // std::string window_name("Test");
    // cv::namedWindow(window_name);
    // cv::imshow(window_name, gray);
    // cv::waitKey();

    //----------------------------------------------------

    //----------------------------------------------------
    // Test pyrDown
    // cv::Mat img_down;
    // cv::pyrDown(src, img_down, cv::Size(), cv::BORDER_REPLICATE);

    // std::string window_name("Test");
    // cv::namedWindow(window_name);
    // cv::imshow(window_name, img_down);
    // cv::waitKey();
    //----------------------------------------------------

    //----------------------------------------------------
    // // Test copying part of a matrix to another
    // cv::Rect_<float> rect2(0.3, 1.3, 400, 200), rect3(0.8, 1.8, 400, 200);
    // cv::Rect rect1(0, 1, 400, 200), rect4(1, 2, 400, 200);
    // cv::Mat res1, res2, res3, res4;
    // src(rect1).copyTo(res1);
    // src(rect2).copyTo(res2);
    // src(rect3).copyTo(res3);
    // src(rect4).copyTo(res4);
    //
    // cv::imwrite("test1.png", res1);
    // cv::imwrite("test2.png", res2);
    // cv::imwrite("test3.png", res3);
    // cv::imwrite("test4.png", res4);
    //----------------------------------------------------

    // //----------------------------------------------------
    // // Add a border on the image
    // if (src.channels() == 3)
    //     cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    // const int border = 10;
    // const int border_type = cv::BORDER_CONSTANT | cv::BORDER_ISOLATED;
    // const cv::Scalar value(255);
    // cv::Mat dev;
    // // Pre-select the part inside the border to ensure the border is added without padding,
    // // since cv::copyMakeBorder() will extrapolate the padding by default. If not, the
    // // border will be added as padding and this will increase the image size.
    // cv::Rect roi(border, border, src.cols - border * 2, src.rows - border * 2);
    // cv::copyMakeBorder(src(roi), dev, border, border, border, border, border_type, value);
    // cv::imwrite("border.png", dev);
    // //----------------------------------------------------

    return 0;
}


