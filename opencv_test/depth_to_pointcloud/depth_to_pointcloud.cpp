
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

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
std::string getCvMatType(int type)
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
    std::cout << "Image Type:" << getCvMatType(src.type()) << std::endl;

    const double fx = 326.0, fy = 326.0, cx = 160.0, cy = 120.0;
    // const double fx =  520.9, fy = 521.0, cx = 325.1, cy = 249.7;

    const double kMaxDepth = 4000;
    const double kMinDepth = 200;
    std::string xyz_fname = image_name.substr(0, image_name.length() - 4) + ".xyz";
    std::ofstream writeout(xyz_fname, std::ios::trunc);
    for (int i = 0; i < src.rows; ++i)
    {
        // unsigned char* p = src.ptr<cv::Vec3b>(i); // 先获取每一行开头的指针
        for (int j = 0; j < src.cols; ++j)
        {
            int b = int(src.at<cv::Vec3b>(i, j)[0]);
            int g = int(src.at<cv::Vec3b>(i, j)[1]);
            int r = int(src.at<cv::Vec3b>(i, j)[2]);
            // g = 2;
            int depth = ((r << 16) | (g << 8)) | b;
            // double depth = double(src.at<unsigned short>(i, j)) / 5;
            if (depth > kMaxDepth || depth < kMinDepth)
                continue;
            // std::cout << i << "," << j << ": " << b << " " << g << " " << r << " ... " << depth << std::endl;
            double z = double(depth) / 1000;
            double x = (i - cx) * z / fx;
            double y = (j - cy) * z / fy;
            writeout << x << " " << y << " " << z << std::endl;
        }
    }
    writeout.close();

    return 0;
}