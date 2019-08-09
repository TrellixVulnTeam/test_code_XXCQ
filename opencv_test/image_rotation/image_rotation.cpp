#include <iostream>
#include <string>
#include <vector>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

void rotateImageWithoutCropping(
    cv::Mat& in_img, cv::Mat& out_img, const cv::Point2f& center, const double angle_in_degree)
{
    int width = in_img.cols, height = in_img.rows;

    // Positive angle is counter-clockwise
    cv::Mat rot = cv::getRotationMatrix2D(center, angle_in_degree, 1.0);  // last parameter is scale factor

    // Determine bounding box of the rotated image. This is to ensure the rotated image is not cropped.
    std::vector<cv::Point2d> corners({{0, 0}, {0, static_cast<double>(height - 1)}, {static_cast<double>(width - 1), 0},
        {static_cast<double>(width - 1), static_cast<double>(height - 1)}});
    cv::Mat corner_mat(corners);
    cv::transform(corner_mat, corner_mat, rot);  // get rotated corners
    double minx = corner_mat.at<cv::Point2d>(0, 0).x, miny = corner_mat.at<cv::Point2d>(0, 0).y;
    double maxx = minx, maxy = miny;
    for (int i = 1; i < 4; ++i)
    {
        minx = std::min(minx, corner_mat.at<cv::Point2d>(i, 0).x);
        miny = std::min(miny, corner_mat.at<cv::Point2d>(i, 0).y);
        maxx = std::max(maxx, corner_mat.at<cv::Point2d>(i, 0).x);
        maxy = std::max(maxy, corner_mat.at<cv::Point2d>(i, 0).y);
    }
    rot.at<double>(0, 2) += -minx;  // add a translation on the transformed image to avoid crop
    rot.at<double>(1, 2) += -miny;

    // Transform the image
    cv::Size out_size(ceil(maxx - minx + 1), ceil(maxy - miny + 1));
    cv::warpAffine(in_img, out_img, rot, out_size);

    printf("Original image size: (%d, %d)\n", width, height);
    printf("Transformed image size: (%d, %d)\n", out_img.cols, out_img.rows);
}

int main(int argc, char const* argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: image_rotation input_image output_image" << std::endl;
        return -1;
    }

    return 0;
}
