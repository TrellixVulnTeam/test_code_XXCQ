#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Eigen>
#include "opencv2/core.hpp"

class RGBDProcessor
{
public:
    struct Intrinsics
    {
        double fx, fy, cx, cy;
    };
    Intrinsics depth_intrinsics_;
    Intrinsics color_intrinsics_;
    std::string rgbd_folder_;
    cv::Mat color_image_;
    cv::Mat depth_image_;
    int color_width_;
    int color_height_;
    int depth_width_;
    int depth_height_;

public:
    RGBDProcessor(const std::string& rgbd_folder);
    ~RGBDProcessor();

    bool readCameraInfoFile();
    bool readColorImage(int frame_idx);
    bool readDepthImage(int frame_idx);
};
