#include "rgbd_processor.h"
#include <fstream>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "print_in_color.h"

RGBDProcessor::RGBDProcessor(const std::string& rgbd_folder)
{
    rgbd_folder_ = rgbd_folder;
    if (rgbd_folder.back() != '/')
        rgbd_folder_ += '/';
    color_width_ = color_height_ = depth_width_ = depth_height_ = -1;
}

RGBDProcessor::~RGBDProcessor() {}

bool RGBDProcessor::readColorImage(int frame_idx)
{
    // BundleFusion RGB-D data 文件名格式是固定的，例如，color image 的名字都是 frame-000001.color.jpg
    std::string str_frame_idx = std::to_string(frame_idx);
    std::string frame_fname = rgbd_folder_ + "frame-" + std::string(6 - str_frame_idx.length(), '0') + str_frame_idx + ".color.jpg";
    color_image_ = cv::imread(frame_fname, cv::IMREAD_COLOR);
    if (!color_image_.data)
    {
        PRINT_RED("ERROR: Cannot read color image %s", frame_fname.c_str());
        return false;
    }
    return true;
}

bool RGBDProcessor::readDepthImage(int frame_idx)
{
    // BundleFusion RGB-D data 文件名格式是固定的，例如，depth image 的名字都是 frame-000001.depth.png
    std::string str_frame_idx = std::to_string(frame_idx);
    std::string frame_fname = rgbd_folder_ + "frame-" + std::string(6 - str_frame_idx.length(), '0') + str_frame_idx + ".depth.png";
    depth_image_ = cv::imread(frame_fname, cv::IMREAD_UNCHANGED);
    if (!depth_image_.data)
    {
        PRINT_RED("ERROR: Cannot depth color image %s", frame_fname.c_str());
        return false;
    }
    return true;
}


bool RGBDProcessor::readCameraInfoFile()
{
    // BundleFusion RGB-D data 文件名格式是固定的，例如相机参数都在 info.txt 文件中
    std::string filename = rgbd_folder_ + "info.txt";
    std::ifstream readin(filename, std::ios::in);
    if (readin.fail() || readin.eof())
    {
        PRINT_RED("ERROR: Cannot read camera parameter file %s", filename.c_str());
        return false;
    }
    std::string str_line;
    std::string str_dummy;
    std::string str;
    float dummy;
    while (!readin.eof() && !readin.fail())
    {
        std::getline(readin, str_line);
        if (readin.eof())
            break;
        std::istringstream iss(str_line);
        iss >> str;
        if (str == "m_colorWidth")
            iss >> str_dummy >> color_width_;
        else if (str == "m_colorHeight")
            iss >> str_dummy >> color_height_;
        else if (str == "m_depthWidth")
            iss >> str_dummy >> depth_width_;
        else if (str == "m_depthHeight")
            iss >> str_dummy >> depth_height_;
        else if (str == "m_calibrationColorIntrinsic")
        {
            iss >> str_dummy >> color_intrinsics_.fx >> dummy >> color_intrinsics_.cx >> dummy >> dummy >>
                color_intrinsics_.fy >> color_intrinsics_.cy;
        }
        else if (str == "m_calibrationDepthIntrinsic")
        {
            iss >> str_dummy >> depth_intrinsics_.fx >> dummy >> depth_intrinsics_.cx >> dummy >> dummy >>
                depth_intrinsics_.fy >> depth_intrinsics_.cy;
        }
    }
    readin.close();
    PRINT_GREEN("Color intrinsics: %f, %f, %f, %f", color_intrinsics_.fx, color_intrinsics_.fy, color_intrinsics_.cx, color_intrinsics_.cy);
    PRINT_GREEN("Depth intrinsics: %f, %f, %f, %f", depth_intrinsics_.fx, depth_intrinsics_.fy, depth_intrinsics_.cx, depth_intrinsics_.cy);
    return true;
}
