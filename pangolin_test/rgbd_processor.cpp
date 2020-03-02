#include "rgbd_processor.h"
#include <fstream>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "print_in_color.h"
#include <pangolin/pangolin.h>

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
    // BundleFusion RGB-D data 文件名格式是固定的，例如每一帧的 color image 文件名都是 frame-000001.color.jpg
    std::string str_frame_idx = std::to_string(frame_idx);
    std::string color_fname =
        rgbd_folder_ + "frame-" + std::string(6 - str_frame_idx.length(), '0') + str_frame_idx + ".color.jpg";
    color_image_ = cv::imread(color_fname, cv::IMREAD_COLOR);
    if (!color_image_.data)
    {
        PRINT_RED("ERROR: Cannot read color image %s", color_fname.c_str());
        return false;
    }
    return true;
}

bool RGBDProcessor::readDepthImage(int frame_idx)
{
    // BundleFusion RGB-D data 文件名格式是固定的，例如每一帧的 depth image 文件名都是 frame-000001.depth.png
    std::string str_frame_idx = std::to_string(frame_idx);
    std::string depth_fname =
        rgbd_folder_ + "frame-" + std::string(6 - str_frame_idx.length(), '0') + str_frame_idx + ".depth.png";
    depth_image_ = cv::imread(depth_fname, cv::IMREAD_UNCHANGED);
    if (!depth_image_.data)
    {
        PRINT_RED("ERROR: Cannot depth color image %s", depth_fname.c_str());
        return false;
    }
    return true;
}

bool RGBDProcessor::readCameraInfoFile()
{
    // BundleFusion RGB-D data 文件名格式是固定的，例如相机参数都在 info.txt 文件中
    std::string info_fname = rgbd_folder_ + "info.txt";
    std::ifstream readin(info_fname, std::ios::in);
    if (readin.fail() || readin.eof())
    {
        PRINT_RED("ERROR: Cannot read camera parameter file %s", info_fname.c_str());
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
    PRINT_GREEN("Color intrinsics: %f, %f, %f, %f", color_intrinsics_.fx, color_intrinsics_.fy, color_intrinsics_.cx,
        color_intrinsics_.cy);
    PRINT_GREEN("Depth intrinsics: %f, %f, %f, %f", depth_intrinsics_.fx, depth_intrinsics_.fy, depth_intrinsics_.cx,
        depth_intrinsics_.cy);
    return true;
}

bool RGBDProcessor::readCameraPoseFile(int frame_idx)
{
    // BundleFusion RGB-D data 文件名格式是固定的，例如每一帧的相机 pose 都在 frame-XXXXXX.pose.txt 文件中
    std::string str_frame_idx = std::to_string(frame_idx);
    std::string pose_fname =
        rgbd_folder_ + "frame-" + std::string(6 - str_frame_idx.length(), '0') + str_frame_idx + ".pose.txt";
    std::ifstream readin(pose_fname, std::ios::in);
    if (readin.fail())
    {
        PRINT_RED("ERROR: cannot open the pose file %s", pose_fname.c_str());
        return false;
    }
    // NOTE: each pose must be a 4x4 matrix and here we don't check the validity.
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            readin >> pose_camera_to_global_(i, j);
    readin.close();
    return true;
}

void RGBDProcessor::drawDepthPointCloud()
{
    glBegin(GL_POINTS);
    for (int i = 0; i < depth_image_.rows; ++i)
    {
        for (int j = 0; j < depth_image_.cols; ++j)
        {
            unsigned short depth = depth_image_.at<unsigned short>(i, j);
            if (depth)
            {
                double z = double(depth) / 1000;
                double x = (j - depth_intrinsics_.cx) * z / depth_intrinsics_.fx;
                double y = (i - depth_intrinsics_.cy) * z / depth_intrinsics_.fy;
                cv::Vec3b bgr = color_image_.at<cv::Vec3b>(i, j);
                glColor3f(float(bgr[2]) / 255, float(bgr[1]) / 255, float(bgr[0]) / 255);
                glVertex3d(x, y, z);
            }
        }
    }
    glEnd();
}
