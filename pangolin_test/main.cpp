#include <limits>
#include <iostream>
#include <string>

#include <pangolin/pangolin.h>
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>

#include <opencv2/imgproc.hpp>

#include "rgbd_processor.h"

void drawCameraFrustum()
{
    const float w = 0.1;
    const float h = w * 0.75;
    const float z = w * 0.6;
    glBegin(GL_LINES);
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();
}

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cout << "Usage: program bundlefusion_rgbd_folder start_frame_index end_frame_index" << std::endl;
        std::cout << "For example: program /dev/copyroom 0 1000" << std::endl;
        return -1;
    }

    std::string rgbd_folder(argv[1]);
    RGBDProcessor processor(rgbd_folder);
    if (!processor.readCameraInfoFile())
        return -1;

    const int kStartFrameIdx = atoi(argv[2]);
    const int kEndFrameIdx = atoi(argv[3]);
    const int kWindowWidth = 1280;
    const int kWindowHeight = 960;

    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Main", kWindowWidth, kWindowHeight);

    // 默认开启全屏模式
    // pangolin::ToggleFullscreen();

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // 设置 Projection Matrix 和 gluLookAt() 参数。
    // 1) pangolin::ProjectionMatrix() 的参数依次是：
    // - w, h: 图片长和宽度
    // - fu, fv: focal length fx and fy;
    // - u0, v0: 中心点在图片中的位置，通常就是 cx 和 cy，即图片正中心
    // - zNear, zFar: 即 projection near plane 和 far plane 的位置（Z 轴上），不过它们都是正值
    // 2) 第二个函数 ModelViewLookAt() 其实就是 OpenGL 中的 gluLookAt() 函数。
    // 官方文档：https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml
    // 前 3 个参数是视点/相机所在位置（eye position），接着 3 个参数是要观看的点的位置，最后一个参数是 up
    // vector（即相机正上方的方向)。这里我们使用了 RGB-D 的经典坐标系，即 +X 向右，+Y 向下，+Z 向内，相机位置在原点朝着
    // +Z 方向，因此相机正上方是 -Y 方向。
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(processor.depth_width_, processor.depth_height_, processor.depth_intrinsics_.fx,
            processor.depth_intrinsics_.fy, processor.depth_intrinsics_.cx, processor.depth_intrinsics_.cy, 0.1, 10),
        pangolin::ModelViewLookAt(0, 0, 0, 0, 0, 0.1, pangolin::AxisNegY));

    // 默认渲染窗口，这里面的几个参数基本是固定的
    pangolin::View& d_cam = pangolin::Display("cam")
                                .SetBounds(0, 1.0f, 0, 1.0f, double(-kWindowWidth) / kWindowHeight)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    // 新建一个用于显示 color image 的小窗口
    // setBounds()
    // 函数中，前两个参数是图片在整个全局窗口中的垂直方向的位置（数值是比例），接着两个是水平方向上的位置（数值同样是比例）
    // SetLock() 函数是设置该小窗口的固定位置是在上下左右的哪个方向。
    const double kRange = 0.2;
    pangolin::View& color_image_view = pangolin::Display("color")
                                           .SetBounds(0, kRange, 0, kRange, double(kWindowWidth) / kWindowHeight)
                                           .SetLock(pangolin::LockLeft, pangolin::LockBottom);
    // 新建用于显示 depth image 的小窗口，将其放置在和 color image 小窗口的右边（通过 SetBounds()
    // 中水平方向的那两个位置参数）
    pangolin::View& depth_image_view =
        pangolin::Display("depth")
            .SetBounds(0, kRange, kRange, kRange * 2, double(kWindowWidth) / kWindowHeight)
            .SetLock(pangolin::LockLeft, pangolin::LockBottom);
    std::cout << "Resize the window to experiment with SetBounds, SetLock and SetAspect." << std::endl;
    std::cout << "Notice that the cubes aspect is maintained even though it covers the whole screen." << std::endl;

    // 新建 color image 的 2D texture 区域，其实就是调用了 OpenGL 中的 glTexImage2D。它其实是在 GPU（没有 GPU
    // 的话应该就是 CPU）中预先分配好一块内存。
    // 官方文档：https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml。
    // 有关 glTexImage2D 的参数的很好的介绍：https://blog.csdn.net/csxiaoshui/article/details/27543615
    // 这里倒数第二个参数是你的 color image 的存储方式，我们使用的 OpenCV 中就是 BGR 的顺序。
    pangolin::GlTexture color_image_texture(
        processor.color_width_, processor.color_height_, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);

    // 新建 depth image 的 texture 区域，有两种选择。
    // 使用 8UC3 的 3-channel 深度图，每个 channel 的值将来都会设定成相同的灰度值，这样最终显示的就是一张灰度图。
    // pangolin::GlTexture depth_image_texture(
    //     processor.depth_width_, processor.depth_height_, GL_RED, false, 0, GL_RED, GL_UNSIGNED_BYTE);
    // 这是对应的最终使用的图片的 Mat，因为 depth image 读取后是按照 16UC1 格式的，无法直接来渲染，因此一会还要转化成
    // 8UC3。 cv::Mat gray_image(processor.depth_height_, processor.depth_width_, CV_8UC3);

    // 使用 8UC1 的 1 channel 深度图，这里只能用类似 GL_RED, GL_GREEN 或者 GL_BLUE 类型了。这样最终显示的就是一张
    // 深浅不一的红色（或绿色蓝色）图片
    pangolin::GlTexture depth_image_texture(
        processor.depth_width_, processor.depth_height_, GL_RED, false, 0, GL_RED, GL_UNSIGNED_BYTE);
    // 单通道图片
    cv::Mat gray_image(processor.depth_height_, processor.depth_width_, CV_8UC1);

    int frame_idx = kStartFrameIdx;

    // Default hooks for exiting (Esc) and fullscreen (tab).
    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        // 背景清理设定成白色
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        // Set some random image data and upload to GPU
        // setImageData(imageArray, 3 * width * height);
        // imageTexture.Upload(imageArray, GL_RGB, GL_UNSIGNED_BYTE);
        if (frame_idx <= kEndFrameIdx)
        {
            if (!processor.readColorImage(frame_idx))
                break;
            if (!processor.readDepthImage(frame_idx))
                break;
            if (!processor.readCameraPoseFile(frame_idx))
                break;
            frame_idx++;

            // 深度图片默认是 16UC1 类型，每个 16-bit 元素代表了一个以 mm 为单位的深度值。
            // 下面是将深度图 normalize 并且转换成 8UC3 的灰度图，方便渲染
            // cv::normalize(processor.depth_image_, processor.depth_image_, 0, 255, cv::NORM_MINMAX, CV_8U);
            // cv::cvtColor(processor.depth_image_, gray_image, cv::COLOR_GRAY2RGB, 3);

            // 下面是将深度图 normalize 成 8UC1 的灰度图，不过只有 Red channel，因此显示出来会是红红的一片
            cv::normalize(processor.depth_image_, gray_image, 0, 255, cv::NORM_MINMAX, CV_8U);
        }

        glPushMatrix();
        // 加入 camera pose 再绘制三维图形。注意 OpenGL 的 transformation matrix 是 column-major 的，而 Eigen 中
        // 的 Matrix 的存储方式恰好正是 column-major 存储的，因此直接用 data() 函数获取其指针就行了。
        // glMultMatrixd(processor.pose_camera_to_global_.data());
        drawCameraFrustum();
        processor.drawDepthPointCloud();
        glPopMatrix();
        // pangolin::glDrawColouredCube();

        // 同样道理，因为 cv::Mat 是 BGR 顺序，因此这里参数也设置成 GL_BGR
        color_image_texture.Upload(processor.color_image_.data, GL_BGR, GL_UNSIGNED_BYTE);
        color_image_view.Activate();  // 显示图片
        // 注意：cv::Mat 存储图片是自下而上的，单纯的渲染所渲染出来的图片是倒置的，因此需使用 RenderToViewportFlipY()
        // 函数 进行渲染，将原本上下倒置的图片进行自下而上渲染，使显示的图片是正的。
        glColor3f(1.0, 1.0, 1.0);
        color_image_texture.RenderToViewportFlipY();
        // imageTexture.RenderToViewport();

        depth_image_texture.Upload(gray_image.data, GL_RED, GL_UNSIGNED_BYTE);
        depth_image_view.Activate();
        glColor3f(1.0, 1.0, 1.0);
        depth_image_texture.RenderToViewportFlipY();

        pangolin::FinishFrame();
    }

    // delete[] imageArray;

    return 0;
}