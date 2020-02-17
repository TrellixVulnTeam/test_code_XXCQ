#include <limits>
#include <iostream>
#include <string>

#include <pangolin/pangolin.h>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

void setImageData(unsigned char* imageArray, int size)
{
    for (int i = 0; i < size; i++)
    {
        imageArray[i] = (unsigned char)(rand() / (RAND_MAX / 255.0));
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: image_basic <bundlefusion_rgbd_folder>" << std::endl;
        return -1;
    }

    std::string rgbd_folder(argv[1]);
    int frame_idx = 0;
    std::string str_frame_idx = std::to_string(frame_idx);
    std::string frame_fname = "frame-" + std::string(6 - str_frame_idx.length(), '0') + str_frame_idx;
    std::string frame_fullname = rgbd_folder + "/" + frame_fname;
    std::cout << "Read image file: " << frame_fullname << std::endl;
    cv::Mat color_img = cv::imread(frame_fullname + ".color.jpg", cv::IMREAD_COLOR);
    std::cout << color_img.type() << std::endl;

    const int kWindowWidth = 1280;
    const int kWindowHeight = 960;
    const int kImageWidth = color_img.cols;
    const int kImageHeight = color_img.rows;

    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Main", kWindowWidth, kWindowHeight);

    // 默认开启全屏模式
    // pangolin::ToggleFullscreen();

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // 设置 Projection Matrix 和 gluLookAt() 参数。
    // pangolin::ProjectionMatrix() 的参数依次是：
    // - w, h: 图片长和宽度
    // - fu, fv: focal length fx and fy;
    // - u0, v0: 中心点在图片中的位置，通常就是 cx 和 cy，即图片正中心
    // - zNear, zFar: 即 projection near plane 和 far plane 的位置（Z 轴上），不过它们都是正值
    //
    // 
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(kImageWidth, kImageHeight, 420, 420, 512, 389, 0.1,1000),
        pangolin::ModelViewLookAt(0, 0, 0, 0, 0, 1, pangolin::AxisNegY));
    // pangolin::OpenGlRenderState s_cam(
    //     pangolin::ProjectionMatrix(kWindowWidth, kWindowHeight, 420, 420, 512, 389, 0.1,1000),
    //     pangolin::ModelViewLookAt(-1, 1, -1, 0, 0, 0, pangolin::AxisY));


    // Aspect ratio allows us to constrain width and height whilst fitting within specified
    // bounds. A positive aspect ratio makes a view 'shrink to fit' (introducing empty bars),
    // whilst a negative ratio makes the view 'grow to fit' (cropping the view).
    pangolin::View& d_cam = pangolin::Display("cam")
                                .SetBounds(0, 1.0f, 0, 1.0f, double(-kWindowWidth) / kWindowHeight)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    // This view will take up no more than a third of the windows width or height, and it
    // will have a fixed aspect ratio to match the image that it will display. When fitting
    // within the specified bounds, push to the top-left (as specified by SetLock).
    pangolin::View& d_image = pangolin::Display("image")
                                  //   .SetBounds(2 / 3.0f, 1.0f, 0, 1 / 3.0f, 640.0 / 480)
                                  .SetBounds(0, 0.2, 0, 0.2, double(kWindowWidth) / kWindowHeight)
                                  .SetLock(pangolin::LockLeft, pangolin::LockBottom);

    std::cout << "Resize the window to experiment with SetBounds, SetLock and SetAspect." << std::endl;
    std::cout << "Notice that the cubes aspect is maintained even though it covers the whole screen." << std::endl;

    // const int width = 32;
    // const int height = 24;
    // unsigned char* imageArray = new unsigned char[3 * width * height];
    // pangolin::GlTexture imageTexture(width, height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    pangolin::GlTexture imageTexture(color_img.cols, color_img.rows, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);

    // Default hooks for exiting (Esc) and fullscreen (tab).
    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        // 背景清理设定成白色
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glColor3f(1.0, 1.0, 1.0);
        pangolin::glDrawColouredCube();

        // Set some random image data and upload to GPU
        // setImageData(imageArray, 3 * width * height);
        // imageTexture.Upload(imageArray, GL_RGB, GL_UNSIGNED_BYTE);

        // 同样道理，因为 cv::Mat 是 BGR 顺序，因此这里参数也设置成 GL_BGR
        imageTexture.Upload(color_img.data, GL_BGR, GL_UNSIGNED_BYTE);

        // display the image
        d_image.Activate();
        glColor3f(1.0, 1.0, 1.0);
        // imageTexture.RenderToViewport();

        // 注意：cv::Mat 存储图片是自下而上的，单纯的渲染所渲染出来的图片是倒置的，因此需使用 RenderToViewportFlipY() 函数
        // 进行渲染，将原本上下倒置的图片进行自下而上渲染，使显示的图片是正的。
        imageTexture.RenderToViewportFlipY();

        pangolin::FinishFrame();
    }

    // delete[] imageArray;

    return 0;
}