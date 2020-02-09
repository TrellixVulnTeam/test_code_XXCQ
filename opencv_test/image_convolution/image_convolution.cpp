#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

//! 使用可拆分卷积实现图像卷积
/*!
    @param in 输入图像的指针，这里输入必须是一个 1-channel 的图片，例如灰度图
    @param rows 图像行数
    @param cols 图像列数
    @param kernelX 核函数拆分后的列向量
    @param kernelY 核函数拆分后的行向量
    @param sizeX 核函数列向量长度
    @param sizeY 核函数行向量长度
    @param out 输出图像的指针
*/
bool convolve2DSeparable(
    unsigned char* in, int rows, int cols, float* kernelX, int sizeX, float* kernelY, int sizeY, unsigned char* out)
{
    if (!in || !out || !rows || !cols || !kernelX || !kernelY)
        return false;
    int N = rows * cols;
    float* tmp = new float[N];  // 额外分配空间用于存储第一次卷积的结果
    for (int i = 0; i < N; ++i)
        tmp[i] = 0;

    // 行向量卷积
    int hy = (sizeY >> 1);
    int jump_gap = ((sizeY - 1) >> 1);  // 每一行遍历后，指向图片元素的指针需要调过的元素个数
    int left_border = hy;
    int right_border = cols - hy;
    float* ptr_tmp = tmp;  // 用指针运算可以加快速度
    unsigned char* ptr_in = in;
    int offset = 0;  // 用于每次向右移动 kernel 一位
    for (int i = 0; i < rows; ++i)
    {
        // 左边界区间 [0, hy - 1]，kernel 右边有一部分用不到
        offset = 0;  // 用于每次向右移动 kernel 一位
        for (int j = 0; j < left_border; ++j)
        {
            *ptr_tmp = 0;  // 其实最开始已经初始化全部的 tmp 为 0 了，为了保险还是再初始化一下
            for (int k = hy + offset, t = 0; k >= 0; t++, k--)
                *ptr_tmp += *(ptr_in + t) * kernelY[k];  // 这里第一个 * 是取地址操作符
            ptr_tmp++;
            offset++;
        }
        // 中区间 [hy, cols - hy - 1]，一般情况
        for (int j = left_border; j < right_border; ++j)
        {
            *ptr_tmp = 0;
            for (int k = sizeY - 1, t = 0; k >= 0; t++, k--)
                *ptr_tmp += *(ptr_in + t) * kernelY[k];
            ptr_tmp++;
            ptr_in++;  // 注意这里也递增了图片指针，始终让它指向 Kernel 的左边界对应的图片的点
        }
        // 右边界区间 [colos - hy, cols - 1]，kernel 左边有一部分用不到
        offset = 1;  // 这次它表示在右边界处理中，kernel 最右边的元素位置
        for (int j = right_border; j < cols; ++j)
        {
            *ptr_tmp = 0;
            for (int k = sizeY - 1, t = 0; k >= offset; t++, k--)
                *ptr_tmp += *(ptr_in + t) * kernelY[k];
            ptr_tmp++;
            ptr_in++;
            offset++;
        }
        // 一行的行向量卷积结束后，输入指针会停在离右边界的距离是 sizeY/2 再加上 1 的位置。
        // 因此需要跳过一部分数据到达下一行。
        ptr_in += jump_gap;
    }

    // 列向量卷积。实现上很巧妙。
    int hx = (sizeX >> 1);  // 类似上面的 hy
    int up_border = hx;     // 区间的两个边界
    int bottom_border = rows - hx;
    ptr_tmp = tmp;  // 从头开始
    unsigned char* ptr_out = out;
    float* sum = new float[cols];  // 这一步额外需要一个行向量记录每一行的结果，因为输出的 out 数组是 unsigned char,
                                   // 但是卷积结果是 float，因此最好还是额外定义一个数组先存储结果，再赋值到 out 中
    for (int i = 0; i < cols; ++i)
        sum[i] = 0;
    // 上边界区间 [0, hx - 1]
    offset = 0;
    for (int i = 0; i < up_border; ++i)
    {
        for (int k = hx + offset; k >= 0; k--)
        {
            // 当 k 减小后，j 又回到了从 0 开始增大，因此，同一列的乘积和结果累积到了 sum[j] 中
            for (int j = 0; j < cols; ++j)
            {
                sum[j] += *ptr_tmp * kernelX[k];
                ptr_tmp++;  // 注意它是始终递增的
            }
        }
        for (int j = 0; j < cols; ++j)
        {
            *ptr_out = (unsigned char)(sum[j]);  // 这里其实默认将 float 转成了 unsigned char
            ptr_out++;
            sum[j] = 0;  // 记得清零供下次使用
        }
        ptr_tmp = tmp;  // 上边界区间中，计算完一行卷积后，ptr_tmp 始终指向开头
        offset++;
    }
    // 中间区间 [hx, rows - hx - 1]。做法和上边界区间很类似。
    float* ptr_init = ptr_tmp; // 拷贝一份初始指针
    for (int i = up_border; i < bottom_border; ++i)
    {
        for (int k = sizeX - 1; k >= 0; k--)
        {
            for (int j = 0; j < cols; ++j)
            {
                sum[j] += *ptr_tmp * kernelX[k];
                ptr_tmp++;
            }
        }
        for (int j = 0; j < cols; ++j)
        {
            *ptr_out = sum[j];
            ptr_out++;
            sum[j] = 0; 
        }
        ptr_init += cols;  // 左上角指针要向下移动一行
        ptr_tmp = ptr_init;
    }
    // 下边界区间 [rows - hx， rows - 1]
    offset = 1;
    for (int i = bottom_border; i < rows; ++i)
    {
        for (int k = sizeX - 1; k >= offset; k--)
        {
            for (int j = 0; j < cols; ++j)
            {
                sum[j] += *ptr_tmp * kernelX[k];
                ptr_tmp++;
            }
        }
        for (int j = 0; j < cols; ++j)
        {
            *ptr_out = sum[j];
            ptr_out++;
            sum[j] = 0;
        }
        offset++;
        ptr_init += cols;
        ptr_tmp = ptr_init;
    }
    // 最后记得释放内存
    delete[] sum;
    delete[] tmp;
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: image_convolution <input_image>" << std::endl;
        return -1;
    }
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (!src.data)
    {
        std::cout << "Cannot read image " << std::string(argv[1]) << "!" << std::endl;
        return -1;
    }
    cv::Mat dev = cv::Mat(src.size(), src.type(), cv::Scalar(0));
    float kernelX[5] = {1 / 16.0f, 4 / 16.0f, 6 / 16.0f, 4 / 16.0f, 1 / 16.0f};
    float kernelY[5] = {1 / 16.0f, 4 / 16.0f, 6 / 16.0f, 4 / 16.0f, 1 / 16.0f};
    convolve2DSeparable(src.data, src.rows, src.cols, kernelX, 5, kernelY, 5, dev.data);

    cv::imshow("Original", src);
    cv::imshow("Result", dev);
    cv::waitKey();

    return 0;
}