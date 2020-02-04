#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

void runFeatureMatching(cv::Mat& color_img1, cv::Mat& color_img2, std::vector<cv::KeyPoint>& keypoints1,
    std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches_out, bool flag_draw_matches = false)
{
    // 计算 ORB 特征点。
    // ORB 参数的详情参考：https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html。
    // 这里用的全是默认参数，因此其实全可以删除。
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    orb->detect(color_img1, keypoints1);
    orb->detect(color_img2, keypoints2);

    // // 画出特征点
    // cv::Mat img_features1;
    // cv::drawKeypoints(color_img1, keypoints1, img_features1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // imshow("ORB特征点", img_features1);

    // 根据特征点位置计算 BRIEF 描述子
    cv::Mat descriptors1, descriptors2;
    orb->compute(color_img1, keypoints1, descriptors1);
    orb->compute(color_img2, keypoints2, descriptors2);

    // 匹配两张图片的特征点。这里使用的是Hammings举例。BFMatcher是Brute-Force匹配方法，即对I1的每个特征点，找出I2中距离最近的特征点。
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors1, descriptors2, matches);

    // 找出所有匹配之间的最小距离和最大距离，即是最相似的和最不相似的两组点之间的距离
    double min_dist = 1e8, max_dist = 0;
    for (int i = 0; i < descriptors1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
    std::cout << "Max distance: " << max_dist << ", Min distance: " << min_dist << std::endl;

    // 筛选出好的匹配点对。当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。但有时候最小距离会非常小，设置一个经验值作为下限。
    for (int i = 0; i < descriptors1.rows; i++)
    {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0))  // 30 是一个经验值
            matches_out.push_back(matches[i]);
    }

    // 绘制匹配结果
    if (flag_draw_matches)
    {
        cv::Mat match_img;
        cv::Mat goodmatch_img;
        cv::drawMatches(color_img1, keypoints1, color_img2, keypoints2, matches, match_img);
        cv::drawMatches(color_img1, keypoints1, color_img2, keypoints2, matches_out, goodmatch_img);
        cv::imshow("初始匹配", match_img);
        cv::imshow("优化后匹配", goodmatch_img);
        cv::waitKey(0);
    }
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::cout << "Usage: program color1 color2 depth1 depth2" << std::endl;
        return -1;
    }

    // Read all 4 images
    cv::Mat color_img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (!color_img1.data)
    {
        std::cout << "Cannot read color image " << std::string(argv[1]) << "!" << std::endl;
        return -1;
    }
    cv::Mat color_img2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (!color_img2.data)
    {
        std::cout << "Cannot read color image " << std::string(argv[2]) << "!" << std::endl;
        return -1;
    }
    cv::Mat depth_img1 = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
    if (!depth_img1.data)
    {
        std::cout << "Cannot read depth image " << std::string(argv[3]) << "!" << std::endl;
        return -1;
    }
    cv::Mat depth_img2 = cv::imread(argv[4], cv::IMREAD_UNCHANGED);
    if (!depth_img2.data)
    {
        std::cout << "Cannot read depth image " << std::string(argv[4]) << "!" << std::endl;
        return -1;
    }

    runFeatureMatching(color_img1, color_img2);

    return 0;
}