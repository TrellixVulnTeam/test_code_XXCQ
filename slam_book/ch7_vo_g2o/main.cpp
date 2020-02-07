#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

// TUM RGB-D 深度图的深度值的 scale factor，深度值除以它就是真正的以米为单位的深度值
const double kDepthScaleFactor = 5000.0;

//! 将一个 2D 像素点转换成相机坐标系中的归一化平面上的点（即 Z=1），因此它也是 2D 的。
cv::Point2d pixelToCameraNormalizedPoint(cv::Point2d& pixel, cv::Mat& intrinsics)
{
    return cv::Point2d((pixel.x - intrinsics.at<double>(0, 2)) / intrinsics.at<double>(0, 0),
        (pixel.y - intrinsics.at<double>(1, 2)) / intrinsics.at<double>(1, 1));
}

//! 两张图片的特征点匹配
bool runFeatureMatching(cv::Mat& color_img1, cv::Mat& color_img2, std::vector<cv::KeyPoint>& keypoints1,
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

    // 至少需要8个对应的特征点对才能计算Fundamental Matrix，这里提前验证一下
    return matches_out.size() > 8;
}

//! 使用对极几何（基础矩阵）方法估计相机位姿
bool estimatePoseEpipolar(std::vector<cv::Point2d>& pixels1, std::vector<cv::Point2d>& pixels2, cv::Mat& intrinsics,
    cv::Mat& R_out, cv::Mat& t_out)
{
    // 计算基础矩阵 Fundamental Matrix。这里使用八点法（第3个参数），其他选择还有 cv::FM_RANSAC（RANSAC）方法。
    // API 文档：https://docs.opencv.org/3.4.9/d9/d0c/group__calib3d.html#gae420abc34eaa03d0c6a67359609d8429
    // 注意在 OpenCV4 中，方法名不再是 CV_FM_8POINT 了。
    // 另外，计算相机位姿只需要计算本质矩阵就够了，这里只是演示计算基础矩阵的方法。
    cv::Mat fundamental_matrix = cv::findFundamentalMat(pixels1, pixels2, cv::FM_8POINT);
    // std::cout << "Fundamental Matrix: " << std::endl << fundamental_matrix << std::endl;

    // 计算本质矩阵 Essential Matrix。这里有一个输出矩阵 mask，标记了用来计算本质矩阵的特征点（这些点标记是 1，如果是 0
    // 则说明没有使用），当然 mask 是 optional 的。
    cv::Mat mask;
    cv::Mat essential_matrix = cv::findEssentialMat(pixels1, pixels2, intrinsics, cv::RANSAC, 0.999, 1.0, mask);
    // std::cout << "Essential Matrix: " << std::endl << essential_matrix << std::endl;

    // 计算单应矩阵 Homography。这里使用 RANSAC 方法。同样的，计算相机位姿并不需要它，这里只是演示方法。
    cv::Mat homography = cv::findHomography(pixels1, pixels2, cv::RANSAC);
    // std::cout << "Homography: " << std::endl << homography << std::endl;

    // 从本质矩阵还原出相机位姿。这里的 mask 来自上面计算本质矩阵的 mask，它在这里既是输入又是输出。
    // 最终，mask.at<unsigned char>(i) == 1 对应的点是被该函数使用了来还原相机位姿的点，0 对应的点则是未使用的点。
    // 注意，输出的旋转阵 R_out 是 3x3 的，而位移 t_out 是 3x1 的矩阵。
    cv::recoverPose(essential_matrix, pixels1, pixels2, intrinsics, R_out, t_out, mask);

    /// 接下来是通过对极几何的知识验证刚刚得到的相机位姿是否正确。类似 Unit-test
    // 1. 验证 E = scale * t^ R
    double t0 = t_out.at<double>(0, 0);
    double t1 = t_out.at<double>(1, 0);
    double t2 = t_out.at<double>(2, 0);
    cv::Mat t_skew_R = (cv::Mat_<double>(3, 3) << 0, -t2, t1, t2, 0, -t0, -t1, t0, 0);
    t_skew_R *= R_out;
    // t^ R 和 本质矩阵 E 之间应该只有一个 scale 的差别，因此，计算这个 scale 矩阵（即每个元素的 scale
    // 值），然后找到该矩阵的最大值和最小值，它们应当非常接近。
    cv::Mat scale_mat;
    cv::divide(t_skew_R, essential_matrix, scale_mat);
    double minval = 0, maxval = 0;
    cv::minMaxLoc(scale_mat, &minval, &maxval);
    if (fabs(maxval - minval) > 1e-3)
        return false;

    // 2. 验证对极约束 x2^T E x1 = 0，这里的 x1 和 x2 是相机坐标系中的归一化平面上的点（即 Z=1）
    for (size_t i = 0; i < pixels1.size(); ++i)
    {
        cv::Point2d x1 = pixelToCameraNormalizedPoint(pixels1[i], intrinsics);
        cv::Point2d x2 = pixelToCameraNormalizedPoint(pixels2[i], intrinsics);
        cv::Mat x1_homo = (cv::Mat_<double>(3, 1) << x1.x, x1.y, 1);  // 记得用齐次坐标才能进行矩阵相乘
        cv::Mat x2_homo = (cv::Mat_<double>(3, 1) << x2.x, x2.y, 1);
        cv::Mat residue = x2_homo.t() * t_skew_R * x1_homo;  // 结果其实是一个 scalar，应该接近 0
        if (residue.at<double>(0, 0) > 1e-2)                 // 实验表明它通常小于 0.01
            return false;
    }
    return true;
}

//! 通过三角测量（Triangulation），从相机位姿和二维特征点中计算出三维点的位置并验证之。
bool estimate3DPointsByTriangulation(std::vector<cv::Point2d>& pixels1, std::vector<cv::Point2d>& pixels2,
    cv::Mat& intrinsics, cv::Mat& R, cv::Mat& t, std::vector<cv::Point3d>& points3d)
{
    // 使用 OpenCV 提供的三角测量的函数 cv::triangulatePoints()。参数说明：
    // - 它必须使用 3x4 的相机位姿矩阵。并且，该位姿指代从全局空间坐标系到 2D 图片的变换矩阵。
    // - 输出的 pts4d 是一个 4xN 的矩阵，其中 N == pixels1.size()
    //   指代点的总个数。该矩阵每一列就是一个空间点的齐次坐标，因此需要再转换一下变成 3D 点。
    cv::Mat T1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    cv::Mat T2 = (cv::Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        t.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
    T1 = intrinsics * T1;  // 前面乘以内参后变成到图片的变换矩阵
    T2 = intrinsics * T2;
    cv::Mat pts4d;
    cv::triangulatePoints(T1, T2, pixels1, pixels2, pts4d);
    for (size_t i = 0; i < pixels1.size(); ++i)
    {
        points3d.push_back(cv::Point3d(pts4d.at<double>(0, i) / pts4d.at<double>(3, i),
            pts4d.at<double>(1, i) / pts4d.at<double>(3, i), pts4d.at<double>(2, i) / pts4d.at<double>(3, i)));
    }

    // 验证三维点。这里比较的是计算的全局三维点在相机坐标系中的点和对应的二维特征点在相机坐标系中的点的距离，即将两者都转换到相机坐标系再比较。
    // 当然，也可以将三维点投影到图片上，比较和对应的二维特征点的距离。不过这种比较方式结果不是很明显，实验显示最大误差接近
    // 2 pixels，其实也 很小了。而在相机坐标系比较的话，误差最大不到 0.01，并且绝大多数的误差都在 0.001
    // 左右，这还是比较直观的。
    for (size_t i = 0; i < pixels1.size(); ++i)
    {
        cv::Point2d pt1_cam = pixelToCameraNormalizedPoint(pixels1[i], intrinsics);
        cv::Point2d pt1_cam_3d(points3d[i].x / points3d[i].z, points3d[i].y / points3d[i].z);
        double dis1 = cv::norm(pt1_cam - pt1_cam_3d);

        cv::Point2d pt2_cam = pixelToCameraNormalizedPoint(pixels2[i], intrinsics);
        cv::Mat proj = R * (cv::Mat_<double>(3, 1) << points3d[i].x, points3d[i].y, points3d[i].z) + t;
        cv::Point2d pt2_cam_3d(
            proj.at<double>(0, 0) / proj.at<double>(2, 0), proj.at<double>(1, 0) / proj.at<double>(2, 0));
        double dis2 = cv::norm(pt2_cam - pt2_cam_3d);
        // std::cout << dis1 << " " << dis2 << std::endl;
        if (dis1 > 1e-2 || dis2 > 1e-2)
            return false;
    }
    return true;
}

//! 通过解决 PnP 问题（Perspective-n-points）来计算相机位姿，该方法需要已知一批特征点对应的全局三维点的位置
bool estimatePoseByPnP(cv::Mat& depth_img1, std::vector<cv::Point2d>& pixels1, std::vector<cv::Point2d>& pixels2,
    cv::Mat& intrinsics, std::vector<double>& distortion, cv::Mat& R_out, cv::Mat& t_out)
{
    std::vector<cv::Point3d> object_pts;  // 第 1 张图片的特征点对应的全局三维点的坐标
    std::vector<cv::Point2d> image_pts;   // 第 2 张图片中的二维特征点
    for (size_t i = 0; i < pixels1.size(); ++i)
    {
        unsigned short depth = depth_img1.at<unsigned short>(int(pixels1[i].x), int(pixels1[i].y));
        double z = depth / kDepthScaleFactor;  // 深度值记得要 scale back，单位是 m
        if (z < 0.1 || z > 6.0)                // 深度值过小或者过大都不准确，舍弃它们
            continue;
        cv::Point2d pt = pixelToCameraNormalizedPoint(pixels1[i], intrinsics);
        object_pts.push_back(cv::Point3d(pt.x * z, pt.y * z, z));
        image_pts.push_back(pixels2[i]);
    }
    if (object_pts.size() < 4)  // 通常 PnP 方法至少需要 4 个特征点对
        return false;

    std::cout << "#3D points: " << object_pts.size() << ", #Original Features: " << pixels1.size() << std::endl;

    // 使用 EPNP 方法解决 PnP问题。几个注意点：
    // - 第 4 个参数的 intrinsics 可以使空矩阵 cv::Mat()，这样的话默认 distortion 参数全是 0；
    // - 输出的旋转是 Axis-Angle 形式的旋转向量（即，向量方向是转轴，长度是转角）
    cv::Mat r;
    cv::solvePnP(object_pts, image_pts, intrinsics, distortion, r, t_out, false, cv::SOLVEPNP_EPNP);
    // 调用罗德里格斯公式转换得到最终的旋转阵
    cv::Rodrigues(r, R_out);

    return true;
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        // 依次读入color image 1 and 2, depth image 1 and 2
        std::cout << "Usage: program color1 color2 depth1 depth2" << std::endl;
        return -1;
    }

    // 读取图片
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
    // 相机内参。这里设置成固定的，来自 TUM RGB-D 数据库中的 "Freiburg 2 RGB" dataset
    cv::Mat intrinsics = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    // 相机的畸变（distortion）系数，来自同一个 dataset
    std::vector<double> distortion = {0.2312, -0.7849, -0.0033, -0.0001, 0.9172};

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<cv::DMatch> matches;

    // 特征值匹配
    if (!runFeatureMatching(color_img1, color_img2, keypoints1, keypoints2, matches, false))
    {
        std::cout << "Incorrect feature matching with only " << matches.size() << " pairs of feature points"
                  << std::endl;
        return -1;
    }

    // 提取成功匹配的特征点，未匹配的特征点不能用来计算相机位姿
    std::vector<cv::Point2d> pixels1, pixels2;
    for (auto& mt : matches)
    {
        pixels1.push_back(keypoints1[mt.queryIdx].pt);  // query 指代第 1 张图片
        pixels2.push_back(keypoints2[mt.trainIdx].pt);  // train 指代第 2 张图片，被弄混了
    }

    // 使用对极几何 Epipolar geometry 方法（Essential matrix）估计 camera pose
    cv::Mat R;  // rotation
    cv::Mat t;  // translation
    if (!estimatePoseEpipolar(pixels1, pixels2, intrinsics, R, t))
    {
        std::cout << "Epipolar pose estimation failed!" << std::endl;
        return -1;
    }
    // 这里的 R 基本准确，但是 t 是不准确的，因为只有单目 SLAM 得到的位移的单位未知
    std::cout << "R: " << std::endl << R << std::endl;
    std::cout << "t: " << std::endl << t << std::endl;

    // 使用三角测量来计算空间点位置
    std::vector<cv::Point3d> points3d;
    if (!estimate3DPointsByTriangulation(pixels1, pixels2, intrinsics, R, t, points3d))
    {
        std::cout << "Triangulation failed!" << std::endl;
        return -1;
    }

    // 通过解决 PnP 问题来计算相机位姿
    cv::Mat R_pnp;
    cv::Mat t_pnp;
    if (!estimatePoseByPnP(depth_img1, pixels1, pixels2, intrinsics, distortion, R_pnp, t_pnp))
    {
        std::cout << "PnP failed!" << std::endl;
        return -1;
    }
    std::cout << "R_pnp: " << std::endl << R_pnp << std::endl;
    std::cout << "t_pnp: " << std::endl << t_pnp << std::endl;

    return 0;
}