#include <iostream>
#include <ceres/ceres.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <random>
// #include <opencv2/core/core.hpp> // 提供高斯噪声的alternative

using namespace std;

// Cost function定义必须在struct内部的重载Operator()中
struct CostFunctor
{
    // x和y是观察值，在构造函数中存储到类成员变量中
    CostFunctor(double x, double y) : x_(x), y_(y) {}

    // 必须是模板类函数，模板类T用于转换输入的观察值x和y。
    // abc是待估计的参数，residue是残差，两者都可以是多维的。这里residue是1维，而abc则是3维。
    template <typename T>
    bool operator()(const T *const abc, T *residual) const
    {
        // Function is y - exp(ax^2 + bx + c)
        residual[0] = T(y_) - ceres::exp(abc[0] * T(x_) * T(x_) + abc[1] * T(x_) + abc[2]);
        return true;
    }
    double x_, y_;
};

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);

    double a = 1.0, b = 2.0, c = 1.0;
    double abc[3] = {0, 0, 0};  // 待估计的3个参数
    const int kNumPoints = 100;

    // 设置高斯噪声生成器。这里使用STL中的normal_distribution类
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);  // 使用时间作为seed，否则生成的随机数在每次程序运行都是一批相同的数
    double kNoiseSigma = 1.0;  // 噪声的标准差Standard Deviation
    std::normal_distribution<double> dist(0, kNoiseSigma); // 第1个参数是mean，第2个是标准差
    // cv::RNG rng;               // OpenCV的随机数

    // 随机生成一些有噪声的点对
    vector<double> x_data, y_data;
    std::ofstream writeout("curve_data.dat", std::ios::trunc);
    for (int i = 0; i < kNumPoints; ++i)
    {
        double x = double(i) / kNumPoints;
        x_data.push_back(x);
        // 增加高斯噪声。被注释的是使用OpenCV提供的高斯函数。
        // y_data.push_back(exp(a * x * x + b * x + c) + rng.gaussian(kNoiseSigma));
        y_data.push_back(exp(a * x * x + b * x + c) + dist(generator));
        writeout << x << " " << y_data.back() << std::endl;
    }
    writeout.close();

    // 构建最小二乘问题
    ceres::Problem problem;
    for (int i = 0; i < kNumPoints; ++i)
    {
        /*
        建立cost_function指针。有关参数的说明：
        1) 这里AutoDiffCostFunction是自动求导的方式，编码比较简单，这里就用它了。对应的
        还有数值求导（NumericDiff），以及自行推导导数提供给Ceres。
        2) AutoDiffCostFunction的模板类参数中，第1个参数就是刚刚定义的cost function
        结构体的名字，第2个参数是残差（residue）的维度（这里残差维度是1，即scalar），接
        下来的参数是每一个block中的待估计的变量（即未知数）的维度/个数。这里我们把这a,b,c
        这三个参数写到一个block中，因此就是1。
        3) 设置block的原则通常是，看它们能否在一起计算/求导。例如，如果对于Bundle
        Adjustment（BA）问题，未知数通常有6个变换矩阵T的参数，1个focal length和2个radial
        distortion。于是，将这9个数字放在一个block中。另外，每一个三维空间点也是一个未知数，
        因此它们作为一个block，未知数是3。BA问题中残差通常是2（即像素点的位置差）。于是，最终
        定义的函数就是：AutoDiffCostFunction<CostFunction, 2, 9, 3>(...)
        */
        ceres::CostFunction *cost_function =
            new ceres::AutoDiffCostFunction<CostFunctor, 1, 3>(new CostFunctor(x_data[i], y_data[i]));

        // 将cost function、核函数和待计算的未知数变量传入。这里第2个参数是核函数LossFunction，
        // 这里暂时是空，因为这里我们根本没用核函数。
        problem.AddResidualBlock(cost_function, nullptr, abc);
    }

    // options用于设置Solver的参数，包含了：用Line Search还是Trust Region；是否输出到stdout；
    // 迭代步长；线性系统的solver类型等
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;  // 普通的Dense QR分解来解线性系统，优点是适用于任意的方阵
    ceres::Solver::Summary summary;  // 可以总结一个最终的Report输出，包含了诸如迭代次数，能量变化，最终是否收敛等

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // 最后一步，调用Solve()解决问题
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    std::cout << "Time for solution: " << time_used.count() << endl;

    LOG(WARNING) << "Time " << time_used.count();  // 测试一下GLog的使用（GLog默认被Ceres包含了）

    std::cout << summary.BriefReport() << std::endl;
    // 最终结果就在abc这个数组中
    std::cout << "Result: a = " << abc[0] << ", b = " << abc[1] << ", c = " << abc[2] << std::endl;

    return 0;
}