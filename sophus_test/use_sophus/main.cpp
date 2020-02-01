#include <iostream>
#include <cmath>
using namespace std;
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/so3.h"
#include "sophus/se3.h"

int main()
{
    // 沿 Z 轴转 90 度的旋转矩阵
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    Sophus::SO3 SO3_R(R);  // Sophus::SO(3)可以直接从旋转矩阵构造
    cout << "SO(3) from matrix: " << SO3_R << endl;


    return 0;
}