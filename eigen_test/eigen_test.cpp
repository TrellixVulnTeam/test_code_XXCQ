/***************************************************************/
/*
    Common usage of Eigen.

    Ref:
    - A Simple Eigen Library Tutorial: https://www.cc.gatech.edu/classes/AY2015/cs4496_spring/Eigen.html
    - Official:
    https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
    https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html
    
*/

#include <iostream>
#include <Eigen/Eigen>
using namespace std;

void printPose(const Eigen::Affine3f& pose)
{
    Eigen::Matrix3f rot = pose.rotation();
    Eigen::Vector3f trans = pose.translation();
    cout << "Rotation:" << endl << rot << std::endl;
    cout << "Translation: " << endl << trans << std::endl;
}

void testMatrixBasic()
{
    Eigen::Matrix3f rot;
    rot << 0.924951, 0.379856, 0.0132056, -0.379948, 0.924993, 0.00526768, -0.0102141, -0.00988978, 0.999899;
    cout << "Rotation matrix:" << endl;
    cout << rot << endl;

    Eigen::Matrix3f rot_inv = rot.inverse();
    cout << "Inverse matrix:" << endl;
    cout << rot_inv << endl;

    // Matrix norm
    // Official Ref: https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html
    float norm_val = rot.squaredNorm();  // Frobenius norm (sum of squares of each value)
}

//! Official Ref: https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
void testSparseLinearSystem() {}

void testQuaternion()
{
    Eigen::Quaterniond rot;

    rot.setFromTwoVectors(Eigen::Vector3d(0, 1, 0), pos);

    Eigen::Matrix<double, 3, 3> rotationMatrix;
    // Convert a quaternion to rotation matrix3d
    rotationMatrix = rot.toRotationMatrix();

    Eigen::Quaterniond q(2, 0, 1, -3);
    std::cout << "This quaternion consists of a scalar " << q.w() << " and a vector " << std::endl
              << q.vec() << std::endl;

    q.normalize();

    std::cout << "To represent rotation, we need to normalize it such that its length is " << q.norm() << std::endl;

    Eigen::Vector3d vec(1, 2, -1);
    Eigen::Quaterniond p;
    p.w() = 0;
    p.vec() = vec;
    Eigen::Quaterniond rotatedP = q * p * q.inverse();
    Eigen::Vector3d rotatedV = rotatedP.vec();
    std::cout << "We can now use it to rotate a vector " << std::endl
              << vec << " to " << std::endl
              << rotatedV << std::endl;

    // convert a quaternion to a 3x3 rotation matrix:
    Eigen::Matrix3d R = q.toRotationMatrix();

    std::cout << "Compare with the result using an rotation matrix " << std::endl << R * vec << std::endl;

    Eigen::Quaterniond a = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond b = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond c;
    // Adding two quaternion as two 4x1 vectors is not supported by the Eigen API.
    // That is, c = a + b is not allowed. The solution is to add each element:

    c.w() = a.w() + b.w();
    c.x() = a.x() + b.x();
    c.y() = a.y() + b.y();
    c.z() = a.z() + b.z();
}

void testMatrixXd()
{
    // Transpose and inverse:
    Eigen::MatrixXd A(3, 2);
    A << 1, 2, 2, 3, 3, 4;

    Eigen::MatrixXd B = A.transpose();  // the transpose of A is a 2x3 matrix
    // computer the inverse of BA, which is a 2x2 matrix:
    Eigen::MatrixXd C = (B * A).inverse();
    C.determinant();  // compute determinant
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d m = Eigen::Matrix3d::Random();
    m = (m + Eigen::Matrix3d::Constant(1.2)) * 50;
    Eigen::Vector3d v2(1, 2, 3);

    cout << "m =" << endl << m << endl;
    cout << "m * v2 =" << endl << m * v2 << endl;

    // Accessing matrices:
    Eigen::MatrixXd A2 = Eigen::MatrixXd::Random(7, 9);
    std::cout << "The fourth row and 7th column element is " << A2(3, 6) << std::endl;

    Eigen::MatrixXd B2 = A2.block(1, 2, 3, 3);
    std::cout << "Take sub-matrix whose upper left corner is A(1, 2)" << std::endl << B2 << std::endl;

    Eigen::VectorXd a2 = A2.col(1);  // take the second column of A
    Eigen::VectorXd b2 = B2.row(0);  // take the first row of B2

    Eigen::VectorXd c2 = a2.head(3);  // take the first three elements of a2
    Eigen::VectorXd d2 = b2.tail(2);  // take the last two elements of b2
}

void testAffineMatrix()
{
    // Use manually created rotation and translation
    Eigen::Matrix3f rot;
    rot << 0.924951, 0.379856, 0.0132056, -0.379948, 0.924993, 0.00526768, -0.0102141, -0.00988978, 0.999899;
    Eigen::Vector3f trans(-0.5003, -0.321357, -0.00425079);
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.linear() = rot;  // assign rotation matrix to the linear (rotation) part
    transform.translation() = trans;
    cout << "Input transformation: " << endl;
    printPose(transform);

    Eigen::Vector3f pt(1.9, 2.5, -0.763);
    Eigen::Vector3f pt_trans = transform * pt;

    /// Test Euler angles from the rotation matrix of an Affine matrix
    // Here angles[0] is rotation for Z-axis, angles[1] is Y-axis, angles[2] is X-axis.
    // Actually any order of the eulerAngles() parameters is fine (such as (0,1,2) or even (0,2,1)),
    // but note to use the same order of angles to recover the transformation later.
    Eigen::Vector3f angles = transform.rotation().eulerAngles(2, 1, 0);
    cout << angles[2] << " " << angles[1] << " " << angles[0] << endl;
    Eigen::Matrix3f rot_new;
    // Note to use the same order as the eulerAngles() parameters in the multiplication here.
    // Here the 'rot_new' should be exactly the same as transform.rotation()
    rot_new = Eigen::AngleAxisf(angles[0], Eigen::Vector3f::UnitZ()) *
              Eigen::AngleAxisf(angles[1], Eigen::Vector3f::UnitY()) *
              Eigen::AngleAxisf(angles[2], Eigen::Vector3f::UnitX());

    // // Another example. Now angles[0] is for X-axis, angles[1] is Y, angles[2] is Z.
    // Eigen::Vector3f angles = transform.rotation().eulerAngles(0, 1, 2);
    // rot_new = Eigen::AngleAxisf(angles[0], Eigen::Vector3f::UnitX()) *
    //           Eigen::AngleAxisf(angles[1], Eigen::Vector3f::UnitY()) *
    //           Eigen::AngleAxisf(angles[2], Eigen::Vector3f::UnitZ());
    // Eigen::Vector3f angles = transform.rotation().eulerAngles(2, 1, 0);
    // cout << angles[2] << " " << angles[1] << " " << angles[0] << endl;
    // rot_new = Eigen::AngleAxisf(angles[0], Eigen::Vector3f::UnitZ()) *
    //           Eigen::AngleAxisf(angles[1], Eigen::Vector3f::UnitY()) *
    //           Eigen::AngleAxisf(angles[2], Eigen::Vector3f::UnitX());
}

int main()
{
    testMatrixBasic();
    testAffineMatrix();
    return 0;
}
