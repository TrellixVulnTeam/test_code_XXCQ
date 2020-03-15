#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <math.h>

/// 相机类。
class Camera
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// Convert an angle to radian
    static float toRadian(float angle) { return angle * M_PI / 180; }

    /// 透视投影的自行实现，返回该投影矩阵。输入参数有所变化。注意，这里假设透视的 volume 是对称的。即，近平面的
    /// 右边界就是左边界位置的相反数 right = -left, 并且上端边界是下边界的相反数 top = -bottom（即，Z 轴穿过了
    /// 近平面的正中心）。
    /// Reference: http://www.songho.ca/opengl/gl_transform.html#matrix
    /// @param right 近平面的右边界
    /// @param top 近平面的左边界
    /// @param near 近平面位置，通常比较小，例如 0.1f
    /// @param far 远平面位置，通常是一个很大的值，例如 100.0f
    static Eigen::Matrix4f perspectiveProjection(float right, float top, float near, float far)
    {
        assert(near < far && right > 0 && top > 0);
        Eigen::Matrix4f res = Eigen::Matrix4f::Zero();
        res(0, 0) = near / right;
        res(1, 1) = near / top;
        res(2, 2) = -(far + near) / (far - near);
        res(3, 2) = -1.0;
        res(2, 3) = -(2.0 * far * near) / (far - near);
        return res;
    }
    /// 透视投影的一个重载函数，返回该投影矩阵。算法相同，只是输入参数不同。这里的输入参数和 gluPerspective() 完全相同。
    /// Ref: https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
    /// @param fovy Y 方向（垂直方向）的 fov，单位是角度，例如 45
    /// @param aspect 近平面的宽高比，即 width / height
    /// @param near 近平面位置，通常比较小，例如 0.1f
    /// @param far 远平面位置，通常是一个很大的值，例如 100.0f
    static Eigen::Matrix4f perspectiveProjectionFov(float fovy, float aspect, float near, float far)
    {
        assert(fovy > 0 && aspect > 0 && near < far);
        float rad = toRadian(fovy);
        float tanHalfFovy = tan(rad / 2.0);
        Eigen::Matrix4f res = Eigen::Matrix4f::Zero();
        res(0, 0) = 1.0 / (aspect * tanHalfFovy);
        res(1, 1) = 1.0 / (tanHalfFovy);
        res(2, 2) = -(far + near) / (far - near);
        res(3, 2) = -1.0;
        res(2, 3) = -(2.0 * far * near) / (far - near);
        return res;
    }

    /// 自行实现 gluLookAt() 函数，返回该矩阵。输入参数和其完全相同。
    /// @param eye 相机的位置
    /// @param center 被观察者的位置
    /// @param up 相机正上方的方向
    /// @return 4x4 的 Camera View 矩阵。
    /// Reference: https://learnopengl-cn.github.io/01%20Getting%20started/09%20Camera/
    static Eigen::Matrix4f cameraLookAt(const Eigen::Vector3f& position, const Eigen::Vector3f& facingVector, const Eigen::Vector3f& upVector)
    {
        Eigen::Vector3f u = upVector.normalized();      // up vector
        Eigen::Vector3f f = facingVector.normalized();  // front vector，注意这是 direction vector 反过来
        Eigen::Vector3f r = (f.cross(u)).normalized();    // right vector
        Eigen::Matrix4f res;
        res << r.x(), r.y(), r.z(), -r.dot(position), 
            u.x(), u.y(), u.z(), -u.dot(position), 
            -f.x(), -f.y(), -f.z(), f.dot(position), 
            0, 0, 0, 1;
        return res;
    }

    /// 定义一个简单的相机移动类，记录各种状态
    enum class ModelViewMode
    {
        NOTHING,
        ROTATE,
        TRANSLATE,
        ZOOM,
        MOVE_X, // 朝相机的三个轴方向平移
        MOVE_Y,
        MOVE_Z
    };

    Camera(const Eigen::Vector3f& cameraPosition = Eigen::Vector3f(0.0, 0.0, 3.0),
        const Eigen::Vector3f& targetPosition = Eigen::Vector3f(0.0, 0.0, 0.0),
        const Eigen::Vector3f& upVector = Eigen::Vector3f(0.0, 1.0, 0.0), 
        float fov = 45.0f, 
        float aspect = 4.0f / 3,
        float speed = 10.0f, 
        float sensitivity = 0.1f)
        : m_cameraPosition(cameraPosition),
          m_targetPosition(targetPosition),
          m_upVector(upVector),
          m_sensitivity(sensitivity),
          m_fov(fov),
          m_aspectRatio(aspect),
          m_modelViewMode(ModelViewMode::NOTHING)
    {
        m_projectionMatrix = perspectiveProjectionFov(m_fov, m_aspectRatio, 0.1f, 100.0f);
        m_facingVector = m_targetPosition - m_cameraPosition;
        updateCameraView();
    }

    Camera(float cx, float cy, float cz, // camera position
        float tx, float ty, float tz,    // target position
        float ux, float uy, float uz,    // camera up vector
        float fov = 45.0f, 
        float aspect = 4.0f/3, 
        float speed = 10.0f, 
        float sensitivity = 0.1f)
        : m_cameraPosition(cx, cy, cz),
          m_targetPosition(tx, ty, tz),
          m_upVector(ux, uy, uz),
          m_sensitivity(sensitivity),
          m_fov(fov),
          m_aspectRatio(aspect),
          m_modelViewMode(ModelViewMode::NOTHING)
    {
        m_projectionMatrix = perspectiveProjectionFov(m_fov, m_aspectRatio, 0.1f, 100.0f);
        m_facingVector = m_targetPosition - m_cameraPosition;
        updateCameraView();
    }

    ~Camera() {}

    /// 处理 camera view rotation
    void processRotation(float xOffset, float yOffset)
    {
        xOffset *= m_sensitivity;
        yOffset *= m_sensitivity;
        // 注意，模型旋转的方向是和 camera facing vector 旋转方向相反的，因为我们在（比如用鼠标拖拽）旋转模型时，其实这里实现的并非是旋转模型，而是旋转 camera view，
        // 即相机朝着模型旋转方向的反方向旋转。这里先旋转相机的 facing vector，然后再计算新的 camera position 的位置。
        m_facingVector = Eigen::AngleAxisf(toRadian(-xOffset), m_upVector) * Eigen::AngleAxisf(toRadian(-yOffset), m_rightVector) * m_facingVector;
        m_facingVector.normalize();
        m_cameraPosition = m_targetPosition - (m_targetPosition - m_cameraPosition).norm() * m_facingVector;
        updateCameraView();
    }

    /// 处理 camera view translation
    void processTranslation(ModelViewMode mode, float offset)
    {
        Eigen::Vector3f delta = Eigen::Vector3f::Zero();
        if (mode == ModelViewMode::MOVE_Z)
            delta = m_facingVector * offset;
        else if (mode == ModelViewMode::MOVE_X)
            delta = m_rightVector * offset;
        else if (mode == ModelViewMode::MOVE_Y)
            delta = m_upVector * offset;
        m_cameraPosition += delta;
        m_targetPosition += delta;
        m_cameraViewMatrix = cameraLookAt(m_cameraPosition, m_facingVector, m_upVector);
    }
    /// 处理缩放，通过更改 projection matrix 中的 fov 来实现。当然，修改近平面的位置其实也可以。
    void processZoom(float yoffset)
    {
        if (m_fov >= 1.0f && m_fov <= 90.0f)
            m_fov -= yoffset;
        if (m_fov <= 1.0f)
            m_fov = 1.0f;
        if (m_fov >= 90.0f)
            m_fov = 90.0f;
        m_projectionMatrix = perspectiveProjectionFov(m_fov, m_aspectRatio, 0.1f, 100.0f);
    }

    void setModelViewMode(ModelViewMode mode) { m_modelViewMode = mode; }
    ModelViewMode getModelViewMode() { return m_modelViewMode; } 
    Eigen::Matrix4f getCameraViewMatrix() { return m_cameraViewMatrix; }
    Eigen::Matrix4f getProjectionMatrix() { return m_projectionMatrix; }

private:
    Eigen::Vector3f m_cameraPosition;    // 相机的位置
    Eigen::Vector3f m_targetPosition;    // 被观察物体的位置
    Eigen::Vector3f m_facingVector;      // 相机的朝向（就是 m_targetPosition - m_cameraPosition）
    Eigen::Vector3f m_upVector;          // 相机正上方的方向
    Eigen::Vector3f m_rightVector;       // 相机右边的方向，这里的三个向量组成了相机坐标系
    Eigen::Matrix4f m_cameraViewMatrix;  // camera view matrix
    Eigen::Matrix4f m_projectionMatrix;  // projection matrix
    ModelViewMode m_modelViewMode;       // 模型当前状态
    float m_sensitivity;             // 数值越大，旋转越敏感，每次旋转的幅度越大
    float m_fov;                     // 用于 zoom in/out，数值越大，渲染物体越大
    float m_aspectRatio;             // 用于 perspective projection

    void updateCameraView()
    {
        m_facingVector.normalize();
        m_rightVector = (m_facingVector.cross(m_upVector)).normalized();
        m_upVector = m_rightVector.cross(m_facingVector);
        m_cameraViewMatrix = cameraLookAt(m_cameraPosition, m_facingVector, m_upVector);
    }
};