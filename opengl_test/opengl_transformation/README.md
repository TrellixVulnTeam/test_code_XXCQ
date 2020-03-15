# opengl_transformation

![](resources/demo_opengl_transformation.gif)

本工程是一个有关 OpenGL Transformation Pipeline 的模板工程。它实现的内容有：
- 定义了一个 Camera Class，负责所有和 Camera View 相关的处理，包括：
    - 计算 Camera View Matrix，即，自行实现了 `gluLookAt()` 函数；
    - 计算 Projection Matrix，目前只有 Perspective Projection，即自行实现了 `gluPerspective()` 函数。
    - 处理平移和旋转带来的 Camera View Matrix 的变化，这样的话在 glfw 主程序中通过监视键盘和鼠标输入便可实现平移和旋转物体。
- 使用 Camera Class 的 glfw 主程序。它还包含了使用鼠标和键盘控制模型旋转和平移的功能。

## 控制方法

运行程序后：
- 鼠标左键拖拽：旋转模型；
- 鼠标左键和 <kbd>Command</kbd> 按键结合：平移模型；
- <kbd>WASD</kbd>: 前后左右移动摄像机视点。

## 注意
- Camera Class 中并不包含 Model Matrix（即，模型的局部坐标系和 Glbal 全局坐标系之间的转移矩阵），这个要用户自行实现。
- 通常，最终的 OpenGL Transformation Matrix 是三个矩阵的乘积：`Projection Matrix * Camera View Matrix * Model Matrix`