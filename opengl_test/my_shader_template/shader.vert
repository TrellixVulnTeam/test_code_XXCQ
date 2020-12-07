// 注意 shader core version 也要和你用的 OpenGL version 保持一致
#version 330 core
layout (location = 0) in vec3 aPos; // 位置变量的属性位置值为0

out vec4 vertexColor; // 为片段着色器指定一个颜色输出

void main()
{
    gl_Position = vec4(aPos, 1.0); // 注意我们如何把一个 vec3 作为 vec4 的构造器的参数
    vertexColor = vec4(1.0, 0.0, 1.0, 1.0); // 把输出变量设置为暗红色
}
