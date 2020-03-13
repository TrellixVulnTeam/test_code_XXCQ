#version 410 core
out vec4 FragColor;

in vec4 vertexColor; // 从 Vertex Shader 的输出得到的颜色值作为输入

void main()
{
    FragColor = vertexColor; // 就是简单的赋值
}
