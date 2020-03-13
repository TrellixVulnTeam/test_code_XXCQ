#pragma once

#include <glad/glad.h>  // 包含 glad 来获取所有的必须 OpenGL 头文件（用 glew 当然也可以）
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Eigen>

/// 自定义的 Shader 类，用于读取并操作 Vertex Shader 和 Fragment Shader
class Shader
{
public:
    unsigned int m_programID;  // 整个 Shader 类就一个成员变量，就是 shader program ID

    // 将读取两个 shaders 放入构造函数中
    Shader() {}

    /// 读取 Shader 文件并返回读取的字符串
    std::string readShaderFile(const char *shaderFilename)
    {
        std::string shaderString;
        std::ifstream shaderFile;
        // 保证可以抛出异常
        shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        try
        {
            // 读取 Vertex Shader 文件到数据流，然后再转换成 string
            shaderFile.open(shaderFilename);
            std::stringstream shaderStream;
            shaderStream << shaderFile.rdbuf();
            shaderFile.close();
            shaderString = shaderStream.str();
        }
        catch (std::ifstream::failure e)
        {
            std::cout << "ERROR: Failed to read shader file " << shaderFilename << std::endl;
        }
        return shaderString;
    }

    /// 读取、编译和链接两个 shaders 并创建 shader program
    bool loadShaders(const char *vertexShaderFilename, const char *fragmentShaderFilename)
    {
        int infoLogLength;
        int success;

        /// 读取和编译 Vertex Shader
        std::string vertexShaderString = readShaderFile(vertexShaderFilename);
        if (vertexShaderString.empty())
            return false;
        const char *vertexShaderStringPointer = vertexShaderString.c_str();
        int vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShaderID, 1, &vertexShaderStringPointer, NULL);
        glCompileShader(vertexShaderID);
        glGetShaderiv(vertexShaderID, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderiv(vertexShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
            std::vector<char> vertexShaderErrorMessage(infoLogLength + 1);
            glGetShaderInfoLog(vertexShaderID, infoLogLength, NULL, &vertexShaderErrorMessage[0]);
            std::cout << "ERROR: Failed to compile vertex shader. " << std::endl;
            for (char c : vertexShaderErrorMessage)
                std::cout << c;
            std::cout << std::endl;
            return false;
        }

        /// 读取和编译 Fragment Shader
        std::string fragmentShaderString = readShaderFile(fragmentShaderFilename);
        if (fragmentShaderString.empty())
            return false;
        int fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
        const char *fragmentSourceStringPointer = fragmentShaderString.c_str();
        glShaderSource(fragmentShaderID, 1, &fragmentSourceStringPointer, NULL);
        glCompileShader(fragmentShaderID);
        glGetShaderiv(fragmentShaderID, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderiv(fragmentShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
            std::vector<char> fragmentShaderErrorMessage(infoLogLength + 1);
            glGetShaderInfoLog(fragmentShaderID, infoLogLength, NULL, &fragmentShaderErrorMessage[0]);
            std::cout << "ERROR: Failed to compile fragment shader. " << std::endl;
            for (char c : fragmentShaderErrorMessage)
                std::cout << c;
            std::cout << std::endl;
            return false;
        }

        /// 链接到 shader program
        m_programID = glCreateProgram();
        glAttachShader(m_programID, vertexShaderID);
        glAttachShader(m_programID, fragmentShaderID);
        glLinkProgram(m_programID);
        // 检查链接是否成功
        glGetProgramiv(m_programID, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramiv(m_programID, GL_INFO_LOG_LENGTH, &infoLogLength);
            std::vector<char> programErrorMessage(infoLogLength + 1);
            glGetProgramInfoLog(m_programID, infoLogLength, NULL, &programErrorMessage[0]);
            std::cout << "ERROR: Failed to link shaders to program. " << std::endl;
            for (char c : programErrorMessage)
                std::cout << c;
            std::cout << std::endl;
            return false;
        }
        /// 删除 shaders，因为已经链接到程序中的 shaders 就不再需要了
        glDetachShader(m_programID, vertexShaderID);
        glDetachShader(m_programID, fragmentShaderID);
        glDeleteShader(vertexShaderID);
        glDeleteShader(fragmentShaderID);
        return true;
    }

    /// 使用当前的 shader program （其实就是简单封装了一下）
    void useProgram() const { glUseProgram(m_programID); }

    /// 删除当前的 shader program
    void deleteProgram() { glDeleteProgram(m_programID); }

    //-----------------------------------------------------------------------
    /// Utility functions: 一些修改 uniform 变量的函数

    /// Utility function: 设置 bool uniform 变量
    void setBool(const std::string &name, bool value) const
    {
        glUniform1i(glGetUniformLocation(m_programID, name.c_str()), (int)value);
    }
    /// Utility function: 设置 int uniform 变量
    void setInt(const std::string &name, int value) const
    {
        glUniform1i(glGetUniformLocation(m_programID, name.c_str()), value);
    }
    /// Utility function: 设置 float uniform 变量
    void setFloat(const std::string &name, float value) const
    {
        glUniform1f(glGetUniformLocation(m_programID, name.c_str()), value);
    }
    /// Utility function: 设置 vec2 uniform 变量
    void setVec2f(const std::string &name, const Eigen::Vector2f &value) const
    {
        glUniform2fv(glGetUniformLocation(m_programID, name.c_str()), 1, &value[0]);
    }
    /// Utility function: 设置 vec2 uniform 变量
    void setVec2f(const std::string &name, float x, float y) const
    {
        glUniform2f(glGetUniformLocation(m_programID, name.c_str()), x, y);
    }
    /// Utility function: 设置 vec3 uniform 变量
    void setVec3f(const std::string &name, const Eigen::Vector3f &value) const
    {
        glUniform3fv(glGetUniformLocation(m_programID, name.c_str()), 1, &value[0]);
    }
    /// Utility function: 设置 vec3 uniform 变量
    void setVec3f(const std::string &name, float x, float y, float z) const
    {
        glUniform3f(glGetUniformLocation(m_programID, name.c_str()), x, y, z);
    }
    /// Utility function: 设置 vec4 uniform 变量
    void setVec4f(const std::string &name, const Eigen::Vector4f &value) const
    {
        glUniform4fv(glGetUniformLocation(m_programID, name.c_str()), 1, &value[0]);
    }
    /// Utility function: 设置 vec4 uniform 变量
    void setVec4f(const std::string &name, float x, float y, float z, float w) const
    {
        glUniform4f(glGetUniformLocation(m_programID, name.c_str()), x, y, z, w);
    }
    /// Utility function: 设置 Matrix2f uniform 变量
    void setMatrix2f(const std::string &name, const Eigen::Matrix2f &mat) const
    {
        // 这里的参数 GL_FALSE 是说要不要转置。而 Eigen 的矩阵就是 column-major，和 OpenGL
        // 中的相同，因此也就不用再转置了。
        glUniformMatrix2fv(glGetUniformLocation(m_programID, name.c_str()), 1, GL_FALSE, mat.data());
    }
    /// Utility function: 设置 Matrix3f uniform 变量
    void setMatrix3f(const std::string &name, const Eigen::Matrix3f &mat) const
    {
        // 这里的参数 GL_FALSE 是说要不要转置。而 Eigen 的矩阵就是 column-major，和 OpenGL
        // 中的相同，因此也就不用再转置了。
        glUniformMatrix3fv(glGetUniformLocation(m_programID, name.c_str()), 1, GL_FALSE, mat.data());
    }
    /// Utility function: 设置 Matrix4f uniform 变量
    void setMatrix4f(const std::string &name, const Eigen::Matrix4f &mat) const
    {
        // 这里的参数 GL_FALSE 是说要不要转置。而 Eigen 的矩阵就是 column-major，和 OpenGL
        // 中的相同，因此也就不用再转置了。
        glUniformMatrix4fv(glGetUniformLocation(m_programID, name.c_str()), 1, GL_FALSE, mat.data());
    }
};
