# Opengl 工程配置的一些说明

## 确定你的显卡的 OpenGL 版本

首先一定要做的是，确定你的显卡支持的 OpenGL 版本。这个版本没法改，只要安装好了显卡驱动，这个版本就固定了。接下来的 GLAD 的配置以及 GLFW 函数中指定的 OpenGL 版本号必须都和这个保持一致。

有关 Mac 不同机型支持的 OpenGL 版本可以查看：https://support.apple.com/en-us/HT202823.

Ubuntu/Linux 中可以用 `glxinfo | grep "OpenGL"` 命令来查看。你会得到类似这样：
```bash
OpenGL vendor string: VMware, Inc.
OpenGL renderer string: llvmpipe (LLVM 10.0.0, 256 bits)
OpenGL core profile version string: 3.3 (Core Profile) Mesa 20.0.8
OpenGL core profile shading language version string: 3.30
OpenGL core profile context flags: (none)
OpenGL core profile profile mask: core profile
OpenGL core profile extensions:
OpenGL version string: 3.1 Mesa 20.0.8
OpenGL shading language version string: 1.40
OpenGL context flags: (none)
OpenGL extensions:
OpenGL ES profile version string: OpenGL ES 3.1 Mesa 20.0.8
OpenGL ES profile shading language version string: OpenGL ES GLSL ES 3.10
OpenGL ES profile extensions:
```
其中那个 OpenGL core profile version string 3.3 就是我们需要的 OpenGL Core 版本。下面那个 "OpenGL version string: 3.1" 我个人也不清楚，似乎是 non-core version 和 Core version 的区别。不过经测试，我的机器上可以运行 3.3 版本。

## GLAD 的配置

GLAD 下载并且配置到自己的工程中的整个流程：
- 进入 [GLAD 下载网站](https://glad.dav1d.de/)，在 API -> gl 中选择**你的机器对应的 OpenGL 版本**（上一步中介绍的，例如3.3），右边的 Profile 选择 Core。下面的 Options 中勾选 Generate a loader。然后点击 Generate 按钮，它会生成一个 `glad.zip` 压缩包。下载到本地并解压。
- 解压后，应该有一个 include 文件夹（包括两个文件： `glad/glad.h` 和 `KHR/khrplatform.h`）和一个 `src/glad.c` 文件。将 include 文件夹中的**全部内容**（即两个子文件夹）直接拷贝到你的工程中的 include 路径下，也可以干脆拷贝到一个系统的 include 路径，例如 `/usr/local/include`中。接着，将那个 glad.c 文件添加到你的工程的编译目录中。即，添加到 CMakeLists.txt 中。我的做法是，将这个文件拷贝到自己工程目录下的 `glad/glad.c` 中，然后在 CMakeLists.txt 中把它加上，类似：
```cmake
add_executable( ${PROJECT_NAME} XXX.cpp ../glad/glad.c)
```
这样就完成了配置。


## GLFW 中设置 OpenGL context 版本

注意必须和你的 OpenGL 版本以及 GLAD 对应的版本保持一致。

```cpp
    // 指定你使用的 OpenGL 版本。有关 Mac 不同机型支持的 OpenGL 版本可以查看：https://support.apple.com/en-us/HT202823.
    // Ubuntu/Linux 中可以用 `glxinfo | grep "OpenGL"` 命令来查看。注意也要和你下载的 glad 对应的 OpenGL version 保持一致。
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
```

另外，OpenGL Shader 中的 Context Version 也必须和你的机器的 OpenGL 版本保持一致，通常在 shader 文件的开头：

```cpp
#version 330 core
...
```