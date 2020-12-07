/*
    第二个 OpenGL 程序：绘制一个三角形
    参考教程：https://learnopengl-cn.github.io/01%20Getting%20started/04%20Hello%20Triangle/
*/

// 请确认是在包含GLFW的头文件之前包含了GLAD的头文件。
// GLAD的头文件包含了正确的OpenGL头文件（例如GL/gl.h），所以需要在其它依赖于OpenGL的头文件之前包含GLAD。
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void keyPressCallBack(GLFWwindow* window, int key, int scancode, int action, int mods);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
bool g_flagShowWireFrame = false;

// 两个 Shader，因为比较简单，就直接放在变量中了。更常见的做法是将 shader 放在单独的文件中，然后再读取文件。
// 这两个 shaders 基本没做事情。
const char* vertexShaderSource =
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos, 1.0);\n"
    "}\0";
const char* fragmentShaderSource =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n\0";

int main()
{
    // glfw: initialize and configure
    // -------------------------------------------------------------------------------
    glfwInit();
    // 指定你使用的 OpenGL 版本。有关 Mac 不同机型支持的 OpenGL 版本可以查看：https://support.apple.com/en-us/HT202823
    // Ubuntu/Linux 中可以用 `glxinfo | grep "OpenGL"` 命令来查看。注意也要和你下载的 glad 对应的 OpenGL version
    // 保持一致。
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 下面这行是 Mac 系统必须的
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // uncomment this statement to fix compilation on OS X
#endif

    // glfw window creation
    // -------------------------------------------------------------------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers before calling any OpenGL function
    // -------------------------------------------------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // build and compile our shader program
    // -------------------------------------------------------------------------------
    // vertex shader
    int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    int infoLogLength;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        // 如果 Shader 编译出错，错误信息
        glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &infoLogLength);  // 返回错误信息的长度
        std::vector<char> vertexShaderErrorMessage(infoLogLength + 1);
        glGetShaderInfoLog(vertexShader, 512, NULL, &vertexShaderErrorMessage[0]);
        printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED: %s\n", &vertexShaderErrorMessage[0]);
    }
    // fragment shader
    int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        // 和上面的“打印出错信息”的代码几乎一样
        glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<char> fragmentShaderErrorMessage(infoLogLength + 1);
        glGetShaderInfoLog(fragmentShader, 512, NULL, &fragmentShaderErrorMessage[0]);
        printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED: %s\n", &fragmentShaderErrorMessage[0]);
    }
    // link shaders
    int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetShaderiv(shaderProgram, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<char> shaderProgramErrorMessage(infoLogLength + 1);
        glGetProgramInfoLog(shaderProgram, 512, NULL, &shaderProgramErrorMessage[0]);
        printf("ERROR::SHADER::PROGRAM::LINKING_FAILED: %s\n", &shaderProgramErrorMessage[0]);
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // -------------------------------------------------------------------------------
    // 想要绘制两个三角形组成的矩形
    float vertices[] = {
        0.5f, 0.5f, 0.0f,    // top right
        0.5f, -0.5f, 0.0f,   // bottom right
        -0.5f, -0.5f, 0.0f,  // bottom left
        -0.5f, 0.5f, 0.0f    // top left
    };
    unsigned int indices[] = {
        // note that we start from 0!
        0, 1, 3,  // first Triangle
        1, 2, 3   // second Triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);  // 最开始肯定是先定义这几个 objects
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);  // 必须先绑定 VAO，此后的所有的 VBO 和 EBO 都会在这个 VAO 中了

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound
    // vertex buffer object so afterwards we can safely unbind。
    // 解绑 VBO 是可以的，不过其实无所谓。解绑必须要在 glVertexAttribPointer() 调用之后。解绑的参数都是设置为
    // 0。换句话说，生成 的 VAO, VBO, EBO 等对象都不是 0（通常是从 1 开始的）。
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO;
    // keep the EBO bound。一定不要在解绑 VAO 之前解绑 EBO。所以干脆就别解绑 EBO 了。
    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens.
    // Modifying other VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs)
    // when it's not directly necessary。
    // 解绑 VAO 还是有点用的，好处是可以避免其他的 VAO 修改它，如果只有一个 VAO 就无所谓了。
    glBindVertexArray(0);

    // 调用监视键盘输入的回调函数
    glfwSetKeyCallback(window, keyPressCallBack);

    // render loop
    // -------------------------------------------------------------------------------
    do
    {
        // render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // draw our first triangle
        glUseProgram(shaderProgram);
        // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a
        // bit more organized
        glBindVertexArray(VAO);  // 绘制前，绑定相应的 VAO
        // glDrawArrays(GL_TRIANGLES, 0, 6);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);  // 使用这个 VAO 中的 EBO 绘制时所用的函数
        // glBindVertexArray(0); // no need to unbind it every time

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);  // 双缓冲方法：交换后缓冲区和前缓冲区
        glfwPollEvents();         // 检查各种触发事件，例如键盘鼠标等
    } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwGetKey(window, GLFW_KEY_Q) != GLFW_PRESS);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

/// 定义 keyboard press 的回调函数（而不是将判断键盘输入放在主函数的循环中），可以更好的监视键盘输入，
/// 一个好处是可以避免重复多次的键盘输入来不及响应。该函数的输入参数是固定的。
void keyPressCallBack(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_W && action == GLFW_PRESS)  // 必须加上判断该按键是按下的，否则的话，按下和抬起都在考虑之类
        g_flagShowWireFrame = !g_flagShowWireFrame;
    if (g_flagShowWireFrame)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

/// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays (即，通常这里的宽度和高度
    // 要明显大于窗口大小)
    // glViewport() 用于将 normalized device coordinates 坐标（它永远会在 [-1, 1] 之间）映射为视口
    // 中的坐标。例如，假设窗口大小是 (800, 600)，那么 [-1, 1] 的范围就会被映射到 [0, 800] 和 [0. 600] 的屏幕
    // 坐标范围。
    glViewport(0, 0, width, height);
}