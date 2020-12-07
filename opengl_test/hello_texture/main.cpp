#include "shader.h"
#include <GLFW/glfw3.h>
#include "stb_image.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void keyPressCallBack(GLFWwindow* window, int key, int scancode, int action, int mods);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
bool g_flagShowWireFrame = false;
const float kTextureRatioInterval = 0.1;
float textureRatio = 0.5;

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

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // -------------------------------------------------------------------------------
    // 想要绘制两个三角形组成的矩形。这次每个顶点包含了三维坐标、颜色值和纹理坐标三个属性。一个顶点共有 8 floats。
    // float vertices[] = {
    //     // positions          // colors           // texture coords
    //     0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,    // top right
    //     0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,   // bottom right
    //     -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom left
    //     -0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f    // top left
    // };
    float vertices[] = {
        // positions          // colors           // texture coords
        0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 2.0f,    // top right
        0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f,   // bottom right
        -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom left
        -0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 2.0f    // top left
    };
    // 两个三角形，其 indices 还是老样子
    unsigned int indices[] = {
        0, 1, 3,  // first triangle
        1, 2, 3   // second triangle
    };

    // 创建 VBO, EBO, VAO 这些代码都是老样子
    // -------------------------------------------------------------------------------
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

    // 设置顶点的第一个属性：三维坐标。注意倒数第二个参数是 stride，此时一个顶点共有 8 floats。
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);  // 参数就是上面函数的第一个参数
    // 设置顶点的第二个属性：颜色值。注意参数的变化。
    // 第一个参数是属性的 index，最后一个是 offset，即该属性在整个顶点中的起始位置。
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // 设置顶点的第三个属性：纹理坐标。一个纹理坐标是 2 floats。
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // 解绑 VAO 还是有点用的，好处是可以避免其他的 VAO 修改它，如果只有一个 VAO 就无所谓了。
    glBindVertexArray(0);

    // 定义纹理
    // -------------------------------------------------------------------------------
    unsigned int textureID1, textureID2;
    // 第一个纹理
    glGenTextures(1, &textureID1);
    glBindTexture(GL_TEXTURE_2D, textureID1);
    // 一个纹理通常需要设置两种参数。第一是是两个轴向的 wrapping，即纹理坐标越界时的环绕方式）
    glTexParameteri(
        GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // 默认就是 REPEAT 类型，表明会重复范围内的纹理
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // 第二是设置当纹理被放大和缩小时候的 Filter 方式。
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // 纹理被缩小（通常是指物体被拉远了）
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  // 纹理被放大（通常是指物体被拉近了）
    // 读入纹理图片、分配空间以及生产 mipmap
    int width;
    int height;
    int nrChannels;
    // 这个函数是通知 stb_image.h，提前定义说要在垂直方向上翻转图片（这是为了和纹理坐标系保持一致）。不过这似乎是一个
    // 全局的 flag，因此定义一次就行了，后面载入第二张图片时也会默认翻转图片。
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load("../container.jpg", &width, &height, &nrChannels, 0);
    if (data)
    {
        // 在 GPU 上分配一块空间供纹理图片使用。
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    /// 接下来是第二个纹理
    glGenTextures(1, &textureID2);
    glBindTexture(GL_TEXTURE_2D, textureID2);
    // 这几个参数设置和上面是一样的
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    data = stbi_load("../awesomeface.png", &width, &height, &nrChannels, 0);
    if (data)
    {
        // note that the awesomeface.png has transparency and thus an alpha channel, so make sure to tell OpenGL the
        // data type is of GL_RGBA
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    // 使用自定义的 Shader 类来管理 shaders。
    // -------------------------------------------------------------------------------
    Shader myshader;
    if (!myshader.loadShaders("shader.vert", "shader.frag"))
    {
        std::cout << "ERROR in loading shaders. Quitting..." << std::endl;
        return -1;
    }
    // 这里开启使用 shader program 一次，为了设置刚刚定义的纹理和 Fragment Shader 中的纹理对象的对应关系。
    myshader.useProgram();
    // 这里的 texture1 和 texture2 是 Fragment Shader 中的变量名（必须完全 match）
    myshader.setInt("texture1", 0);
    myshader.setInt("texture2", 1);

    // 调用监视键盘输入的回调函数
    glfwSetKeyCallback(window, keyPressCallBack);

    // render loop
    // -------------------------------------------------------------------------------
    do
    {
        // render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // 绑定两个纹理（即同时使用两个纹理）
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, textureID2);

        myshader.useProgram();
        myshader.setFloat("textureRatio", textureRatio);

        /// 设置一个旋转矩阵，旋转角度随着时间不断增大，使得物体能够始终保持旋转。
        Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
        // glfwGetTime() 函数获取的是从启动 glfw 程序开始到当前的时间。
        Eigen::AngleAxisf rotationAxis((float)glfwGetTime(), Eigen::Vector3f(0, 0, 1));
        T.rotate(rotationAxis);
        T.translate ( Eigen::Vector3f (1.0, 0.0, 0.0) ); // 这个是增加一个旋转
        myshader.setMatrix4f("transform", T.matrix());

        glBindVertexArray(VAO);  // 绘制前，绑定相应的 VAO
        // 这是不用 EBO 时使用的绘制三角形的命令，不过此时输入三角形的顶点一般是有重复的。最后一个参数是顶点总个数。
        // glDrawArrays(GL_TRIANGLES, 0, 6);

         // 使用这个 VAO 中的 EBO 绘制时所用的函数。第二个参数是要绘制的顶点个数。最后一个参数是 offset 值，不过这里就是 0。
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0); 

        glfwSwapBuffers(window);  // 双缓冲方法：交换后缓冲区和前缓冲区
        glfwPollEvents();         // 检查各种触发事件，例如键盘鼠标等
    } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwGetKey(window, GLFW_KEY_Q) != GLFW_PRESS);

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

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

    // 测试一个简单的纹理效果：通过上下键来修改 GLSL 中的一个 uniform 变量，该变量用于平衡两个纹理图片所占的比例
    if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
    {
        textureRatio -= kTextureRatioInterval;
        if (textureRatio < 0)
            textureRatio = 0;
    }
    if (key == GLFW_KEY_UP && action == GLFW_PRESS)
    {
        textureRatio += kTextureRatioInterval;
        if (textureRatio > 1)
            textureRatio = 1;
    }
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