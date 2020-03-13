#include "shader.h"
#include <GLFW/glfw3.h>
#include "stb_image.h"
#include "camera.h"

void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void keyPressCallBack(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseCursorCallBack(GLFWwindow * window, double xpos, double ypos);
void mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void processInputFast(GLFWwindow *window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
const float kTextureRatioInterval = 0.1;
float g_textureRatio = 0.5;

// Camera
Camera g_myCamera;
float g_lastX = SCR_WIDTH / 2.0f;
float g_lastY = SCR_HEIGHT / 2.0f;
bool g_leftMouseDownFirstTime = true; // 第一次按下鼠标时，系统没有上一次鼠标位置的记录，因此要特别区分
float g_deltaTime = 0.0f;	// time between current frame and last frame
float g_lastFrame = 0.0f;

int main()
{
    // glfw: initialize and configure
    // -------------------------------------------------------------------------------
    glfwInit();
    // 指定你使用的 OpenGL 版本。有关 Mac 不同机型支持的 OpenGL 版本可以查看：https://support.apple.com/en-us/HT202823
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
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
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    // glad: load all OpenGL function pointers before calling any OpenGL function
    // -------------------------------------------------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // -------------------------------------------------------------------------------
    // 这次要绘制一个 Cube。每个顶点包含了三维坐标和纹理坐标三个属性。一个顶点共有 8 floats。注意这里每个平面
    // 都对应有纹理，即每个顶点会有多个纹理坐标，故此时就暂时不用 EBO 了，省的麻烦。
    float vertices[] = {
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 
        0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f, 
        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f,

        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 
        0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 
        0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 
        -0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f,

        -0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 1.0f, 1.0f, 
        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 
        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 
        -0.5f, 0.5f, 0.5f, 1.0f, 0.0f,

        0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f, 
        0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 
        0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 
        0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 
        0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 
        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f,

        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f, 
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 
        -0.5f, 0.5f, 0.5f, 0.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f
    };

    Eigen::Vector3f cubePositions[] = {
        Eigen::Vector3f( 0.0f,  0.0f,  0.0f),
        Eigen::Vector3f( 2.0f,  5.0f, -15.0f),
        Eigen::Vector3f(-1.5f, -2.2f, -2.5f),
        Eigen::Vector3f(-3.8f, -2.0f, -12.3f),
        Eigen::Vector3f( 2.4f, -0.4f, -3.5f),
        Eigen::Vector3f(-1.7f,  3.0f, -7.5f),
        Eigen::Vector3f( 1.3f, -2.0f, -2.5f),
        Eigen::Vector3f( 1.5f,  2.0f, -2.5f),
        Eigen::Vector3f( 1.5f,  0.2f, -1.5f),
        Eigen::Vector3f(-1.3f,  1.0f, -1.5f)
    };
    const int kNumTriangles = sizeof(vertices) / (sizeof(float) * 5);
    const int kNumCubes = sizeof(cubePositions) / sizeof(Eigen::Vector3f);

    // 创建 VBO, VAO 这些代码都是老样子
    // -------------------------------------------------------------------------------
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);  // 最开始肯定是先定义这几个 objects
    glGenBuffers(1, &VBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);  // 必须先绑定 VAO，此后的所有的 VBO 和 EBO 都会在这个 VAO 中了

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // 设置顶点的第一个属性：三维坐标。注意倒数第二个参数是 stride，此时一个顶点共有 8 floats。
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);  // 参数就是上面函数的第一个参数
    // 设置顶点的第二个属性：纹理坐标。这是 2 floats。注意参数的变化。
    // 第一个参数是属性的 index，最后一个是 offset，即该属性在整个顶点中的起始位置。
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // 解绑 VAO 还是有点用的，好处是可以避免其他的 VAO 修改它，如果只有一个 VAO 就无所谓了。
    glBindVertexArray(0);

    // 定义纹理
    // -------------------------------------------------------------------------------
    unsigned int textureID1, textureID2;
    // 第一个纹理
    glGenTextures(1, &textureID1);
    glBindTexture(GL_TEXTURE_2D, textureID1);
    // 一个纹理通常需要设置两种参数。第一是是两个轴向的 wrapping，即纹理坐标越界时的环绕方式）
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // 默认就是 REPEAT 类型，重复界限内的纹理
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
    // ----------------------------------------------------------------
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

    // 监视输入的回调函数
    // ----------------------------------------------------------------
    glfwSetKeyCallback(window, keyPressCallBack);
    glfwSetCursorPosCallback(window, mouseCursorCallBack);
    glfwSetScrollCallback(window, mouseScrollCallback);

    // render loop
    // -------------------------------------------------------------------------------
    glEnable(GL_DEPTH_TEST);
    do
    {
        // per-frame time logic
        float currentFrame = glfwGetTime();
        g_deltaTime = currentFrame - g_lastFrame;
        g_lastFrame = currentFrame;

        processInputFast(window);

        // render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 绑定两个纹理（即同时使用两个纹理）
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, textureID2);

        myshader.useProgram();
        myshader.setFloat("textureRatio", g_textureRatio);

        /// 接下来是计算 OpenGL Transformation 并绘制模型
        // ------------------------------------------------------
        glBindVertexArray(VAO);  // 绘制前，绑定相应的 VAO

        // 这里的 camera matrix 已经包含了 projection, camera view 以及通过键盘或者鼠标输入带来的变换矩阵。
        Eigen::Matrix4f cameraViewMatrix = g_myCamera.getCameraViewMatrix();
        Eigen::Matrix4f projectionMatrix = g_myCamera.getProjectionMatrix();
        // 这里我们想在不同位置绘制多个一样的 cube，不过其实只需要使用一个 VAO 即可
        for (int i = 0; i < kNumCubes; i++)
        {
            // model matrix
            Eigen::Isometry3f modelMatrixIso = Eigen::Isometry3f::Identity();
            modelMatrixIso.pretranslate(cubePositions[i]); // 增加平移
            modelMatrixIso.rotate(Eigen::AngleAxisf(20.0f * i, Eigen::Vector3f(1.0f, 0.3f, 0.5f))); // 增加旋转

            // 最终的转移矩阵是多个矩阵相乘
            Eigen::Matrix4f finalTransformationMatrix = projectionMatrix * cameraViewMatrix * modelMatrixIso.matrix();
            myshader.setMatrix4f("transform", finalTransformationMatrix);
            // 投影矩阵这里是固定的，因此提前计算出来放在这里

            // 本问题中没有用 EBO，因此用如下命令绘制三角形的命令，不过此时输入三角形的顶点必须是有重复的。
            // 最后一个参数是顶点总个数。
            glDrawArrays(GL_TRIANGLES, 0, kNumTriangles);
        }

        glfwSwapBuffers(window);  // 双缓冲方法：交换后缓冲区和前缓冲区
        glfwPollEvents();         // 检查各种触发事件，例如键盘鼠标等
    } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwGetKey(window, GLFW_KEY_Q) != GLFW_PRESS);

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

/// 定义 keyboard press 的回调函数（而不是将判断键盘输入放在主函数的循环中）。
/// 它的好处是，可以更好的处理一个完整的键盘输入，可以避免按一次按键触发多次响应。即，它的原理似乎是，处理完
/// 一次键盘响应后就会歇一下隔一点点时间再处理下一次键盘响应。因此，它更适用于我们想要按一次按键处理一件事情后马上停止，
/// 例如切换某种状态、打开或者关闭某个开关等等。该函数的输入参数是固定的。
void keyPressCallBack(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // 测试一个简单的纹理效果：通过上下键来修改 GLSL 中的一个 uniform 变量，该变量用于平衡两个纹理图片所占的比例
    if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
    {
        g_textureRatio -= kTextureRatioInterval;
        if (g_textureRatio < 0)
            g_textureRatio = 0;
    }
    if (key == GLFW_KEY_UP && action == GLFW_PRESS)
    {
        g_textureRatio += kTextureRatioInterval;
        if (g_textureRatio > 1)
            g_textureRatio = 1;
    }
}

/// 监视键盘和鼠标输入。注意这并非回调函数，而是直接放在主循环中的。它的优点是处理速度快，因此更适合处理某种需要
/// 连续显示的事情，例如平移物体等。
void processInputFast(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        g_myCamera.processKeyboardMovement(Camera::ModelViewMode::MOVE_FORWARD, g_deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        g_myCamera.processKeyboardMovement(Camera::ModelViewMode::MOVE_BACKWARD, g_deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        g_myCamera.processKeyboardMovement(Camera::ModelViewMode::MOVE_LEFT, g_deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        g_myCamera.processKeyboardMovement(Camera::ModelViewMode::MOVE_RIGHT, g_deltaTime);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouseCursorCallBack(GLFWwindow * window, double xpos, double ypos)
{
    if (glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS &&
        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        g_myCamera.setModelViewMode(Camera::ModelViewMode::TRANSLATE);
    }
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) 
    {
        g_myCamera.setModelViewMode(Camera::ModelViewMode::ROTATE);
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) 
    {
        g_leftMouseDownFirstTime = true; // 鼠标左键抬起后，将它 reset，为了下一次鼠标按下时使用
        g_myCamera.setModelViewMode(Camera::ModelViewMode::NOTHING);
    }
    Camera::ModelViewMode currMode = g_myCamera.getModelViewMode();
    if (currMode == Camera::ModelViewMode::TRANSLATE || currMode == Camera::ModelViewMode::ROTATE)
    {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);  // get current cursor position
        if (g_leftMouseDownFirstTime)
        {
            // 第一次按下鼠标时，并没有上一次的鼠标位置的记录，故此时要区别对待
            g_lastX = xpos;
            g_lastY = ypos;
            g_leftMouseDownFirstTime = false;
        }
        float xoffset = xpos - g_lastX;
        float yoffset = ypos - g_lastY;  // GLFW 窗口中的 Y-coordinate（纵向）是从底向上增大的
        g_lastX = xpos;
        g_lastY = ypos;
        if (currMode == Camera::ModelViewMode::TRANSLATE)
        {
            g_myCamera.processMouseTranslation(xoffset, yoffset);
        }
        else
        {
            g_myCamera.processMouseRotation(xoffset, yoffset);
        }
    }
}

void mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    g_myCamera.processMouseScroll(yoffset);
}

/// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays (即，通常这里的宽度和高度
    // 要明显大于窗口大小)
    // glViewport() 用于将 normalized device coordinates 坐标（它永远会在 [-1, 1] 之间）映射为视口
    // 中的坐标。例如，假设窗口大小是 (800, 600)，那么 [-1, 1] 的范围就会被映射到 [0, 800] 和 [0. 600] 的屏幕
    // 坐标范围。
    glViewport(0, 0, width, height);
}