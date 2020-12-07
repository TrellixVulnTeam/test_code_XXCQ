cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME opengl_transformation)
project(${PROJECT_NAME})

set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 使用 GLFW 很简单，无需再用 include_directories 添加头文件路径了
find_package( glfw3 REQUIRED )
find_package( Eigen3 REQUIRED )

include_directories( ${EIGEN3_INCLUDE_DIR})

# 这是将 shader 文件拷贝到可执行文件路径中，否则在运行时会找不到它们
configure_file(${CMAKE_SOURCE_DIR}/shader.vert ${CMAKE_BINARY_DIR}/shader.vert COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/shader.frag ${CMAKE_BINARY_DIR}/shader.frag COPYONLY)

# 本工程使用的是 GLAD，因此记得加上 glad.c 这个文件到你的工程中
# （glad 的头文件已经提前放在 /usr/local/include 路径下了）
add_executable( ${PROJECT_NAME} 
    main.cpp 
    stb_image_wrap.cpp # 这个文件是为了编译 std_image.h 中的文件而加的
    ../glad/glad.c
)

# 这里额外增加一个 libdl.so 的路径，否则编译时有时会出现 glad.c 相关的一个错误。也可以直接用 -ldl
target_link_libraries( ${PROJECT_NAME} glfw ${CMAKE_DL_LIBS})