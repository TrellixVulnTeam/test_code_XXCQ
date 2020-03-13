cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME hello_window)
project(${PROJECT_NAME})

# set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 使用 GLFW 很简单，无需再用 include_directories 添加头文件路径了
find_package( glfw3 REQUIRED )

# 本工程使用的是 GLAD，因此记得加上 glad.c 这个文件到你的工程中
# （glad 的头文件已经提前放在 /usr/local/include 路径下了）
add_executable( ${PROJECT_NAME} main.cpp ../glad/glad.c)

target_link_libraries( ${PROJECT_NAME} glfw)