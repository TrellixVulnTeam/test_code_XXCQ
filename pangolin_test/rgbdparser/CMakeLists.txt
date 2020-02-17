cmake_minimum_required(VERSION 3.1.0)

set(PROJECT_NAME rgbd_parser)

# 记得设置支持 C++ 14，否则 pangolin 程序编译会出错 
set (CMAKE_CXX_STANDARD 14)

project(${PROJECT_NAME})

find_package(OpenCV REQUIRED)
find_package(Pangolin 0.4 REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} rgbdparser.cpp)
target_link_libraries(${PROJECT_NAME} ${Pangolin_LIBRARIES} ${OpenCV_LIBS})