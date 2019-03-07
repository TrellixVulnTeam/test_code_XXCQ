cmake_minimum_required(VERSION 2.8.12)

project(image_moments)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include)

link_directories(/usr/lib/x86_64-linux-gnu)

add_executable(image_moments moments_demo.cpp)
target_link_libraries(image_moments ${OpenCV_LIBS} gflags)

