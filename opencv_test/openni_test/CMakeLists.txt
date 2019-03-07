cmake_minimum_required(VERSION 2.8.12)

project(videocapture_openni)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include)

link_directories(/usr/lib/x86_64-linux-gnu)

add_executable(videocapture_openni videocapture_openni.cpp)
target_link_libraries(videocapture_openni ${OpenCV_LIBS} gflags)

