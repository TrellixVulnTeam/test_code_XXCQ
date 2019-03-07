cmake_minimum_required(VERSION 2.8.12)

project(camera_calibration)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include)

link_directories(/usr/lib/x86_64-linux-gnu)

add_executable(camera_calibration camera_calibration.cpp)
target_link_libraries(camera_calibration ${OpenCV_LIBS} gflags)

