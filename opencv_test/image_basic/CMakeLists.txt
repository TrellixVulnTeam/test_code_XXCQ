cmake_minimum_required(VERSION 2.8.12)

project(image_basic)

# Use opencv 4.0
set(OpenCV_DIR /home/chao/dev/opencv-4.0-build/lib/cmake/opencv4)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(image_basic image_basic.cpp)

target_link_libraries(image_basic ${OpenCV_LIBS})
