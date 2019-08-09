cmake_minimum_required(VERSION 3.1.0)

project(opencv_test)

# Use opencv 4.0
set(OpenCV_DIR /home/chao/dev/opencv-4.0-build/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(3d_mat)
add_subdirectory(camera_calibration)
add_subdirectory(image_moments)
add_subdirectory(gen_pure_teximages)
add_subdirectory(image_basic)
add_subdirectory(openni_test)
add_subdirectory(image_rotation)
