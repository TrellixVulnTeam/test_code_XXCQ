cmake_minimum_required(VERSION 3.1.0)

project(opencv_test)

# Use opencv 4.0. This is the directory where 'OpenCVConfig.cmake' file is located.
# set(OpenCV_DIR /home/chao/dev/opencv-4.0-build/lib/cmake/opencv4)
set(OpenCV_DIR /home/chao/dev/opencv-3.4.3_build)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(3d_mat)
add_subdirectory(camera_calibration)
add_subdirectory(image_moments)
add_subdirectory(gen_pure_teximages)
add_subdirectory(image_basic)
add_subdirectory(openni_test)
add_subdirectory(image_rotation)
add_subdirectory(multi_thread)
add_subdirectory(encode_decode)
