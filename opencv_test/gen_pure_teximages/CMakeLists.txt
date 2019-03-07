cmake_minimum_required(VERSION 2.8.12)

project(gen_pure_teximages)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include)

link_directories(/usr/lib/x86_64-linux-gnu)

add_executable(gen_pure_teximages gen_pure_teximages.cpp)
target_link_libraries(gen_pure_teximages ${OpenCV_LIBS} gflags)

