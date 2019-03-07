cmake_minimum_required(VERSION 2.8)

project(helloworld)

# Specify the version if you need
# find_package(Ceres 1.13.0 REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# helloworld
add_executable(helloworld helloworld.cc)
target_link_libraries(helloworld ${CERES_LIBRARIES})