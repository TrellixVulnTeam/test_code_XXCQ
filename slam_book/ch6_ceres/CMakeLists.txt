cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME ceres_curve_fitting)

project(${PROJECT_NAME})

set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})