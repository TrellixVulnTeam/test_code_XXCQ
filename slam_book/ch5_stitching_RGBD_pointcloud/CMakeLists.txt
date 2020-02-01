cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(PROJECT_NAME stitching_pointcloud)
project(${PROJECT_NAME})

set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( PCL REQUIRED COMPONENTS common io )

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS} )

add_definitions( ${PCL_DEFINITIONS} )

add_executable( ${PROJECT_NAME} main.cpp )

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} )