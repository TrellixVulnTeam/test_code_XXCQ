cmake_minimum_required(VERSION 3.5)

SET(PROJECT_NAME use_sophus)
project(${PROJECT_NAME})

# 设置FindPackage.cmake的路径。这里不用设是因为已经把FindSophus.cmake放入cmake的modules路径中了。
# set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package( Eigen3 REQUIRED )
find_package( Sophus REQUIRED )

message(STATUS "Eigen DIR: " ${EIGEN3_INCLUDE_DIR})
message(STATUS "Sophus DIR: " ${SOPHUS_INCLUDE_DIR})

include_directories(${EIGEN3_INCLUDE_DIR} ${SOPHUS_INCLUDE_DIR})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries( ${PROJECT_NAME} ${SOPHUS_LIBRARY} )