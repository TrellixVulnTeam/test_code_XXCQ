cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME g2o_curve_fitting)

project(${PROJECT_NAME})

set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
# set( CMAKE_CXX_FLAGS "-std=c++11 -O3 -fsanitize=address" )
# set( CMAKE_CXX_FLAGS "-std=c++11 -O3 -fsanitize=leak" )

# message(STATUS "CMAKE_CXX_FLAGS: "${CMAKE_CXX_FLAGS})
# message(STATUS "CMAKE_BUILD_TYPE: "${CMAKE_BUILD_TYPE})

# 设定Find模块的目录给接下来FIND_PACKAGE()命令，即FindLibrary.cmake所在的位置
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules ${PROJECT_SOURCE_DIR}/cmake_modules_sanitizers)

find_package(Eigen3 REQUIRED)
find_package(G2O REQUIRED)
# find_package(Sanitizers)

# message(STATUS "G2O DIR: "${G2O_INCLUDE_DIR})
# message(STATUS "G2O Library: "${G2O_CORE_LIBRARY} " and " ${G2O_STUFF_LIBRARY})

include_directories(${EIGEN3_INCLUDE_DIR} ${G2O_INCLUDE_DIR} )

add_executable(${PROJECT_NAME} main.cpp)
# add_sanitizers(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
