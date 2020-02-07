cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME g2o_vo)

project(${PROJECT_NAME})

set( CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall" )

# 设定Find模块的目录给接下来FIND_PACKAGE()命令，即FindLibrary.cmake所在的位置
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(G2O REQUIRED)

# message(STATUS "G2O DIR: "${G2O_INCLUDE_DIR})
# message(STATUS "G2O core: " ${G2O_CORE_LIBRARY})
# message(STATUS "G2O stuff: " ${G2O_STUFF_LIBRARY})
# message(STATUS "G2O sba: " ${G2O_TYPES_SBA})
# message(STATUS "G2O csparse: " ${G2O_SOLVER_CSPARSE})

include_directories(${EIGEN3_INCLUDE_DIR} ${G2O_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS} 
    ${G2O_LIBRARIES}
)


    