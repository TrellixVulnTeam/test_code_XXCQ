
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(PROJECT_NAME passthrough)

project(${PROJECT_NAME})

find_package(PCL 1.8 REQUIRED COMPONENTS io common filters)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (${PROJECT_NAME} main.cpp)
target_link_libraries (${PROJECT_NAME} ${PCL_LIBRARIES})
