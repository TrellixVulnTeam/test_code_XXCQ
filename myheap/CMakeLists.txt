cmake_minimum_required(VERSION 3.1.0)

project(heap_test)

# set(Boost_INCLUDE_DIR /usr/local/include)
# set(Boost_LIBRARY_DIR /usr/local/lib)
# find_package(Threads REQUIRED)
# include_directories(${Boost_INCLUDE_DIR})
# link_directories(${Boost_LIBRARY_DIR})

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME} main.cpp)
