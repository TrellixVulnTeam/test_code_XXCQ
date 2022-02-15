cmake_minimum_required(VERSION 2.8.12)

project(openmp_test)

# if (CMAKE_COMPILER_IS_GNUCXX)
# 	set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -msse4.1")
# endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)


if(APPLE)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

endif()

find_package(OpenMP REQUIRED)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

link_directories(/usr/local/opt/libomp/lib)

add_executable(openmp_test main.cpp)
target_link_libraries(openmp_test OpenMP::OpenMP_CXX)
