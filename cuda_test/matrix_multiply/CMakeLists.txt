CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT(CUDA_Test)

FIND_PACKAGE(CUDA)

# Add CUBLAS path otherwise the library cannot be found during target link
link_directories(/usr/local/cuda/lib64)
# Find other libraries (Such as OpenCV)
# set(OpenCV_DIR /home/chao/dev/opencv-4.0-build/lib/cmake/opencv4)
# FIND_PACKAGE(OpenCV)

# print some path
MESSAGE(STATUS "current path: ${CMAKE_CURRENT_SOURCE_DIR}")

# Find all cpp and cu files recursively if needed
# FILE(GLOB_RECURSE CPP_LIST *.cpp)
# FILE(GLOB_RECURSE CU_LIST *.cu)
# MESSAGE(STATUS "cpp list: ${CPP_LIST}")
# MESSAGE(STATUS "cu list: ${CU_LIST}")

# Support C++11
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu++0x")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Include libraries if needed
# INCLUDE_DIRECTORIES(/usr/local/cuda-8.0/include)
# INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
# MESSAGE(STATUS "opencv include directory: ${OpenCV_INCLUDE_DIRS}")

# Support C++11 for CUDA NVCC
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11;-O2)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)
# Compute ability
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
# LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)

# CUDA_ADD_EXECUTABLE(CUDA_Test ${CPP_LIST} ${CU_LIST})
CUDA_ADD_EXECUTABLE(CUDA_Test matrix_mul_cublas.cu)

# Link other libraries (such as OpenCV) if needed
# TARGET_LINK_LIBRARIES(CUDA_Test ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(CUDA_Test ${CUDA_LIBRARIES} -lcublas -lcublas_device -lcudadevrt)
# MESSAGE(STATUS "opencv libs: ${OpenCV_LIBS}")
# MESSAGE(STATUS "cuda libs: ${CUDA_LIBRARIES}" -lcublas -lcublas_device -lcudadevrt)

MESSAGE(STATUS "**************************************")
