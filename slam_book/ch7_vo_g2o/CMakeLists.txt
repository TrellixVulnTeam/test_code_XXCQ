cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME g2o_vo)

project(${PROJECT_NAME})

set( CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall" )

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(G2O REQUIRED)
# G2O 中的 CSparse 库需要使用 CSParse 库（来自 suite-sparse），因此这里增加它的依赖
find_package( CSparse REQUIRED )

# message(STATUS "G2O DIR: "${G2O_INCLUDE_DIR})
# message(STATUS "G2O core: " ${G2O_CORE_LIBRARY})
# message(STATUS "G2O stuff: " ${G2O_STUFF_LIBRARY})
# message(STATUS "G2O sba: " ${G2O_TYPES_SBA})
# message(STATUS "G2O csparse: " ${G2O_SOLVER_CSPARSE})

include_directories(
    ${EIGEN3_INCLUDE_DIR} 
    ${G2O_INCLUDE_DIR} 
    ${OpenCV_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
)
add_executable(${PROJECT_NAME} g2o_vo_3df2d.cpp)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS} 
    ${G2O_LIBRARIES}
    ${CSPARSE_LIBRARY}
)


    