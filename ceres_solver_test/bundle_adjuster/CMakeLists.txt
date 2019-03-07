cmake_minimum_required(VERSION 2.8)

project(simple_bundle_adjuster)

# Specify the version if you need
# find_package(Ceres 1.13.0 REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# simple_bundle_adjuster
add_executable(simple_bundle_adjuster simple_bundle_adjuster.cc)
target_link_libraries(simple_bundle_adjuster ${CERES_LIBRARIES})