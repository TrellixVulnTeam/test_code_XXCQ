cmake_minimum_required(VERSION 3.1.0)

project(boost_test)

# set(Boost_INCLUDE_DIR /usr/local/include)
# set(Boost_LIBRARY_DIR /usr/local/lib)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

add_subdirectory(helloworld)
add_subdirectory(program_options)
add_subdirectory(uuid_test)
