cmake_minimum_required(VERSION 3.0)

set(Boost_INCLUDE_DIR /usr/local/include)
set(Boost_LIBRARY_DIR /usr/local/lib)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

add_executable(helloworld helloworld.cpp)
target_link_libraries( helloworld ${Boost_LIBRARIES} )