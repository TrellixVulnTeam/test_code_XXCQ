cmake_minimum_required(VERSION 3.1)
<<<<<<< HEAD
project(cgal_version)

find_package(CGAL)
add_executable(cgal_version main.cpp)
target_link_libraries(cgal_version CGAL::CGAL)
=======
project(my_executable)

find_package(CGAL)
add_executable(my_executable check_version.cpp)
target_link_libraries(my_executable CGAL::CGAL)
>>>>>>> 08dd02ffa669bf2682b942dade4e8483a49e7dfb
