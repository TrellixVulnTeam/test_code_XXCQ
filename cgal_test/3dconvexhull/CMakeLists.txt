<<<<<<< HEAD
cmake_minimum_required(VERSION 3.1)
project(3dconvexhull)

#CGAL_Qt5 is needed for the drawing and CGAL_Core is needed for this special Kernel.
find_package(CGAL REQUIRED COMPONENTS Qt5 Core)
if(CGAL_FOUND AND CGAL_Qt5_FOUND)
  #required to use basic_viewer
  add_definitions(-DCGAL_USE_BASIC_VIEWER -DQT_NO_KEYWORDS)
  #create the executable of the application
  add_executable(3dconvexhull 3dconvexhull.cpp)
  #link it with the required CGAL libraries
  target_link_libraries(3dconvexhull CGAL::CGAL CGAL::CGAL_Qt5 CGAL::CGAL_Core)
else()
  message("ERROR: this program requires CGAL and CGAL_Qt5 and will not be compiled.")
endif()
=======
# Created by the script cgal_create_cmake_script
# This is the CMake script for compiling a CGAL application.


project( 3dconvexhull_ )

cmake_minimum_required(VERSION 2.8.10)

set(CMAKE_CXX_STANDARD 14)

find_package(CGAL QUIET COMPONENTS Core )

if ( CGAL_FOUND )

  create_single_source_cgal_program( "quickhull_3.cpp" )

else()
  
    message(STATUS "This program requires the CGAL library, and will not be compiled.")
  
endif()

>>>>>>> 08dd02ffa669bf2682b942dade4e8483a49e7dfb
