# Build and Use CGAL

## Build CGAL from source

Official installation manual: [https://doc.cgal.org/latest/Manual/installation.html](https://doc.cgal.org/latest/Manual/installation.html)

Special places to notice:
- Using *ccmake* (gui version of cmake in linux) maybe a little easier to set paths; 
- Set Qt5 path (only if you want to render results in CGAL). Follow instructions below.


### Compile CGAL with Qt5 and use it in your own program
Usually Qt5 is installed by default in some weird place which CMake usually cannot find (such as `/opt/`). You need to set Qt5_DIR path as the directory containing the file **Qt5Config.cmake** (instead of the Qt5 root folder). For instance, in Qt5.7.1 the directory could be 
```cmake
Qt5_DIR=/opt/Qt5.7.1/5.7/gcc_64/lib/cmake/Qt5/`
```
You can also set Qt5_DIR as global environment by adding `export Qt5_DIR=/path` to `~/.bashrc` so that Qt5 can be found in your own program. 

Meanwhile, usually Qt5 library files should be put in `/usr/lib/x86_64-linux-gnu/`, so in order to find the Qt5 library in your own program, a easy way is to copy all library files from your Qt5 lib folder to it by 
```shell
sudo cp /opt/Qt5.7.1/5.7/gcc_64/lib/lib*.so* /usr/lib/x86_64-linux-gnu/
```
Note that it will obviously bring in conflict between different Qt5 versions, so my suggestion is to install only 1 version of Qt5.

## Use CGAL with CMake in your own program

Here I copy the content from the official reference link here just in case the link disappears or is changed later: [https://github.com/CGAL/cgal/wiki/How-to-use-CGAL-with-CMake-or-your-own-build-system](https://github.com/CGAL/cgal/wiki/How-to-use-CGAL-with-CMake-or-your-own-build-system)

Since CGAL-4.12, linking with the CGAL libraries using the following syntax should be sufficient:
```cmake
find_package(CGAL)
add_executable(my_executable my_source_file.cpp)
target_link_libraries(my_executable CGAL::CGAL)
```
And for the other CGAL libraries, if they are needed, it is the same. For example, with CGAL_Core:
```cmake
find_package(CGAL REQUIRED COMPONENTS Core)
target_link_libraries(my_executable CGAL::CGAL CGAL::CGAL_Core)
```
To use CGAL with Qt5, specify the Qt5 path according to the description shown before. 

Here is a minimal example of CMakeLists.txt using Qt5:
```cmake
cmake_minimum_required(VERSION 3.1)
project(test_cgal)
#CGAL_Qt5 is needed for the drawing and CGAL_Core is needed for this special Kernel.
find_package(CGAL REQUIRED COMPONENTS Qt5 Core)
if(CGAL_FOUND AND CGAL_Qt5_FOUND)
  #required to use basic_viewer
  add_definitions(-DCGAL_USE_BASIC_VIEWER -DQT_NO_KEYWORDS)
  #create the executable of the application
  add_executable(test_cgal test.cpp)
  #link it with the required CGAL libraries
  target_link_libraries(test_cgal CGAL::CGAL CGAL::CGAL_Qt5 CGAL::CGAL_Core)
else()
  message("ERROR: this program requires CGAL and CGAL_Qt5 and will not be compiled.")
endif()
```
And a testing cpp file:
```cpp
// test.cpp
#include <CGAL/Exact_predicates_exact_constructions_kernel_with_sqrt.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/draw_surface_mesh.h>

#define CGAL_USE_BASIC_VIEWER 1
typedef CGAL::Exact_predicates_exact_constructions_kernel_with_sqrt K;
typedef K::Point_3 Point;
typedef CGAL::Surface_mesh<Point> Mesh;

int main()
{
  Mesh m;
  CGAL::make_icosahedron<Mesh, Point>(m);
  CGAL::draw(m);
  return 0;
}
```

### Specify different CGAL path
Set CGAL_DIR in cmake command if you want to use different CGAL path:
```cmake
cmake -DCGAL_DIR=/path-to-CGALConfig.cmake-file <your-CMakeLists-path>
```
where `CGAL_DIR` is the directory containing the file **CGALConfig.cmake**, such as inside your CGAL installation path `/CGAL-installation-path/lib/cmake/CGAL`.

### Use CMakeLists.txt generator for CGAL

There is also a shell-script CGAL CMakeList generator program to use: [https://github.com/CGAL/cgal/blob/master/Scripts/scripts/cgal_create_cmake_script](https://github.com/CGAL/cgal/blob/master/Scripts/scripts/cgal_create_cmake_script). Just run the program to create your own CMakeLists.txt and build from there. See comments for details about usage.

## Usage of CGAL

### Basic model viewer

Inside the model viewer (created by `CGAL::draw()`):
- CTRL + q: close the window;
- e: draw/hide edges;
- v: draw/hide vertices;
- f: draw/hide faces

