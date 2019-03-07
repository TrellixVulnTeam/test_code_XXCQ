# Build and Use OpenCV 

## Build from source

Official reference:
[https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html](https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html)

CMake command is very simple:
```cmake
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/path-to-install/ <opencv_source_directory>
```
NOTE for the space after each -D in the command. Without spaces after -D in the above example does NOT work. One strong suggestion is to use *ccmake* (gui version of CMake in linux), which is easier and friendly to be used to set up flags. 

The steps using *ccmake* is like this:
- Go to build folder and open ccmake:
```shell
mkdir build
cd build
ccmake <opencv_source_directory>
```
- Press **c** to configure once;
- Set all paths. Some important ones are:
```cmake
CMAKE_BUILD_TYPE = Release
CMAKE_INSTALL_PREFIX = /path-you-want-to-install/
WITH_CUDA = ON # if you want to build OpenCV with CUDA
OPENCV_EXTRA_MODULES_PATH=<opencv_contrib_path>/modules # if you want to build extra modules
```
- Configure again until no errors;
- Finally press **g** to generate.
- Build and install by
```shell
make -j4
sudo make install
```

### Build extra modules of OpenCV from source

Source repo and building reference: [https://github.com/opencv/opencv_contrib](https://github.com/opencv/opencv_contrib)

Some extra modules in OpenCV are stored in opencv_contrib, including the famous SIFT and SURF and many latest tools. If you want to build this, you need to download the source repo and then set the flag like this:
```cmake
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source_directory>
```
Also, it's easier to set it in *ccmake*.

## Use OpenCV in your own program

A simple CMakeLists.txt file using OpenCV is:
```cmake
# CMakeLists.txt sample
cmake_minimum_required(VERSION 3.1)
project(opencv_test)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(opencv_test main.cpp)
target_link_libraries(opencv_test ${OpenCV_LIBS})
```

### Specify different OpenCV path
Set OpenCV_DIR flag in cmake command if you want to use a different OpenCV path:
```shell
 cmake -DOpenCV_DIR=/path-to-OpenCVConfig.cmake/ <your-CMakeLists-path>
```
Note that the OpenCV_DIR path is the directory containing the **OpenCVConfig.cmake** file, such as `/opencv-4.0-install-path/lib/cmake/opencv4` after building and installing OpenCV 4.0.