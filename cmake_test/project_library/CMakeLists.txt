cmake_minimum_required(VERSION 3.5)

project(hello_mylib)

# 在CMakeLists.txt文件中修改默认的安装路径CMAKE_INSTALL_PREFIX，该路径默认是/usr/local
# 如果不想在这里修改，可以在调用cmake的时候使用：
#       cmake -DCMAKE_INSTALL_PREFIX=/your_install_path/ /your_CMAKELISTS.txt_path/
# NOTE: 下面的语句必须要放在PROJECT(xxx)这条命令的“后面“。
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
# 这里用默认工程路径PROJECT_BINARY_DIR（即你当前调用cmake命令时所在的路径）作为安装路径
  SET(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
message(STATUS "This is INSTALL dir: "${CMAKE_INSTALL_PREFIX})

# 用法: ADD_SUBDIRECTORY(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
add_subdirectory(src lib) 