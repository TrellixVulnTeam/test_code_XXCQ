cmake_minimum_required(VERSION 3.5)

# 定义工程。该命令用法是：
#   PROJECT(projectname [CXX] [C] [Java])
# 特点：
# - 这个指令隐式的定义了两个cmake变量: <projectname>_BINARY_DIR 以及<projectname>_SOURCE_DIR，
#   这里就是 HELLO_BINARY_DIR 和 HELLO_SOURCE_DIR。其中，HELLO_BINARY_DIR此时就是你运行cmake
#   命令时所在的路径，而HELLO_SOURCE_DIR是你的CMakeLists.txt文档所在的路径。
# - 同时，cmake系统也帮助我们预定义了 PROJECT_BINARY_DIR 和 PROJECT_SOURCE_DIR 变量，他们的值
#   分别跟 HELLO_BINARY_DIR 与 HELLO_SOURCE_DIR 完全一致，因此建议使用"PROJECT_"这种变量。
project(HELLO)
# 不妨打印出来这几个变量对比一下是否相同。
message(STATUS "This is BINARY dir: " ${HELLO_BINARY_DIR})
message(STATUS "This is BINARY dir: " ${PROJECT_BINARY_DIR}) # same as above
message(STATUS "This is SOURCE dir: "${HELLO_SOURCE_DIR})
message(STATUS "This is SOURCE dir: "${PROJECT_SOURCE_DIR})

# 在CMakeLists.txt文件中修改默认的安装路径CMAKE_INSTALL_PREFIX，该路径默认是/usr/local
# 如果不想在这里修改，可以在调用cmake的时候使用：
# cmake -DCMAKE_INSTALL_PREFIX=/your_install_path/ /your_CMAKELISTS.txt_path/
# NOTE: 下面的语句必须要放在PROJECT(xxx)这条命令的“后面“。
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
# 这里用默认工程路径PROJECT_BINARY_DIR（即你当前调用cmake命令时所在的路径）作为安装路径
  SET(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
message(STATUS "This is INSTALL dir: "${CMAKE_INSTALL_PREFIX})

# 安装各种文件到目标安装目录。注意这里的目标路径（DESTINATION后的路径）都是
# 在${CMAKE_INSTALL_PREFIX}路径下的相对路径
# 安装普通文件（FILES关键字指代普通文件）
INSTALL(FILES COPYRIGHT ${PROJECT_SOURCE_DIR}/README.md DESTINATION doc)
# 安装程序（FILES关键字指代可执行程序类文件）
INSTALL(PROGRAMS runhello.sh DESTINATION bin)
# 安装文件夹（DIRECTORY关键字指代文件夹）
# 注意，DIRECTORY后的这个path结尾带不带反斜杠"/"是有区别的。
# - 如果结尾不带"/"（例如abc），表明将要拷贝整个文件夹abc全部文件和该文件夹本身到目标路径中（安装后结果就是/install_path/doc/abc/）；
# - 如果结尾带"/"（例如abc/），则表明拷贝文件夹abc内部的全部文件，但不包含abc这个文件夹本身（安装后的结果就是/install_path/doc/...）。
INSTALL(DIRECTORY doc/ DESTINATION doc)

# 加入子工程并设定最终生成的可执行文件的目录。子工程目录src中必须有相应的CMakeLists.txt。用法: 
#   ADD_SUBDIRECTORY(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
add_subdirectory(src bin) 
