# project_use_library
这是一个使用库文件的项目模板，简单介绍了如何使用已有的库文件在自己的工程中。

特点：
- 定义了自己的FindHello.cmake模块文件，用于寻找Hello这个库，它会在FIND_PACKAGE()中被调用。
- 提供了两种链接库文件和头文件到自己工程的方法：使用FIND_PACKAGE()寻找库文件，或者已知库文件的位置并显式设定。

## 用法
```shell
cd project_use_library
mkdir build
cd build
cmake .. 
make
```
