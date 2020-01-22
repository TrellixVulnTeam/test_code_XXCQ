# project_template
这是一个cmake多个子工程的项目模板，它可以编译src文件夹中的cpp文件，生成可执行文件，并安装相关的所有文件。该工程特点是：
- 包含了一个子工程。子工程的源码文件放在src中，并在src中定义一个子CMakeLists.txt，然后用工程根目录的CMakeLists.txt调用子目录src的CMakeLists.txt。这可以很方便的扩展到多个子工程的project中；
- 可以安装可执行程序和其他诸如doc, copyright, shell script等相关文件。详细参见根目录的CMakeLists.txt。
- 在CMakeLists.txt修改了默认的安装路径，即CMAKE_INSTALL_PREFIX路径。

## 用法
```shell
cd project_template
mkdir build # You can use any custom path to put your built files
cd build
# 下面一句也可以增加自定义的安装路径：cmake -DCMAKE_INSTALL_PREFIX=/path/ ..
cmake .. 
make
make install # 安装相关文件到安装目录
```
