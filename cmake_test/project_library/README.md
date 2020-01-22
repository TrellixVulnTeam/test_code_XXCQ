# project_library
这是一个可以生成动态库和静态库的项目模板。它可以将src文件夹中的cpp文件编译成动态库和静态库文件，并安装库文件以及相关的头文件。

## 用法
```shell
cd project_library
mkdir build # You can use any custom path to put your built files
cd build
# 下面一句也可以增加自定义的安装路径：cmake -DCMAKE_INSTALL_PREFIX=/path/ ..
cmake .. 
make
make install # 安装相关文件到安装目录
```
