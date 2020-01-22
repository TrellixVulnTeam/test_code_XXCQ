# 生成shared动态库文件（即so文件）和static静态库文件（即.a文件）
ADD_LIBRARY(hello SHARED hello.cpp)
ADD_LIBRARY(hello_static STATIC hello.cpp)
# 上面的两行ADD_LIBRARY()命令中，shared和static库文件必须是不同的名字（如果使用相同名字会报错），
# 但我们需要的通常是相同名字的库文件（当然后缀不同），例如libhello.so和libhello.a，
# 下面这行命令作用是重新设置static库文件的名字为hello。
SET_TARGET_PROPERTIES(hello_static PROPERTIES OUTPUT_NAME "hello")

# 下面这两行命令是用于避免cmake删除相同名字的库文件，使得我们可以同时得到libhello.so和libhello.a文件。但是经过测试，
# 不使用这两行命令也能同时得到文件名相同（后缀不同的）这两个库文件。或许是，新的cmake版本修改了这个缺点吧。
# SET_TARGET_PROPERTIES(hello PROPERTIES CLEAN_DIRECT_OUTPUT 1)
# SET_TARGET_PROPERTIES(hello_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

# 增加动态库的版本号
SET_TARGET_PROPERTIES(hello PROPERTIES VERSION 1.2 SOVERSION 1)

# 安装库文件到目标安装目录
# 这里，TARGETS后面是目标库文件的名字，这两个名字必须和上面的ADD_LIBRARY()中使用的名字相同。
# LIBRARY指的是动态库文件，ARCHIVE指的是静态库文件。
# 另外还有关键字”RUNTIME DESTINATION bin“代指可执行文件。
# 注意：这里静态库文件的名字如果写成"hello"的话，会报错说找不到该静态库。因此，静态库文件必须和
# 上面的ADD_LIBRARY()中使用的静态库名字相同（即便上面已经把静态库的名字改成了动态库相同了）
INSTALL(TARGETS hello hello_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
# 安装头文件到目标安装目录。这里头文件被安装在CMAKE_INSTALL_PREFIX/include/hello中
INSTALL(FILES hello.h DESTINATION include/hello)