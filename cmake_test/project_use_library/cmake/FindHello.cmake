## 该文件是我们自己定义的cmake模块文件，用于寻找Hello库文件。

# 找出包含hello.h这个头文件的路径。如果找到，设置给HELLO_INCLUDE_DIR。这里的NAMES和PATHS关键字可以省略。
# 几点注意：
# - FIND_LIBRARY()和FIND_PATH()用法完全一样。
# - NAMES后可以放多个头文件供搜索，似乎是，只要找到一个，该命令就返回。
# - FIND_LIBRARY()默认寻找动态库文件.so，如果要寻找静态库文件.a，需要提前显式设定CMAKE_FIND_LIBRARY_SUFFIXES变量为.a，即：
#       SET(CMAKE_FIND_LIBRARY_SUFFIXES .a)
# - 可以提前将要搜索的所有路径显式设置到CMAKE_LIBRARY_PATH变量中，这样的话，下面的FIND_PATH或者FIND_LIBRARY中的PATHS可以省略。
#   这样的好处是，可以用于多个不同种类的库文件的查找，只需设置一次路径就够了。举例：
#       SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} path1 path2 ...)
#       FIND_LIBRARY(LIB1 NAMES lib1)
#       FIND_LIBRARY(LIB2 NAMES lib2)
#       ...
# 寻找头文件
FIND_PATH(HELLO_INCLUDE_DIR NAMES hello.h PATHS /usr/local /usr/include/hello /usr/local/include/hello)
# FIND_LIBRARY()找到的是库文件的绝对路径的全名，而不仅仅是包含它的文件夹。
FIND_LIBRARY(HELLO_LIBRARY NAMES hello hello_static PATHS /usr/local /usr/lib /usr/local/lib)
# 下面就是定义一些相关的变量，这些变量在外面的CMakeLists.txt中也可以使用。
IF (HELLO_INCLUDE_DIR AND HELLO_LIBRARY)
    SET(HELLO_FOUND TRUE)
ENDIF (HELLO_INCLUDE_DIR AND HELLO_LIBRARY)
IF (HELLO_FOUND)
    # HELLO_FIND_QUIETLY是为了调用FIND_PACKAGE()中的QUIET关键字相对应的。即，如果有QUIET关键字，
    # cmake就会定义HELLO_FIND_QUIETLY这个参数，下面的message语句就不会执行了。
    IF (NOT HELLO_FIND_QUIETLY)
        MESSAGE(STATUS "Found Library Hello. Headers: ${HELLO_INCLUDE_DIR}; Libraries: ${HELLO_LIBRARY}")
    ENDIF (NOT HELLO_FIND_QUIETLY)
ELSE (HELLO_FOUND)
    # HELLO_FIND_REQUIRED对应调用FIND_PACKAGE()中的REQUIRED关键字，表明该库是必须要找到的，否则不予编译。
    IF (HELLO_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find Hello library")
    ENDIF (HELLO_FIND_REQUIRED)
ENDIF (HELLO_FOUND)