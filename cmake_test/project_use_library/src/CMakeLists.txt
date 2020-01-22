SET(PROJECT_NAME test_hello)

# 下面给出两种链接库文件的方法，平时任选一种即可。本文档中默认使用的是第一种方法。
# 1) 使用FIND_PACKAGE()命令，该命令会调用cmake模块文件FindHello.cmake（不过该文件需要我们自行准备）来寻找hello库文件。
# 2) 直接设定库文件的位置（如果你知道的话）。

# 设定Find模块的目录给接下来FIND_PACKAGE()命令，即我们自行定义的FindHello.cmake所在的位置，
# 否则FIND_PACKAGE()将会找不到它。
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# 调用FindHello.cmake来寻找hello库文件。具体可以参看该文件的内容。
# 注意：
# 1) FIND_PACKAGE()命令其实可以增加自定义路径以供搜索，方法是：FIND_PACKAGE(HELLO REQUIRED PATHS <path_to_search>)
# 2) 该命令中的库文件名字和FindXXX.cmake中必须是一致的。即，这里用的是Hello，因此cmake文件名必须是FindHello.cmake，而不能是FindHELLO.cmake。
# 3) FindHello.cmake文件中的命令也是cmake命令，即，其实可以将它们完全放在这里用。其实，FindHello.cmake文件最重要的命令只有FIND_LIBRARY()
#    和FIND_PATH()命令，有时候为了方便，直接放在这里用就行，这样就不用再专门写一个.cmake文件了。这经常应用于需要同时使用多个不同种类的库文件的时候。
FIND_PACKAGE(Hello REQUIRED)
IF(HELLO_FOUND)
    ADD_EXECUTABLE(${PROJECT_NAME} main.cpp)
    # 如果FIND_PACKAGE()成功找到了hello库，那么cmake会定义HELLO_INCLUDE_DIR（代表头文件的路径文件夹）
    # 和HELLO_LIBRARY（代表库文件的绝对路径），因此可以直接使用，而不用再使用LINK_DIRECTORIES命令了。
    INCLUDE_DIRECTORIES(${HELLO_INCLUDE_DIR})
    # 链接要使用的库文件名称到项目。第一个参数是项目名，第二个参数是库文件名称，这里就是绝对路径了，例如/usr/local/lib/libhello.so。
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${HELLO_LIBRARY})
ELSE(HELLO_FOUND)
    # 这里是显式设置库文件的路径，然后链接它到工程中。如果使用该方法，那么就无需使用上面和FIND_PACKAGE相关的命令了。
    SET(MYLIB_PATH ../../project_library/build/install)
    message(STATUS "This is library dir: " ${MYLIB_PATH})
    # 增加要使用的库文件所在路径。
    # 注意，该命令必须放在ADD_EXECUTABLE()命令之前，否则会出现找不到lib文件的错误。
    # 当然了，如果TARGET_LINK_LIBRARIES()命令中使用的是库文件的绝对路径的话（例如库文件是
    # 通过FIND_PACKAGE()命令找到的话），下面这句命令其实也就不需要了，当然也就不会出错了。
    LINK_DIRECTORIES(${MYLIB_PATH}/lib)
    # 生成可执行文件。
    ADD_EXECUTABLE(${PROJECT_NAME} main.cpp)
    # 增加要使用的库的头文件所在路径
    INCLUDE_DIRECTORIES(${MYLIB_PATH}/include)
    # 链接要使用的库文件名称到项目。第一个参数是项目名，第二个参数hello就是库文件名称，默认是动态库，
    # 也可以写全称libhello.so。如果想要显式得链接到静态库，则需要用全称libhello.a
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} hello)
ENDIF(HELLO_FOUND)