/*
  Test function type and function pointer.
  测试几种特殊的指针，包括数组指针、函数指针和类指针。

  Ref:
  https://en.cppreference.com/w/cpp/language/pointer
  https://stackoverflow.com/questions/23219227/function-type-vs-function-pointer-type

*/
#include <iostream>
#include <typeinfo>  // typeid()
#include <string>
#include "../../common/tools.h"

using namespace std;

//! 测试用：C-array 数组名字作为函数参数
//! C-array as function parameter.
void sum(int *nums);     // 最常见的用法：指针作为传入参数
void sum(int nums[]);    // Same as above: 数组名字其实就代表了指向该数组的指针
void sum(int nums[10]);  // Same as above: dimension will be ignored. But this is good for documentation purposes.

//! 测试用。一个简单的函数
int square(int a)
{
    std::cout << "a: " << a << std::endl;
    return a * a;
}

//! 函数类型作为传入参数。
//! 第一个参数就是函数类型（function type），但是，它会被编译器“看做”或者说“默认直接转化”为一个函数指针
//! 类型（function pointer），这发生在进入函数体之前。也就是说，takeInFuncType() 函数其实和下面的
//! takeInFuncPtrType() 函数是完全一样的。这就和（上面例子中的）将 C-array 数组名字作为函数传入参数是一个道理
//! （数组名字被默认转化为了数组指针）。
void takeInFuncType(int f(int), int a)
{
    cout << "Input f is a function type but it will be converted to a function pointer automatically. " << endl;
    f(a);  // OK: 不过这里 f 其实被转化为了函数指针，而不是函数类型
}

//! 函数指针类型作为传入参数。
void takeInFuncPtrType(int (*f)(int), int a)
{
    cout << "Input f is a function pointer. " << endl;
    f(a);
}

//! 测试上面的几个函数
void testFunctionPointer()
{
    int a = 5;
    decltype(a) b = 2;  // b 和 a 的类型完全相同

    decltype(square) f1;  // f1 就是函数类型，它和函数 square 的类型完全相同，但它不等于 square
    cout << "type(f1): " << type_name<decltype(f1)>() << endl;  // 输出看一下类型
    int f2(int);                                                // 定义一个函数类型，f2 和 f1 完全一样
    cout << "type(f2): " << type_name<decltype(f2)>() << endl;

    // ERROR here: 这里 square 虽然是函数名，但是它并非函数类型，而是被编译器看做是函数指针类型，因此
    // 无法将它赋值给 f1，后者是函数类型。
    // f1 = square;  // ERROR: cannot assign a function pointer to a function type.
    // f1(a); // ERROR: f1 是函数类型而不是函数指针，因此无法调用函数

    decltype(square) *f3;  // f3 就是函数指针（pointer to function, or function pointer）
    cout << "type(f3): " << type_name<decltype(f3)>() << endl;  // 打印一下，注意区别
    f3 = square;                                                // OK: 因为两者都是函数指针
    f3(a);                    // OK: 也正确，因为 f3 此时就指向 square 函数
    int (*f4)(int) = square;  // OK: 也正确，f4 和 f3 完全一样，这里是显式定义
    cout << "type(f4): " << type_name<decltype(f4)>() << endl;  // 打印一下能看出是一样的。
    f4(a);                                                      // OK: now f4 is pointing to 'square'

    // ERROR here: 同样道理，f1 是函数类型但不是函数指针，而 takeInFuncType() 虽然第一个参数看上去是
    // 是函数类型，但其实编译器将它看做是函数指针
    // takeInFuncType(f1, a);

    // 下面全都是正确的
    takeInFuncType(square, a);     // OK: function name 'square' is a function pointer
    takeInFuncType(f3, a);         // OK: f3 is a function pointer to 'square'
    takeInFuncPtrType(square, a);  // OK: function name 'square' is a function pointer
    takeInFuncPtrType(f3, a);      // OK: same as above
}

//! 一个简单的类，用于测试其中的函数指针
struct Base
{
    int add(int x)
    {
        int sum = val + x;
        std::cout << "Base: " << val << ", result: " << x << std::endl;
        return sum;
    }
    int val = 5;
};

//! 测试类中的函数指针
void testMemberPointer()
{
    int (Base::*p)(int) = &Base::add;     // 定义一个类函数指针（member function pointer）类型
    decltype(&Base::add) r = &Base::add;  // same as above
    auto q = &Base::add;                  // this is much simpler

    Base base;
    (base.*r)(2);  // 注意：这是唯一正确的使用类函数指针的方法，下面被注释的方法全是错的
    // base.*p(2); // ERROR
    // (base.(*p))(2); // ERROR
    // base.p(2); // ERROR

    Base *bptr = &base;  // 使用一个类指针，然后测试它的类函数指针
    (bptr->*p)(3);       // 注意：这是唯一正确的方法，下面被注释的方法全是错的
    // bptr->*p(2); // ERROR
    // (bptr->(*p))(2); // ERROR
    // bptr->p(2); // ERROR
}

int main()
{
    testFunctionPointer();
    std::cout << "---------------------------" << std::endl;
    testMemberPointer();
    return 0;
}
