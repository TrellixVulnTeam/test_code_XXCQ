/*
  Test function type and function pointer.

  Ref:
  https://stackoverflow.com/questions/23219227/function-type-vs-function-pointer-type

*/
#include <iostream>
#include <typeinfo>  // typeid()
#include <string>
#include "../../common/tools.h"

using namespace std;

//! C-array as function parameter.
void sum(int *nums);
void sum(int nums[]);    // Same as above: array name is actually a pointer to an array
void sum(int nums[10]);  // Same as above: dimension will be ignored. But this is good for documentation purposes.

//! Testing function
int square(int a)
{
    PRINT_YELLOW("a: %d", a);
    std::cout << "a: " << a << std::endl;
    return a * a;
}

//! First parameter here is a function type but it will be converted to a function pointer automatically
//! even before entering into this function body, i.e., the input parameters in takeInFuncType() here are
//! actually exactly the SAME as that in the following function takeInFuncPtrType(). This is similar to
//! the C-array type as a function parameter.
void takeInFuncType(int f(int), int a)
{
    cout << "Input f is a function type but it will be converted to a function pointer automatically. " << endl;
    f(a);  // OK: f now is a function pointer
}

//! First parameter here is a function pointer.
void takeInFuncPtrType(int (*f)(int), int a)
{
    cout << "Input f is a function pointer. " << endl;
    f(a);
}

//! Test function pointers
void testFunctionPointer()
{
    int a = 5;
    // cout << "type(a): " << type(a) << endl;  // function type() is defined in 'tools.h'
    decltype(a) b = 2;                       // b is with type a
    // cout << "type(b): " << type(b) << endl;

    // Test function type and function pointers
    decltype(square) f1;  // f1 is a function type
    cout << "type(f1): " << type_name<decltype(f1)>() << endl;
    int f2(int);  // f2 is also a function type, and the same as f1
    cout << "type(f2): " << type_name<decltype(f2)>() << endl;

    // ERROR here: a function name like 'square' is actually a pointer to a function (converted automatically)
    // instead of a function type.
    // f1 = square;  // ERROR: cannot assign a function pointer to a function type.
    // f1(a); // ERROR: f is a function type instead of function pointer

    decltype(square) *f3;  // f3 is a pointer to function, or called function pointer
    cout << "type(f3): " << type_name<decltype(f3)>() << endl;
    f3 = square;              // OK: both are function pointers
    f3(a);                    // OK: f3 is a function pointer to square
    int (*f4)(int) = square;  // f4 is a function pointer type, same as f3 and 'square'
    cout << "type(f4): " << type_name<decltype(f4)>() << endl;
    f4(a);  // OK: now f4 is pointing to 'square'

    // ERROR: f1 is a function type only instead of a function pointer, so the compiler cannot find what f1 is.
    // takeInFuncType(f1, a);

    // The following are all OK
    takeInFuncType(square, a);     // OK: function name 'square' is a function pointer
    takeInFuncType(f3, a);         // OK: f3 is a function pointer to 'square'
    takeInFuncPtrType(square, a);  // OK: function name 'square' is a function pointer
    takeInFuncPtrType(f3, a);      // OK: same as above
}

struct Base
{
    int add(int n)
    {
        int sum = val + n;
        PRINT_WHITE("Base: %d, Input: %d, Sum: %d", val, n, sum);
        return sum;
    }
    int val = 5;
};

void testMemberPointer()
{
    int (Base::*p)(int) = &Base::add; // define and initialize a member function pointer type
    decltype(&Base::add) r = &Base::add; // same as above
    auto q = &Base::add; // this is much simpler

    Base base;
    (base.*r)(2); // NOTE: this is the only correct usage of calling a pointer to member function
    // base.*p(2); // ERROR
    // (base.(*p))(2); // ERROR
    // base.p(2); // ERROR

    Base *bptr = &base;
    (bptr->*p)(3); // ONLY correct way
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
