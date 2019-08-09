/*

	Test function type and function pointer.

	Ref:
	https://stackoverflow.com/questions/23219227/function-type-vs-function-pointer-type

*/
#include <iostream>
#include <typeinfo>  // typeid()

using namespace std;

//! C-array as function parameter.
void sum(int *nums);
void sum(int nums[]);    // Same as above: array name is actually a pointer to an array
void sum(int nums[10]);  // Same as above: dimention will be ignored. But this is good for documentation purposes.

//! Testing function
int square(int a)
{
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
    cout << "type(a): " << typeid(a).name() << endl;  // typeid().name() is the type of a variable/object
    decltype(a) b = 2;                                // b is with type a
    cout << "type(b): " << typeid(b).name() << endl;

    // Test function type and function pointers
    decltype(square) f1;  // f1 is a function type
    cout << "type(f1): " << typeid(f1).name() << endl;
    int f2(int);
    cout << "type(f2): " << typeid(f2).name() << endl;

    // ERROR here: a function name like 'square' is actually a pointer to a function (converted automatically)
    // instead of a function type.
    // f1 = square;  // ERROR: cannot assign a function pointer to a function type.
    // f1(a); // ERROR: f is a function type instead of function pointer

    decltype(square) *f3;  // f3 is a pointer to function, or called function pointer
    cout << "type(f3): " << typeid(f3).name() << endl;
    f3 = square;  // OK: both are function pointers
    f3(a);        // OK: f3 is a function pointer to square
    int (*f4)(int) = square; // OK: f4 is a pointer to a function with same type as square
    cout << "type(f4): " << typeid(f4).name() << endl;
	f4(a);

    // ERROR: f1 is a function type only instead of a function pointer, so the compiler cannot find what f1 is.
    // takeInFuncType(f1, a);

    // The following are all OK
    takeInFuncType(square, a);     // OK: function name 'square' is a function pointer
    takeInFuncType(f3, a);         // OK: f3 is a function pointer to 'square'
    takeInFuncPtrType(square, a);  // OK: function name 'square' is a function pointer
    takeInFuncPtrType(f3, a);      // OK: same as above
}

int main()
{
    testFunctionPointer();
    return 0;
}