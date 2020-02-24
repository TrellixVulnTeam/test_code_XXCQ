/*
  Test function type and function pointer.
  ���Լ��������ָ�룬��������ָ�롢����ָ�����ָ�롣

  Ref:
  https://en.cppreference.com/w/cpp/language/pointer
  https://stackoverflow.com/questions/23219227/function-type-vs-function-pointer-type

*/
#include <iostream>
#include <typeinfo>  // typeid()
#include <string>
#include "../../common/tools.h"

using namespace std;

//! �����ã�C-array ����������Ϊ��������
//! C-array as function parameter.
void sum(int *nums);     // ������÷���ָ����Ϊ�������
void sum(int nums[]);    // Same as above: ����������ʵ�ʹ�����ָ��������ָ��
void sum(int nums[10]);  // Same as above: dimension will be ignored. But this is good for documentation purposes.

//! �����á�һ���򵥵ĺ���
int square(int a)
{
    std::cout << "a: " << a << std::endl;
    return a * a;
}

//! ����������Ϊ���������
//! ��һ���������Ǻ������ͣ�function type�������ǣ����ᱻ������������������˵��Ĭ��ֱ��ת����Ϊһ������ָ��
//! ���ͣ�function pointer�����ⷢ���ڽ��뺯����֮ǰ��Ҳ����˵��takeInFuncType() ������ʵ�������
//! takeInFuncPtrType() ��������ȫһ���ġ���ͺͣ����������еģ��� C-array ����������Ϊ�������������һ������
//! ���������ֱ�Ĭ��ת��Ϊ������ָ�룩��
void takeInFuncType(int f(int), int a)
{
    cout << "Input f is a function type but it will be converted to a function pointer automatically. " << endl;
    f(a);  // OK: �������� f ��ʵ��ת��Ϊ�˺���ָ�룬�����Ǻ�������
}

//! ����ָ��������Ϊ���������
void takeInFuncPtrType(int (*f)(int), int a)
{
    cout << "Input f is a function pointer. " << endl;
    f(a);
}

//! ��������ļ�������
void testFunctionPointer()
{
    int a = 5;
    decltype(a) b = 2;  // b �� a ��������ȫ��ͬ

    decltype(square) f1;  // f1 ���Ǻ������ͣ����ͺ��� square ��������ȫ��ͬ������������ square
    cout << "type(f1): " << type_name<decltype(f1)>() << endl;  // �����һ������
    int f2(int);                                                // ����һ���������ͣ�f2 �� f1 ��ȫһ��
    cout << "type(f2): " << type_name<decltype(f2)>() << endl;

    // ERROR here: ���� square ��Ȼ�Ǻ����������������Ǻ������ͣ����Ǳ������������Ǻ���ָ�����ͣ����
    // �޷�������ֵ�� f1�������Ǻ������͡�
    // f1 = square;  // ERROR: cannot assign a function pointer to a function type.
    // f1(a); // ERROR: f1 �Ǻ������Ͷ����Ǻ���ָ�룬����޷����ú���

    decltype(square) *f3;  // f3 ���Ǻ���ָ�루pointer to function, or function pointer��
    cout << "type(f3): " << type_name<decltype(f3)>() << endl;  // ��ӡһ�£�ע������
    f3 = square;                                                // OK: ��Ϊ���߶��Ǻ���ָ��
    f3(a);                    // OK: Ҳ��ȷ����Ϊ f3 ��ʱ��ָ�� square ����
    int (*f4)(int) = square;  // OK: Ҳ��ȷ��f4 �� f3 ��ȫһ������������ʽ����
    cout << "type(f4): " << type_name<decltype(f4)>() << endl;  // ��ӡһ���ܿ�����һ���ġ�
    f4(a);                                                      // OK: now f4 is pointing to 'square'

    // ERROR here: ͬ������f1 �Ǻ������͵����Ǻ���ָ�룬�� takeInFuncType() ��Ȼ��һ����������ȥ��
    // �Ǻ������ͣ�����ʵ���������������Ǻ���ָ��
    // takeInFuncType(f1, a);

    // ����ȫ������ȷ��
    takeInFuncType(square, a);     // OK: function name 'square' is a function pointer
    takeInFuncType(f3, a);         // OK: f3 is a function pointer to 'square'
    takeInFuncPtrType(square, a);  // OK: function name 'square' is a function pointer
    takeInFuncPtrType(f3, a);      // OK: same as above
}

//! һ���򵥵��࣬���ڲ������еĺ���ָ��
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

//! �������еĺ���ָ��
void testMemberPointer()
{
    int (Base::*p)(int) = &Base::add;     // ����һ���ຯ��ָ�루member function pointer������
    decltype(&Base::add) r = &Base::add;  // same as above
    auto q = &Base::add;                  // this is much simpler

    Base base;
    (base.*r)(2);  // ע�⣺����Ψһ��ȷ��ʹ���ຯ��ָ��ķ��������汻ע�͵ķ���ȫ�Ǵ��
    // base.*p(2); // ERROR
    // (base.(*p))(2); // ERROR
    // base.p(2); // ERROR

    Base *bptr = &base;  // ʹ��һ����ָ�룬Ȼ����������ຯ��ָ��
    (bptr->*p)(3);       // ע�⣺����Ψһ��ȷ�ķ��������汻ע�͵ķ���ȫ�Ǵ��
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
