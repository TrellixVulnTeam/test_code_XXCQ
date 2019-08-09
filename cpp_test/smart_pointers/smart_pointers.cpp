/*******************************************************/
/*
Ref: https://en.cppreference.com/w/cpp/memory/unique_ptr

*/

#include <iostream>
#include <vector>
#include <memory>
#include <cstdio>
#include <fstream>
#include <cassert>
#include <functional>

struct Base
{
    virtual void bar() { std::cout << "Base::bar\n"; }
    virtual ~Base() = default;
};

struct Derive : Base
{
    Derive() { std::cout << "Derive::Derive\n"; }
    ~Derive() { std::cout << "Derive::~Derive\n"; }
    void bar() override { std::cout << "Derive::bar\n"; }
};

//! A function consuming a unique_ptr can take it by value or by rvalue reference
std::unique_ptr<Derive> passThrough(std::unique_ptr<Derive> p)
{
    p->bar();
    return p;
}

//! Works as a deleter for closing the FILE pointer
void closeFile(std::FILE* fp)
{
    std::fclose(fp);
}

void testUniquePtr()
{
    std::cout << "unique ownership semantics demo\n";
    {
        auto p = std::make_unique<Derive>();
        // Now p owns nothing and holds a null pointer, and q owns the Derive object again
        auto q = passThrough(std::move(p));
        assert(!p);
        q->bar();
    }  // ~D called here after the scope

    std::cout << "Runtime polymorphism demo\n";
    {
        // p is a unique_ptr that owns a Derive object
        std::unique_ptr<Base> p = std::make_unique<Derive>();
        p->bar();

        std::vector<std::unique_ptr<Base>> v;  // unique_ptr can be stored in a container
        v.push_back(std::make_unique<Derive>());
        v.push_back(std::move(p));
        v.emplace_back(new Derive); // now v's size is 3
        for (auto& p : v)
            p->bar();
    }  // ~D called 3 times

    std::cout << "Custom deleter demo\n";
    std::ofstream("demo.txt") << 'x';  // prepare the file to read
    {
        // The second parameter is a custom deleter, which is a function pointer
        std::unique_ptr<std::FILE, decltype(closeFile)*> fp(std::fopen("demo.txt", "r"), closeFile);

        // fopen could have failed; in which case fp holds a null pointer
        if (fp)
            std::cout << (char)std::fgetc(fp.get()) << '\n';
    }
    // fclose() called here, but only if FILE* is not a null pointer
    // (that is, if fopen succeeded)

    std::cout << "Custom lambda-expression deleter demo\n";
    {
		// Use lambda to create a deleter
        std::unique_ptr<Derive, std::function<void(Derive*)>> p(new Derive, [](Derive* ptr) {
            std::cout << "destroying from a custom deleter...\n";
            delete ptr;
        });
        p->bar();
    }  // the lambda above is called and D is destroyed

    std::cout << "Array form of unique_ptr demo\n";
    {
        std::unique_ptr<Derive[]> p{new Derive[3]};
    }  // calls ~D 3 times
}

int main()
{
    testUniquePtr();

    return 0;
}