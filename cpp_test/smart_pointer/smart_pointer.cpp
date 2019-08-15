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
#include <thread>
#include <chrono>
#include <mutex>

struct Base
{
    virtual void bar() { std::cout << "Base::bar\n"; }
    virtual ~Base() = default;
};

struct Derived : public Base
{
    Derived() { std::cout << "Derived::Derived\n"; }
    ~Derived() { std::cout << "Derived::~Derived\n"; }
    void bar() override { std::cout << "Derived::bar\n"; }
};

//! Works as a deleter for closing the FILE pointer
void closeFile(std::FILE* fp)
{
    std::fclose(fp);
}

// A function consuming a unique_ptr can take it by value or by rvalue reference
std::unique_ptr<Derived> passThrough(std::unique_ptr<Derived> p)
{
    p->bar();
    return p;
}

void testUniquePtr()
{
    std::cout << "unique ownership semantics demo\n";
    {
        auto p = std::make_unique<Derived>();
        // Now p owns nothing and holds a null pointer, and q owns the Derived object again
        auto q = passThrough(std::move(p));
        assert(!p);
        q->bar();
    }  // ~D called here after the scope

    std::cout << "Runtime polymorphism demo\n";
    {
        // p is a unique_ptr that owns a Derived object
        std::unique_ptr<Base> p = std::make_unique<Derived>();
        p->bar();

        std::vector<std::unique_ptr<Base>> v;  // unique_ptr can be stored in a container
        v.push_back(std::make_unique<Derived>());
        v.push_back(std::move(p));
        v.emplace_back(new Derived);  // now v's size is 3
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
        std::unique_ptr<Derived, std::function<void(Derived*)>> p(new Derived, [](Derived* ptr) {
            std::cout << "destroying from a custom deleter...\n";
            delete ptr;
        });
        p->bar();
    }  // the lambda above is called and D is destroyed

    std::cout << "Array form of unique_ptr demo\n";
    {
        std::unique_ptr<Derived[]> p{new Derived[3]};
    }  // calls ~D 3 times
}


//! Function for thread
void sharedPtrThread(std::shared_ptr<Base> p)
{
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::shared_ptr<Base> lp = p;  // thread-safe, even though the
                                   // shared use_count is incremented
    {
        static std::mutex io_mutex;
        std::lock_guard<std::mutex> lk(io_mutex);
        std::cout << "local pointer in a thread:\n"
                  << "  lp.get() = " << lp.get() << ", lp.use_count() = " << lp.use_count() << '\n';
    }
}

void testSharedPtr()
{
    std::shared_ptr<Base> p = std::make_shared<Derived>();
    std::cout << "Created a shared Derived (as a pointer to Base)\n"
              << "  p.get() = " << p.get() << ", p.use_count() = " << p.use_count() << '\n';
    std::thread t1(sharedPtrThread, p), t2(sharedPtrThread, p), t3(sharedPtrThread, p);
    p.reset();  // release ownership from main
    std::cout << "Shared ownership between 3 threads and released\n"
              << "ownership from main:\n"
              << "  p.get() = " << p.get() << ", p.use_count() = " << p.use_count() << '\n';
    t1.join();
    t2.join();
    t3.join();
    std::cout << "All threads completed, the last one deleted Derived\n";

	// Just like the deleter usage in unique_ptr, a common custom function can also work
	// as a deleter for shared_ptr. Here only shows lambda expression.
	std::cout << "Custom lambda-expression deleter for shared_ptr\n";
    {
        // Use lambda to create a deleter. Note that unlike the usage of deleter in unique_ptr,
		// here we don't put the function type in 'std::shared_ptr<Derived>', but only the lambda
		// expression deleter itself.
        std::shared_ptr<Derived> q(new Derived, [](Derived* ptr) {
            std::cout << "Destroying shared_ptr resource from a custom deleter ...\n";
            delete ptr;
        });
        q->bar();
        std::cout << "Count: " << q.use_count() << '\n';
        auto q1 = q;
        std::cout << "Count: " << q.use_count() << '\n';
        auto q2(q);
        std::cout << "Count: " << q.use_count() << '\n';
    }  // the lambda above is called and D is destroyed
}

int main()
{
    testUniquePtr();

	std::cout << "----------------------------" << std::endl;

	testSharedPtr();

    return 0;
}
