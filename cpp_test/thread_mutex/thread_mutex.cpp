#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <functional>

std::mutex mutex_;

/*!
 * A class that has thread object as member variable
 */
class ThreadWrapper
{
private:
    //! std::thread object
    std::thread thread_;

public:
    //! Delete the default copy constructor
    ThreadWrapper(const ThreadWrapper&) = delete;

    //! Delete the default Assignment opeartor
    ThreadWrapper& operator=(const ThreadWrapper&) = delete;

    //! Parameterized Constructor
    ThreadWrapper(std::function<void()> thread_func);

    //! Define Move Constructor (move input object to current object)
    ThreadWrapper(ThreadWrapper&& obj);

    //! Define Move Assignment Operator
    ThreadWrapper& operator=(ThreadWrapper&& obj);

    //! Destructor
    ~ThreadWrapper();
};

//! Parameterized Constructor
ThreadWrapper::ThreadWrapper(std::function<void()> thread_func) : thread_(thread_func)
{
    std::cout << "Initialization. " << std::endl;
    std::cout << "Yes" << std::endl;
}

//! Move Constructor
ThreadWrapper::ThreadWrapper(ThreadWrapper&& obj) : thread_(std::move(obj.thread_))
{
    std::cout << "Move Constructor is called" << std::endl;
}

//! Move Assignment Operator
ThreadWrapper& ThreadWrapper::operator=(ThreadWrapper&& obj)
{
    std::cout << "Move Assignment is called" << std::endl;
    if (thread_.joinable())
        thread_.join();
    thread_ = std::move(obj.thread_);
    return *this;
}

// Destructor
ThreadWrapper::~ThreadWrapper()
{
    if (thread_.joinable())
        thread_.join();
}

int main()
{
    // Creating a std::function object for thread usage
    std::function<void()> thread_func = []() {
        std::lock_guard<std::mutex> lock(mutex_);


        // Sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));


        // Print thread ID
        std::cout << "From Thread ID : " << std::this_thread::get_id() << "\n";
    };

    {
        // Create a ThreadWrapper object
        // It will internally start the thread
        ThreadWrapper wrapper(thread_func);
        // When wrapper will go out of scope, its destructor will be called
        // Which will internally join the member thread object
    }
    // Create a vector of ThreadWrapper objects
    std::vector<ThreadWrapper> thread_wrappers;

    // Add ThreadWrapper objects in thread
    ThreadWrapper thread_wrapper1(thread_func);
    ThreadWrapper thread_wrapper2(thread_func);
    thread_wrappers.push_back(std::move(thread_wrapper1));
    thread_wrappers.push_back(std::move(thread_wrapper2));
    ThreadWrapper thread_wrapper3(thread_func);

    // Change the content of vector
    thread_wrappers[1] = std::move(thread_wrapper3);

    // When vector will go out of scope, its destructor will be called, which will
    // internally call the destructor all ThreadWrapper objects , which in turn
    // joins the member thread object.
    return 0;
}
