#include "hello.h"
#include <iostream>

namespace MySpace {
void helloWorld(const std::string& str)
{
    std::cout << "Your message:" + str << std::endl;
}
}  // namespace MySpace