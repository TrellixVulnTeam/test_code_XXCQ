#ifndef TOOLS_H
#define TOOLS_H

#include <iostream>
#include <string>
#include <stdio.h>
#include "termcolor.hpp"
#include <string_view>

//! Print text in color in the console.
#if defined(__linux__) || defined(__APPLE__)

// Use ANSI color escape code. Works on most Unix terminals.
// Ref: https://misc.flogisoft.com/bash/tip_colors_and_formatting
#define PRINT_RED(...)        \
    {                         \
        printf("\033[1;31m"); \
        printf(__VA_ARGS__);  \
        printf("\033[0m\n");  \
    }

#define PRINT_GREEN(...)      \
    {                         \
        printf("\033[1;32m"); \
        printf(__VA_ARGS__);  \
        printf("\033[0m\n");  \
    }

#define PRINT_YELLOW(...)     \
    {                         \
        printf("\033[1;33m"); \
        printf(__VA_ARGS__);  \
        printf("\033[0m\n");  \
    }

#define PRINT_BLUE(...)       \
    {                         \
        printf("\033[1;34m"); \
        printf(__VA_ARGS__);  \
        printf("\033[0m\n");  \
    }
#define PRINT_MAGENTA(...)    \
    {                         \
        printf("\033[1;35m"); \
        printf(__VA_ARGS__);  \
        printf("\033[0m\n");  \
    }

#define PRINT_CYAN(...)       \
    {                         \
        printf("\033[1;36m"); \
        printf(__VA_ARGS__);  \
        printf("\033[0m\n");  \
    }

#define PRINT_WHITE(...)     \
    {                        \
        printf("\033[1;0m"); \
        printf(__VA_ARGS__); \
        printf("\033[0m\n"); \
    }

#define PRINT_BLACK(...)      \
    {                         \
        printf("\033[1;30m"); \
        printf(__VA_ARGS__);  \
        printf("\033[0m\n");  \
    }

#elif defined(_WIN32)

// Use a header-only tool called 'termcolor' to print colored text in Windows.
// Ref: https://github.com/ikalnytskyi/termcolor
//
// Actually termcolor supports Windows, Linux and MacOS, so we actually don't need to define
// different platform macros. However, the text color looks a little dark than result using
// the linux implementation above.

// NOTE that termcolor only supports 'cout',so convert the input to char array at first.
#define PRINT_RED(...)                                                       \
    {                                                                        \
        char buf[1024];                                                      \
        int n = sprintf(buf, __VA_ARGS__);                                   \
        std::cout << termcolor::red << buf << std::endl << termcolor::reset; \
    }

#define PRINT_GREEN(...)                                                       \
    {                                                                          \
        char buf[1024];                                                        \
        int n = sprintf(buf, __VA_ARGS__);                                     \
        std::cout << termcolor::green << buf << std::endl << termcolor::reset; \
    }

#define PRINT_YELLOW(...)                                                       \
    {                                                                           \
        char buf[1024];                                                         \
        int n = sprintf(buf, __VA_ARGS__);                                      \
        std::cout << termcolor::yellow << buf << std::endl << termcolor::reset; \
    }

#define PRINT_BLUE(...)                                                       \
    {                                                                         \
        char buf[1024];                                                       \
        int n = sprintf(buf, __VA_ARGS__);                                    \
        std::cout << termcolor::blue << buf << std::endl << termcolor::reset; \
    }

#define PRINT_MAGENTA(...)                                                       \
    {                                                                            \
        char buf[1024];                                                          \
        int n = sprintf(buf, __VA_ARGS__);                                       \
        std::cout << termcolor::magenta << buf << std::endl << termcolor::reset; \
    }

#define PRINT_CYAN(...)                                                       \
    {                                                                         \
        char buf[1024];                                                       \
        int n = sprintf(buf, __VA_ARGS__);                                    \
        std::cout << termcolor::cyan << buf << std::endl << termcolor::reset; \
    }

#define PRINT_WHITE(...)                                                       \
    {                                                                          \
        char buf[1024];                                                        \
        int n = sprintf(buf, __VA_ARGS__);                                     \
        std::cout << termcolor::white << buf << std::endl << termcolor::reset; \
    }

#define PRINT_BLACK(...)                                                       \
    {                                                                          \
        char buf[1024];                                                        \
        int n = sprintf(buf, __VA_ARGS__);                                     \
        std::cout << termcolor::black << buf << std::endl << termcolor::reset; \
    }
// comment this if you also comment code inside __linux__ and __APPLE__ macros above
#endif

//---------------------------------------------------------------------------------------
//! Get type of any arbitrary variable. Use string_view (from C++17). Supports all platforms.
/*!
    Ref: https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c
    Usage: type_name<decltype(variable)>();
*/
template <class T>
constexpr std::string_view type_name()
{
    using namespace std;
#ifdef __clang__
    string_view p = __PRETTY_FUNCTION__;
    return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
    string_view p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
    return string_view(p.data() + 36, p.size() - 36 - 1);
#else
    return string_view(p.data() + 49, p.find(';', 49) - 49);
#endif
#elif defined(_MSC_VER)
    string_view p = __FUNCSIG__;
    return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}

//---------------------------------------------------------------------------
//! Print a progress bar with given progress.
/*!
  Example:
  for (progress from 0.0 to 1.0)
      printProgressBar(progress);

  NOTE to ensure 'progress' can reach 1.0 exactly.
*/
inline void printProgressBar(float progress)
{
    int bar_width = 70;  // experienced width
    std::cout << "\r[";
    int pos = static_cast<int>(bar_width * progress);
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %" << std::flush;
    if (progress == 1.0)
        std::cout << std::endl;
}

#endif
