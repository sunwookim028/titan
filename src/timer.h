#ifndef _TIMER_H
#define _TIMER_H

#include <chrono>

#define FUNC_TIMER_START\
     auto func_start = std::chrono::high_resolution_clock::now();

#define FUNC_TIMER_END\
    auto func_end = std::chrono::high_resolution_clock::now();\
    *func_elapsed_ms = std::chrono::duration<double, std::milli>(func_end - func_start).count();

#endif
