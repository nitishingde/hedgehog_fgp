#include <iostream>
#include <oneapi/tbb.h>
#include <oneapi/tbb/task_arena.h>
#include <cassert>
#include "tbb_reduce.hpp"
#include "timer.hpp"

void tbb_playground() {
    oneapi::tbb::task_arena arena(40);

    int n = 100'000'000;
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = i + 1;
    }
    int expected = (n * (n + 1)) / 2;
    timer_start(tbb_reduce);
    arena.execute([&v]() {
        int tbbr = tbb_reduce<int>(v);
    });
    assert(tbbr == expected);
    timer_end(tbb_reduce);
    timer_report(tbb_reduce);
}

void benchmark_tbb_reduced() {
    oneapi::tbb::task_arena arena(20);

    size_t n = 100'000'000;
    size_t count = 100;

    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = 1.0 / (double)(i + 1);
    }
    timer_start(tbb_reduce);
    for (size_t i = 0; i < count; ++i) {
        arena.execute([&v]() {
            tbb_reduce<double>(v);
        });
    }
    timer_end(tbb_reduce);
    timer_report(tbb_reduce);
}

int main(){
    benchmark_tbb_reduced();
    return 0;
}
