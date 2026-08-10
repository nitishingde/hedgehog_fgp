#include <iostream>
#include <oneapi/tbb.h>
#include <cassert>
#include "tbb_reduce.hpp"
#include "timer.hpp"

int main(){
    int n = 1'000'000'000;
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = i + 1;
    }
    int expected = (n * (n + 1)) / 2;
    timer_start(tbb_reduce);
    int tbbr = tbb_reduce(v);
    assert(tbbr == expected);
    timer_end(tbb_reduce);
    timer_report(tbb_reduce);
    return 0;
}
