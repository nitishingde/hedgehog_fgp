#include <iostream>
#include <oneapi/tbb.h>
#include <oneapi/tbb/task_arena.h>
#include <cstdlib>
#include "tbb_reduce.hpp"
#include "reduce_graph.hpp"
#include "timer.hpp"

void tbb_playground(size_t n) {
    oneapi::tbb::task_arena arena(40);

    std::vector<size_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = i + 1;
    }
    size_t expected = (n * (n + 1)) / 2;
    timer_start(tbb_reduce);
    arena.execute([&v, expected]() {
        size_t tbbr = tbb_reduce<size_t>(v);
        if (tbbr != expected) {
            printf("error: not equal\n");
            exit(1);
        }
    });
    timer_end(tbb_reduce);
    timer_report(tbb_reduce);
}

void benchmark_tbb_reduced(size_t n, size_t iter) {
    oneapi::tbb::task_arena arena(20);

    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = 1.0 / (double)(i + 1);
    }
    timer_start(tbb_reduce);
    for (size_t i = 0; i < iter; ++i) {
        arena.execute([&v]() {
            tbb_reduce<double>(v);
        });
    }
    timer_end(tbb_reduce);
    timer_report(tbb_reduce);
}

void benchmark_hh_reduced(size_t n, size_t iter) {
    oneapi::tbb::task_arena arena(20);

    auto v = std::make_shared<std::vector<double>>(n);
    for (size_t i = 0; i < n; ++i) {
        (*v)[i] = 1.0 / (double)(i + 1);
    }
    timer_start(tbb_reduce);
    ReduceGraph<double> graph(10, 2);
    graph.executeGraph();
    for (size_t i = 0; i < iter; ++i) {
        graph.pushData(v);
    }
    graph.finishPushingData();
    graph.waitForTermination();
    timer_end(tbb_reduce);
    timer_report(tbb_reduce);
}

int main(){
    size_t n = 100'000'000;
    size_t iter = 10;
    // tbb_playground(n);
    benchmark_tbb_reduced(n, iter);
    benchmark_hh_reduced(n, iter);
    return 0;
}
