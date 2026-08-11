#include <iostream>
#include <oneapi/tbb.h>
#include <oneapi/tbb/task_arena.h>
#include <cstdlib>
#include "tbb_functions.hpp"
#include "hh_graphs.hpp"
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

void benchmark_tbb_reduce(size_t n, size_t iter) {
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

void benchmark_tbb_saxpy(size_t n, size_t iter) {
    oneapi::tbb::task_arena arena(20);

    std::vector<double> x(n, 1.0);
    std::vector<double> y(n, 2.0);
    std::vector<double> z(n, 3.0);
    double a = 4.0;

    timer_start(tbb_saxpy);
    for (size_t i = 0; i < iter; ++i) {
        arena.execute([&]() {
            tbb_saxpy<double>(z, a, x, y);
        });
    }
    timer_end(tbb_saxpy);
    timer_report(tbb_saxpy);
}

void benchmark_hh_reduce(size_t n, size_t iter) {
    auto v = std::make_shared<std::vector<double>>(n);
    for (size_t i = 0; i < n; ++i) {
        (*v)[i] = 1.0 / (double)(i + 1);
    }
    timer_start(hh_reduce);
    ReduceGraph<double> graph(1, 20);
    graph.executeGraph();
    for (size_t i = 0; i < iter; ++i) {
        graph.pushData(v);
    }
    graph.finishPushingData();
    graph.waitForTermination();
    timer_end(hh_reduce);
    timer_report(hh_reduce);
    graph.createDotFile("tbb_reduce.dot", hh::ColorScheme::EXECUTION, hh::StructureOptions::QUEUE);
}

void benchmark_hh_saxpy(size_t n, size_t iter) {
    std::vector<double> x(n, 1.0);
    std::vector<double> y(n, 2.0);
    std::vector<double> z(n, 3.0);
    double a = 4.0;
    auto data = std::make_shared<SAXPYData<double>>(x, y, z, a);

    timer_start(hh_saxpy);
    SAXPYGraph<double> graph(1, 20); // WARN: do not increase the task count!
    graph.executeGraph();
    for (size_t i = 0; i < iter; ++i) {
        graph.pushData(data);
    }
    graph.finishPushingData();
    graph.waitForTermination();
    timer_end(hh_saxpy);
    timer_report(hh_saxpy);
    graph.createDotFile("tbb_saxpy.dot", hh::ColorScheme::EXECUTION, hh::StructureOptions::QUEUE);
}

int main(){
    size_t n = 100'000'000;
    size_t iter = 10;
    // tbb_playground(n);
    benchmark_tbb_reduce(n, iter);
    benchmark_hh_reduce(n, iter);
    benchmark_tbb_saxpy(n, iter);
    benchmark_hh_saxpy(n, iter);
    return 0;
}
