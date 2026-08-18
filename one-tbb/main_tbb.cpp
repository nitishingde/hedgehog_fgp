#include <CLI/CLI.hpp>
#include <iostream>
#include <oneapi/tbb.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_scan.h>
#include <numeric>
#include <print>
#include "common_utility.hpp"

using namespace std::string_literals;

static void testParallelFor(const int32_t N, const int32_t ITERS) {
    constexpr auto a = 4.0f;
    std::vector<float> x(N, 1.f);
    std::vector<float> y(N, 2.f);
    std::vector<float> z(N, 3.f);

    std::vector<double> times;
    for(int32_t it = 0; it < ITERS; ++it) {
        const auto start = std::chrono::steady_clock::now();
        tbb::parallel_for(0, N, [&](const int32_t i) {
            z[i] = a*x[i] + y[i];
        });
        const auto end = std::chrono::steady_clock::now();
        times.emplace_back(toMilliSeconds(end-start));
    }

    if(constexpr auto ans = 6.f; 1.e-6 < std::abs(z[0]-ans) or 1.e-6 < std::abs(z[N-1]-ans)) {
        throw std::runtime_error("SAXPY implementation is incorrect!\n");
    }

    std::print("[{:16}][{:8}][Min {:8.3f}ms][AVG {:8.3f}ms][Max {:8.3f}ms]\n",
        "TBB",
        "SAXPY",
        *std::ranges::min_element(times),
        std::accumulate(times.begin(), times.end(), 0.0)/static_cast<double>(times.size()),
        *std::ranges::max_element(times)
    );
}

static void testParallelReduce(const int32_t N, const int32_t ITERS) {
    std::vector<float> data(N);
    tbb::parallel_for(0, N, [&](const int32_t i) {
        data[i] = static_cast<float>(i + 1);
    });

    float result = std::numeric_limits<float>::min();
    std::vector<double> times;
    for(int32_t it = 0; it < ITERS; ++it) {
        const auto start = std::chrono::steady_clock::now();
        result = tbb::parallel_reduce(tbb::blocked_range<int32_t>(0, N), std::numeric_limits<float>::min(),
            [&](tbb::blocked_range<int32_t> const& r, float acc) -> float {
                for (int32_t i = r.begin(); i != r.end(); i++) {
                    acc = std::max(acc, data[i]);
                }
                return acc;
            },
            [](float lhs, float rhs) -> float { return std::max(lhs, rhs); });
        const auto end = std::chrono::steady_clock::now();
        times.emplace_back(toMilliSeconds(end-start));
    }

    if(1.e-6 < std::abs(result - static_cast<float>(N))) {
        throw std::runtime_error("REDUCE implementation is incorrect!\n");
    }

    std::print("[{:16}][{:8}][Min {:8.3f}ms][AVG {:8.3f}ms][Max {:8.3f}ms]\n",
        "TBB",
        "REDUCE",
        *std::ranges::min_element(times),
        std::accumulate(times.begin(), times.end(), 0.0)/static_cast<double>(times.size()),
        *std::ranges::max_element(times)
    );
}

static void testParallelScan(const int32_t N, const int32_t ITERS) {
    std::vector<int32_t> srcData(N);
    std::vector<int32_t> dstData((N+1)/2);
    tbb::parallel_for(0, N, [&](int32_t i) {
        srcData[i] = i;
    });

    std::vector<double> times;
    for(int32_t it = 0; it < ITERS; ++it) {
        const auto start = std::chrono::steady_clock::now();
        tbb::parallel_scan(
            tbb::blocked_range<int32_t>(0, N),
            0,
            [&](const tbb::blocked_range<int32_t> &r, int32_t count, bool isFinalPass) -> int32_t {
                for(int i = r.begin(); i < r.end(); ++i) {
                    if(srcData[i]%2 == 1) continue;

                    if(isFinalPass) {
                        dstData[count] = srcData[i];
                    }
                    count++;
                }
                return count;
            },
            [](int32_t rhs, int32_t lhs) -> int32_t { return lhs + rhs; }
        );
        const auto end = std::chrono::steady_clock::now();
        times.emplace_back(toMilliSeconds(end-start));
    }

    if(dstData[0]%2 != 0 or dstData[((N+1)/2)-1]%2 != 0) {
        throw std::runtime_error("SCAN implementation is incorrect!\n");
    }

    std::print("[{:16}][{:8}][Min {:8.3f}ms][AVG {:8.3f}ms][Max {:8.3f}ms]\n",
        "TBB",
        "COPY_IF",
        *std::ranges::min_element(times),
        std::accumulate(times.begin(), times.end(), 0.0)/static_cast<double>(times.size()),
        *std::ranges::max_element(times)
    );
}

int main(int argc, char **argv) {
    CLI::App app{"TBB Baseline"};
    auto problemSize = 100'000'000;

    app.add_option("-N", problemSize, "problem size")
       ->check(CLI::PositiveNumber)
       ->capture_default_str();
    CLI11_PARSE(app, argc, argv);

    constexpr auto ITERS = 10;

    testParallelFor(problemSize, ITERS);
    testParallelReduce(problemSize, ITERS);
    testParallelScan(problemSize, ITERS);

    return 0;
}
