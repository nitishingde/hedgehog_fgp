#include <CLI/CLI.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <numeric>
#include <print>

#include "common_utility.h"

using namespace std::string_literals;

template<typename ExecutionSpace, typename MemorySpace = ExecutionSpace::memory_space>
static void testParallelFor(const ExecutionSpace &executionSpace, const int32_t N, const int32_t ITERS) {
    constexpr auto a = 4.0f;

    const auto viewX = Kokkos::View<float*, MemorySpace>("x", N);
    const auto viewY = Kokkos::View<float*, MemorySpace>("y", N);
    const auto viewZ = Kokkos::View<float*, MemorySpace>("z", N);
    Kokkos::deep_copy(executionSpace, viewX, 1.f);
    Kokkos::deep_copy(executionSpace, viewY, 2.f);
    Kokkos::deep_copy(executionSpace, viewZ, 3.f);


    std::vector<double> times;
    for(int32_t it = 0; it < ITERS; ++it) {
        const auto start = std::chrono::steady_clock::now();
        Kokkos::parallel_for("SAXPY", Kokkos::RangePolicy(executionSpace, 0, N), KOKKOS_LAMBDA(const int32_t i) {
            viewZ(i) = a*viewX(i) + viewY(i);
        });
        executionSpace.fence();
        const auto end = std::chrono::steady_clock::now();
        times.emplace_back(toMilliSeconds(end-start));
    }

    if(constexpr auto ans = 6.f; 1.e-6 < std::abs(viewZ(0)-ans) or 1.e-6 < std::abs(viewZ(N-1)-ans)) {
        throw std::runtime_error("SAXPY implementation is incorrect!\n");
    }

    std::print("[{:16}][{:8}][Min {:8.3f}ms][AVG {:8.3f}ms][Max {:8.3f}ms]\n",
        "Kokkos-"s + executionSpace.name(),
        "SAXPY",
        *std::ranges::min_element(times),
        std::accumulate(times.begin(), times.end(), 0.0)/static_cast<double>(times.size()),
        *std::ranges::max_element(times)
    );
}

template<typename ExecutionSpace, typename MemorySpace = ExecutionSpace::memory_space>
static void testParallelReduce(const ExecutionSpace &executionSpace, const int32_t N, const int32_t ITERS) {
    const auto view = Kokkos::View<float*, MemorySpace>("x", N);
    Kokkos::parallel_for("INIT", Kokkos::RangePolicy(executionSpace, 0, N), KOKKOS_LAMBDA(const int32_t i) {
        view(i) = static_cast<float>(i);
    });

    float result = std::numeric_limits<float>::min();
    std::vector<double> times;
    for(int32_t it = 0; it < ITERS; ++it) {
        const auto start = std::chrono::steady_clock::now();
        result = std::numeric_limits<float>::min();
        Kokkos::parallel_reduce("REDUCE", Kokkos::RangePolicy(executionSpace, 0, N), KOKKOS_LAMBDA(const int32_t i, float &partialResult) {
            partialResult = Kokkos::max(partialResult, view(i));
        }, Kokkos::Max<float>(result));
        executionSpace.fence();
        const auto end = std::chrono::steady_clock::now();
        times.emplace_back(toMilliSeconds(end-start));
    }

    if(1.e-6 < std::abs(result - static_cast<float>(N))) {
        throw std::runtime_error("REDUCE implementation is incorrect!\n");
    }

    std::print("[{:16}][{:8}][Min {:8.3f}ms][AVG {:8.3f}ms][Max {:8.3f}ms]\n",
        "Kokkos-"s + executionSpace.name(),
        "REDUCE",
        *std::ranges::min_element(times),
        std::accumulate(times.begin(), times.end(), 0.0)/static_cast<double>(times.size()),
        *std::ranges::max_element(times)
    );
}

template<typename ExecutionSpace, typename MemorySpace = ExecutionSpace::memory_space>
static void testParallelScan(const ExecutionSpace &executionSpace, const int32_t N, const int32_t ITERS) {
    const auto srcView = Kokkos::View<int32_t*, MemorySpace>("src", N);
    auto dstView = Kokkos::View<int32_t*, MemorySpace>("dst", (N+1)/2);
    Kokkos::parallel_for("INIT", Kokkos::RangePolicy(executionSpace, 0, N), KOKKOS_LAMBDA(const int32_t i) {
        srcView(i) = i;
    });

    std::vector<double> times;
    for(int32_t it = 0; it < ITERS; ++it) {
        const auto start = std::chrono::steady_clock::now();
        Kokkos::parallel_scan("SCAN", Kokkos::RangePolicy(executionSpace, 0, N), KOKKOS_LAMBDA(const int32_t i, int32_t &count, bool isFinalPass) {
            if(srcView[i]%2 == 1) return;

            if(isFinalPass) {
                dstView[count] = srcView[i];
            }
            count++;
        });
        executionSpace.fence();
        const auto end = std::chrono::steady_clock::now();
        times.emplace_back(toMilliSeconds(end-start));
    }

    if(dstView(0)%2 != 0 or dstView(((N+1)/2)-1)%2 != 0) {
        throw std::runtime_error("SCAN implementation is incorrect!\n");
    }

    std::print("[{:16}][{:8}][Min {:8.3f}ms][AVG {:8.3f}ms][Max {:8.3f}ms]\n",
        "Kokkos-"s + executionSpace.name(),
        "COPY_IF",
        *std::ranges::min_element(times),
        std::accumulate(times.begin(), times.end(), 0.0)/static_cast<double>(times.size()),
        *std::ranges::max_element(times)
    );
}

int main(int argc, char **argv) {
    CLI::App app{"Kokkos Baseline"};
    auto problemSize = 100'000'000;

    app.add_option("-N", problemSize, "problem size")
       ->check(CLI::PositiveNumber)
       ->capture_default_str();

    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    const auto kokkosSG  = Kokkos::ScopeGuard();

    constexpr auto ITERS = 10;

    const auto executionSpace = ExecutionSpace();
    executionSpace.print_configuration(std::cout);
    std::println();

    testParallelFor(executionSpace, problemSize, ITERS);
    testParallelReduce(executionSpace, problemSize, ITERS);
    testParallelScan(executionSpace, problemSize, ITERS);

    return 0;
}
