#include <CLI/CLI.hpp>
#include <iostream>
#include <oneapi/tbb.h>
#include <oneapi/tbb/task_arena.h>
#include <cstdlib>
#include "hh_graphs.hpp"
#include "timer.hpp"
#include "common_utility.hpp"
#include <print>

static void testParallelFor(const auto N, const int32_t computeThreads, const int32_t ITERS) {
    const auto data = std::make_shared<SAXPYData<float>>(
        std::vector(N, 1.f),
        std::vector(N, 2.f),
        std::vector(N, 3.f),
        4.f
    );

    SAXPYGraph<float> graph(1, computeThreads);
    graph.executeGraph(true);

    auto times = std::vector<double>();
    times.reserve(ITERS*2);
    for(int32_t it = 0; it < ITERS; ++it) {
        const auto start  = std::chrono::steady_clock::now();
        graph.pushData(data);
        auto       result = graph.getBlockingResult();
        const auto end    = std::chrono::steady_clock::now();

        times.emplace_back(toMilliSeconds(end-start));
        auto &[x, y, z, a] = *std::get<std::shared_ptr<SAXPYData<float>>>(*result);
        if(constexpr auto ans = 6.f; 1.e-6 < std::abs(z.front()-ans) or 1.e-6 < std::abs(z.back()-ans)) {
            throw std::runtime_error("SAXPY implementation is incorrect!\n");
        }
    }
    graph.finishPushingData();
    graph.waitForTermination();

    graph.createDotFile("hh_tbb_saxpy.dot", hh::ColorScheme::EXECUTION, hh::StructureOptions::QUEUE, hh::InputOptions::SEPARATED);
    std::print("[{:16}][{:8}][Min {:8.3f}ms][AVG {:8.3f}ms][Max {:8.3f}ms]\n",
        "Hedgehog TBB",
        "SAXPY",
        *std::ranges::min_element(times),
        std::accumulate(times.begin(), times.end(), 0.0)/static_cast<double>(times.size()),
        *std::ranges::max_element(times)
    );
}

static void testParallelReduce(const int32_t N, const int32_t computeThreads, const int32_t ITERS) {
    const auto data = std::make_shared<std::vector<float>>(N);
    std::ranges::iota(*data, 1);

    ReduceGraph<float> graph(1, computeThreads);
    graph.executeGraph(true);

    auto times = std::vector<double>();
    times.reserve(ITERS*2);
    for(int32_t it = 0; it < ITERS; ++it) {
        const auto start  = std::chrono::steady_clock::now();
        graph.pushData(data);
        auto       result = graph.getBlockingResult();
        const auto end    = std::chrono::steady_clock::now();

        times.emplace_back(toMilliSeconds(end-start));
        if(const auto value = *std::get<std::shared_ptr<float>>(*result); 1.e-6 < std::abs(value-static_cast<float>(N))) {
           throw std::runtime_error("Reduce implementation is incorrect!\n");
        }
    }
    graph.finishPushingData();
    graph.waitForTermination();

    graph.createDotFile("hh_tbb_reduce.dot", hh::ColorScheme::EXECUTION, hh::StructureOptions::QUEUE, hh::InputOptions::SEPARATED);
    std::print("[{:16}][{:8}][Min {:8.3f}ms][AVG {:8.3f}ms][Max {:8.3f}ms]\n",
        "Hedgehog TBB",
        "REDUCE",
        *std::ranges::min_element(times),
        std::accumulate(times.begin(), times.end(), 0.0)/static_cast<double>(times.size()),
        *std::ranges::max_element(times)
    );
}

static void testParallelScan(const int32_t N, const int32_t computeThreads, const int32_t ITERS) {
    const auto data = std::make_shared<CopyIfData>(std::vector<int32_t>(N), std::vector((N+1)/2, 1));
    std::ranges::iota(data->src, 0);

    ScanGraph<int32_t> graph(1, computeThreads);
    graph.executeGraph(true);

    auto times = std::vector<double>();
    times.reserve(ITERS*2);
    for(int32_t it = 0; it < ITERS; ++it) {
        const auto start  = std::chrono::steady_clock::now();
        graph.pushData(data);
        auto       result = graph.getBlockingResult();
        const auto end    = std::chrono::steady_clock::now();

        times.emplace_back(toMilliSeconds(end-start));
        if(const auto &[src, dst] = *std::get<std::shared_ptr<CopyIfData>>(*result); dst.front()%2 != 0 or dst.back()%2 != 0) {
            throw std::runtime_error("Scan implementation is incorrect!\n");
        }
    }
    graph.finishPushingData();
    graph.waitForTermination();

    graph.createDotFile("baseline_scan.dot", hh::ColorScheme::EXECUTION, hh::StructureOptions::QUEUE, hh::InputOptions::SEPARATED);
    std::print("[{:16}][{:8}][Min {:8.3f}ms][AVG {:8.3f}ms][Max {:8.3f}ms]\n",
        "Hedgehog TBB",
        "COPY_IF",
        *std::ranges::min_element(times),
        std::accumulate(times.begin(), times.end(), 0.0)/static_cast<double>(times.size()),
        *std::ranges::max_element(times)
    );
}

int main(const int argc, char **argv) {
    constexpr auto ITERS = 10;

    CLI::App app{"Hedgehog TBB"};
    auto problemSize = 100'000'000;
    auto computeThreads = std::max(1, static_cast<int32_t>(std::thread::hardware_concurrency()));

    app.add_option("-N", problemSize, "problem size")
       ->check(CLI::PositiveNumber)
       ->capture_default_str();
    app.add_option("-t,--threads", computeThreads, "compute threads")
       ->check(CLI::PositiveNumber)
       ->capture_default_str();
    CLI11_PARSE(app, argc, argv);

    testParallelFor(problemSize, computeThreads, ITERS);
    testParallelReduce(problemSize, computeThreads, ITERS);
    testParallelScan(problemSize, computeThreads, ITERS);

    return 0;
}
