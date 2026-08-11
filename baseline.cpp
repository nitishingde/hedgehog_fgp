#include <CLI/CLI.hpp>
#include <hedgehog.h>
#include <print>
#include "common_utility.h"

namespace {
    struct SaxpyWorkUnit {
        std::span<float> x;
        std::span<float> y;
        std::span<float> z;
        float            a;
    };

    struct SaxpyData {
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> z;
        float              a;
    };

    class ParallelForCT final: public hh::AbstractTask<1, SaxpyWorkUnit, SaxpyWorkUnit> {
    public:
        explicit ParallelForCT(const int32_t computeThreads):
            AbstractTask("", computeThreads, false) {}

        void execute(std::shared_ptr<SaxpyWorkUnit> data) override {
            auto [x, y, z, a] = *data;
            for(int32_t i = 0; i < z.size(); ++i) {
                z[i] = a*x[i] + y[i];
            }
            this->addResult(data);
        }

        std::shared_ptr<AbstractTask> copy() override {
            return std::make_shared<ParallelForCT>(this->numberThreads());
        }
    };

    class ParallelForSM final: public hh::AbstractTask<2, SaxpyData, SaxpyWorkUnit, SaxpyData, SaxpyWorkUnit, std::vector<double>> {
    private:
        int32_t                               computeThreads_ = 1;
        int32_t                               iter_           = 10;
        std::shared_ptr<SaxpyData>            data_           = nullptr;
        int32_t                               ttl_            = 0;
        std::atomic_bool                      canTerminate_   = false;
        std::shared_ptr<std::vector<double>>  times_          = std::make_shared<std::vector<double>>();
        std::chrono::steady_clock::time_point start_          = {};

    public:
        explicit ParallelForSM(const int32_t computeThreads, const int32_t iter = 10):
            AbstractTask("SM", 1, false),
            computeThreads_(computeThreads), iter_(iter) {
            times_->reserve(2*iter);
        }

        void execute(std::shared_ptr<SaxpyData> data) override {
            start_ = std::chrono::steady_clock::now();
            auto &[x, y, z, a] = *data;
            assert(x.size() == y.size());
            assert(x.size() == z.size());

            const auto N     = static_cast<int32_t>(z.size());
            const auto range = std::max(1000'000, (N+computeThreads_-1)/computeThreads_);
            ttl_ = 0;
            for(int32_t i = 0; i < N; i += range, ++ttl_) {
                const auto workLen = std::min(range, N-i);
                this->addResult(std::make_shared<SaxpyWorkUnit>(
                    std::span(&x[i], workLen),
                    std::span(&y[i], workLen),
                    std::span(&z[i], workLen),
                    a
                ));
            }
            data_ = data;
            canTerminate_.store(false);
        }

        void execute(const std::shared_ptr<SaxpyWorkUnit> workUnit) override {
            if(--ttl_; ttl_ == 0) {
                times_->emplace_back(toMilliSeconds(std::chrono::steady_clock::now()-start_));
                iter_--;
                if(iter_) {
                    execute(data_);
                }
                else {
                    this->addResult(data_);
                    this->addResult(times_);
                    canTerminate_.store(true);
                    data_ = nullptr;
                }
            }
        }

        [[nodiscard]] bool canTerminate() const override {
            return canTerminate_.load();
        }
    };

    using ReductionData     = std::vector<float>;
    using ReductionWorkUnit = std::span<float>;
    using ReductionResult   = float;

    class ParallelReduceCT final: public hh::AbstractTask<1, ReductionWorkUnit, ReductionResult> {
    public:
        explicit ParallelReduceCT(const int32_t computeThreads):
            AbstractTask("ParallelReduceCT", computeThreads, false) {}

        void execute(const std::shared_ptr<ReductionWorkUnit> workUnit) override {
            const auto data   = *workUnit;
            auto       result = std::numeric_limits<ReductionResult>::min();
            for(const auto el: data) {
                result = std::max(result, el);
            }
            this->addResult(std::make_shared<ReductionResult>(result));
        }

        std::shared_ptr<AbstractTask> copy() override {
            return std::make_shared<ParallelReduceCT>(this->numberThreads());
        }
    };

    class ParallelReduceSM final: public hh::AbstractTask<2, ReductionData, ReductionResult, ReductionWorkUnit, std::vector<double>, ReductionResult> {
    private:
        ReductionResult                       result_         = {};
        int32_t                               computeThreads_ = {};
        int32_t                               iter_           = 10;
        std::shared_ptr<ReductionData>        data_           = nullptr;
        int32_t                               ttl_            = 0;
        std::atomic_bool                      canTerminate_   = false;
        std::shared_ptr<std::vector<double>>  times_          = std::make_shared<std::vector<double>>();
        std::chrono::steady_clock::time_point start_          = {};

    public:
        explicit ParallelReduceSM(const int32_t computeThreads, const int32_t iter = 10):
            AbstractTask("ParallelReduceSM", 1, false),
            computeThreads_(computeThreads), iter_(iter) {
            times_->reserve(iter*2);
        }

        void execute(const std::shared_ptr<ReductionData> data) override {
            start_ = std::chrono::steady_clock::now();
            auto &array = *data;

            const auto N     = static_cast<int32_t>(array.size());
            const auto range = std::max(1000'000, (N+computeThreads_-1)/computeThreads_);
            ttl_ = 0;
            for(int32_t i = 0; i < N; i += range, ++ttl_) {
                const auto workLen = std::min(range, N-i);
                this->addResult(std::make_shared<ReductionWorkUnit>(&array[i], workLen));
            }
            data_ = data;
            canTerminate_.store(false);
            result_ = std::numeric_limits<float>::min();
        }

        void execute(const std::shared_ptr<ReductionResult> result) override {
            result_ = std::max(result_, *result);
            if(--ttl_; ttl_ == 0) {
                times_->emplace_back(toMilliSeconds(std::chrono::steady_clock::now()-start_));
                iter_--;
                if(iter_) {
                    execute(data_);
                }
                else {
                    this->addResult(std::make_shared<float>(result_));
                    this->addResult(times_);
                    canTerminate_.store(true);
                    data_ = nullptr;
                }
            }
        }

        [[nodiscard]] bool canTerminate() const override {
            return canTerminate_.load();
        }
    };
}

static void testParallelFor(const auto N, const int32_t computeThreads, const int32_t ITERS) {
    const auto data = std::make_shared<SaxpyData>(
        std::vector(N, 1.f),
        std::vector(N, 2.f),
        std::vector(N, 3.f),
        4.f
    );

    auto       graph = hh::Graph<1, SaxpyData, SaxpyData, std::vector<double>>();
    const auto pfCT  = std::make_shared<ParallelForCT>(computeThreads);
    const auto pfSM  = std::make_shared<ParallelForSM>(computeThreads, ITERS);

    graph.inputs(pfSM);
    graph.edges(pfSM, pfCT);
    graph.edges(pfCT, pfSM);
    graph.outputs(pfSM);
    graph.executeGraph();

    const auto start = std::chrono::steady_clock::now();
    graph.pushData(data);
    graph.finishPushingData();
    graph.waitForTermination();
    const auto end = std::chrono::steady_clock::now();

    graph.createDotFile("baseline_saxpy.dot", hh::ColorScheme::EXECUTION, hh::StructureOptions::QUEUE);
    while(auto result = graph.getBlockingResult()) {
        std::visit(hh::ResultVisitor{
            [](const std::shared_ptr<SaxpyData> &val) {
                auto &[x, y, z, a] = *val;
                if(constexpr auto ans = 6.f; 1.e-6 < std::abs(z.front()-ans) or 1.e-6 < std::abs(z.back()-ans)) {
                    throw std::runtime_error("SAXPY implementation is incorrect!\n");
                }
            },
            [start, end, ITERS](const std::shared_ptr<std::vector<double>> &times) {
                std::print("[{:16}][{:8}][Min {:8.3f}ms][AVG {:8.3f}ms][Max {:8.3f}ms][Graph/{} {:8.3f}ms]\n",
                    "Hedgehog",
                    "SAXPY",
                    *std::ranges::min_element(*times),
                    std::accumulate(times->begin(), times->end(), 0.0)/static_cast<double>(times->size()),
                    *std::ranges::max_element(*times),
                    ITERS,
                    toMilliSeconds(end-start)/ITERS
                );
            }
        }, *result);
    }
}

static void testParallelReduce(const int32_t N, const int32_t computeThreads, const int32_t ITERS) {
    const auto data = std::make_shared<std::vector<float>>(N);
    std::ranges::iota(*data, 0);

    auto       graph = hh::Graph<1, ReductionData, ReductionResult, std::vector<double>>();
    const auto pfCT  = std::make_shared<ParallelReduceCT>(computeThreads);
    const auto pfSM  = std::make_shared<ParallelReduceSM>(computeThreads, ITERS);

    graph.inputs(pfSM);
    graph.edges(pfSM, pfCT);
    graph.edges(pfCT, pfSM);
    graph.outputs(pfSM);
    graph.executeGraph();

    const auto start = std::chrono::steady_clock::now();
    graph.pushData(data);
    graph.finishPushingData();
    graph.waitForTermination();
    const auto end = std::chrono::steady_clock::now();

    graph.createDotFile("baseline_reduce.dot", hh::ColorScheme::EXECUTION, hh::StructureOptions::QUEUE);
    while(auto result = graph.getBlockingResult()) {
        std::visit(hh::ResultVisitor{
            [N](const std::shared_ptr<ReductionResult> &val) {
                if(1.e-6 < std::abs(*val- static_cast<ReductionResult>(N))) {
                    throw std::runtime_error("REDUCE implementation is incorrect!\n");
                }
            },
            [start, end, ITERS](const std::shared_ptr<std::vector<double>> &times) {
                std::print("[{:16}][{:8}][Min {:8.3f}ms][AVG {:8.3f}ms][Max {:8.3f}ms][Graph/{} {:8.3f}ms]\n",
                    "Hedgehog",
                    "REDUCE",
                    *std::ranges::min_element(*times),
                    std::accumulate(times->begin(), times->end(), 0.0)/static_cast<double>(times->size()),
                    *std::ranges::max_element(*times),
                    ITERS,
                    toMilliSeconds(end-start)/ITERS
                );
            }
        }, *result);
    }
}

int main(const int argc, char **argv) {
    constexpr auto ITERS = 10;

    CLI::App app{"Hedgehog Baseline"};
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

    return 0;
}
