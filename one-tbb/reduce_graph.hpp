#ifndef REDUCE_GRAPH
#define REDUCE_GRAPH
#include <hedgehog.h>
#include "tbb_task.hpp"
#include "tbb_reduce.hpp"

template <typename T>
class ReduceTask : public hh::TBBTask<1, std::vector<T>, T> {
    size_t numberSubThreads;

  public:
    ReduceTask(size_t numberThreads, size_t numberSubThreads)
        : hh::TBBTask<1, std::vector<T>, T>("ReduceTask", numberThreads, numberSubThreads),
          numberSubThreads(numberSubThreads) {}

    void execute(std::shared_ptr<std::vector<T>> values) {
        this->addResult(std::make_shared<T>(tbb_reduce<T>(*values)));
    }

    std::shared_ptr<hh::AbstractTask<1, std::vector<T>, T>> copy() override {
        return std::make_shared<ReduceTask<T>>(this->numberThreads(), this->numberSubThreads);
    }
};

template <typename T>
class ReduceGraph : public hh::Graph<1, std::vector<T>, T> {
  public:
    ReduceGraph(size_t numberThreads, size_t numberSubThreads) : hh::Graph<1, std::vector<T>, T>("ReduceGraph") {
        auto reduceTask = std::make_shared<ReduceTask<T>>(numberThreads, numberSubThreads);
        this->inputs(reduceTask);
        this->outputs(reduceTask);
    }
};

#endif
