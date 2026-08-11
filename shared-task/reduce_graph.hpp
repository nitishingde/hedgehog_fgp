#ifndef REDUCE_GRAPH
#define REDUCE_GRAPH
#include <hedgehog.h>
#include <vector>
#include <cstdio>
#include "shared_task.hpp"

template <typename T>
class ReduceTask : public SharedTask<1, std::vector<T>, T> {
    std::vector<T> accs_;

  public:
    ReduceTask(size_t numberThreads, size_t numberSubThreads)
        : SharedTask<1, std::vector<T>, T>("ReduceTask", numberThreads, numberSubThreads),
          accs_(numberSubThreads) {}

    void execute(std::shared_ptr<std::vector<T>> values) {
        this->process([&](size_t threadIndex, size_t threadCount) {
            size_t blockSize = values->size() / threadCount + (values->size() % threadCount > 0 ? 1 : 0);
            size_t startIndex = threadIndex * blockSize;
            size_t endIndex = std::min(startIndex + blockSize, values->size());
            T acc = 0;

            for (size_t i = startIndex; i < endIndex; ++i) {
                acc += (*values)[i];
            }
            accs_[threadIndex] = acc;
        });

        T sum = 0;
        for (T &acc : accs_) {
            sum += acc;
            acc = {};
        }
        this->addResult(std::make_shared<T>(sum));
    }
};

template <typename T>
class ReduceGraph : public hh::Graph<1, std::vector<T>, T> {
  public:
    ReduceGraph() : hh::Graph<1, std::vector<T>, T>("ReduceGraph") {
        auto reduceTask = std::make_shared<ReduceTask<T>>(1, 20);
        this->inputs(reduceTask);
        this->outputs(reduceTask);
    }
};

#endif
