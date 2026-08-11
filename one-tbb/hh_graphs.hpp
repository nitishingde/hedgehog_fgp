#ifndef HH_GRAPHS
#define HH_GRAPHS
#include <hedgehog.h>
#include "tbb_task.hpp"
#include "tbb_functions.hpp"

// reduce //////////////////////////////////////////////////////////////////////

template <typename T>
class ReduceTask : public hh::TBBTask<1, std::vector<T>, T> {
  public:
    ReduceTask(size_t numberThreads, size_t numberSubThreads)
        : hh::TBBTask<1, std::vector<T>, T>("ReduceTask", numberThreads, numberSubThreads) {}

    void execute(std::shared_ptr<std::vector<T>> values) {
        this->addResult(std::make_shared<T>(tbb_reduce<T>(*values)));
    }

    std::shared_ptr<hh::AbstractTask<1, std::vector<T>, T>> copy() override {
        return std::make_shared<ReduceTask<T>>(this->numberThreads(), this->numberSubThreads());
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

// saxpy ///////////////////////////////////////////////////////////////////////

template <typename T>
struct SAXPYData {
    std::vector<T> x;
    std::vector<T> y;
    std::vector<T> z;
    T a;
};

template <typename T>
class SAXPYTask : public hh::TBBTask<1, SAXPYData<T>, SAXPYData<T>> {
  public:
    SAXPYTask(size_t numberThreads, size_t numberSubThreads)
        : hh::TBBTask<1, SAXPYData<T>, SAXPYData<T>>("SAXPYTask", numberThreads, numberSubThreads) {}

    void execute(std::shared_ptr<SAXPYData<T>> data) {
        auto &[x, y, z, a] = *data;
        tbb_saxpy<T>(z, a, x, y);
        this->addResult(data);
    }

    std::shared_ptr<hh::AbstractTask<1, SAXPYData<T>, SAXPYData<T>>> copy() override {
        return std::make_shared<SAXPYTask<T>>(this->numberThreads(), this->numberSubThreads());
    }
};

template <typename T>
class SAXPYGraph : public hh::Graph<1, SAXPYData<T>, SAXPYData<T>> {
  public:
    SAXPYGraph(size_t numberThreads, size_t numberSubThreads) : hh::Graph<1, SAXPYData<T>, SAXPYData<T>>("SAXPYGraph") {
        auto saxpyTask = std::make_shared<SAXPYTask<T>>(numberThreads, numberSubThreads);
        this->inputs(saxpyTask);
        this->outputs(saxpyTask);
    }
};

#endif
