#ifndef HH_GRAPHS
#define HH_GRAPHS
#include <hedgehog.h>
#include "tbb_task.hpp"
#include <oneapi/tbb.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_scan.h>

// reduce //////////////////////////////////////////////////////////////////////

template <typename T>
class ReduceTask : public hh::TBBTask<1, std::vector<T>, T> {
  public:
    ReduceTask(size_t numberThreads, size_t numberSubThreads)
        : hh::TBBTask<1, std::vector<T>, T>("ReduceTask", numberThreads, numberSubThreads) {}

    void execute(std::shared_ptr<std::vector<T>> values) {
        T max = tbb::parallel_reduce(
            tbb::blocked_range<int32_t>(0, values->size()), std::numeric_limits<float>::min(),
            [&](tbb::blocked_range<int32_t> const& r, T init) -> T {
                for (int32_t i = r.begin(); i != r.end(); i++) {
                    init = std::max(init, values->operator[](i));
                }
                return init;
            },
            [](T lhs, T rhs) -> T {
                return std::max(lhs, rhs);
            }
        );
        this->addResult(std::make_shared<T>(max));
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
        tbb::parallel_for(0UL, z.size(), 1UL, [&](size_t i) {
            z[i] = a * x[i] + y[i];
        });
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

// scan ////////////////////////////////////////////////////////////////////////

struct CopyIfData {
    std::vector<int32_t> src;
    std::vector<int32_t> dst;
};

class ScanTask : public hh::TBBTask<1, CopyIfData, CopyIfData> {
  public:
    ScanTask(size_t numberThreads, size_t numberSubThreads)
        : hh::TBBTask<1, CopyIfData, CopyIfData>("ScanTask", numberThreads, numberSubThreads) {}

    void execute(std::shared_ptr<CopyIfData> data) {
        tbb::parallel_scan(
            tbb::blocked_range<int32_t>(0, data->src.size()),
            0,
            [&](const tbb::blocked_range<int32_t>& r, int32_t count, bool isFinalPass) -> int32_t {
                for(int i = r.begin(); i < r.end(); ++i) {
                    if(data->src[i]%2 == 1) continue;

                    if(isFinalPass) {
                        data->dst[count] = data->src[i];
                    }
                    count++;
                }
                return count;
            },
            [](int32_t left, int32_t right) {
                return left + right;
            }
        );
        this->addResult(data);
    }

    std::shared_ptr<hh::AbstractTask<1, CopyIfData, CopyIfData>> copy() override {
        return std::make_shared<ScanTask>(this->numberThreads(), this->numberSubThreads());
    }
};

template <typename T>
class ScanGraph : public hh::Graph<1, CopyIfData, CopyIfData> {
  public:
    ScanGraph(size_t numberThreads, size_t numberSubThreads) : hh::Graph<1, CopyIfData, CopyIfData>("ScanGraph") {
        auto scanTask = std::make_shared<ScanTask>(numberThreads, numberSubThreads);
        this->inputs(scanTask);
        this->outputs(scanTask);
    }
};

#endif
