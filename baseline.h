#ifndef HEDGEHOG_FGP_BASELINE_H
#define HEDGEHOG_FGP_BASELINE_H

#include <hedgehog.h>
#include <latch>

class LatchCV {
private:
    std::atomic_int32_t     counter_ = {0};
    std::mutex              mutex_   = {};
    std::condition_variable cv_      = {};

public:
    explicit LatchCV(const int32_t count): counter_(count) {}

    void wait() {
        auto lock = std::unique_lock(mutex_);
        cv_.wait(lock, [this]{ return counter_.load() == 0; });
    }

    void count_down() {
        if(counter_.fetch_sub(1) == 1) {
            auto lg = std::lock_guard(mutex_);
            cv_.notify_one();
        }
    }
};

class LatchAtomics {
private:
    std::atomic_int32_t counter_ = {0};

public:
    explicit LatchAtomics(const int32_t count): counter_(count) {}

    void wait() const {
        for(auto current = counter_.load(std::memory_order_acquire); 0 < current; current = counter_.load(std::memory_order_acquire)) {
            counter_.wait(current, std::memory_order_relaxed);
        }
    }

    void count_down() {
        if(counter_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            counter_.notify_one();
        }
    }
};

using Latch = std::latch;
// using Latch = LatchCV;
// using Latch = LatchAtomics;

template<typename Data, typename Range, typename Latch = Latch>
struct WorkUnit {
    std::shared_ptr<Data>  data;
    Range                  range;
    std::shared_ptr<Latch> latch;
};

struct SaxpyData {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    float              a;
};

using SaxpyWorkUnit = WorkUnit<SaxpyData, std::tuple<int32_t, int32_t>>;

class SaxpyTask: public hh::AbstractTask<2, SaxpyData, SaxpyWorkUnit, SaxpyData> {
public:
    explicit SaxpyTask(const int32_t computeThreads):
        AbstractTask("SaxpyTask", computeThreads, false) {}

    void execute(std::shared_ptr<SaxpyData> data) override {
        auto       &[x, y, z, a]  = *data;
        const auto N              = static_cast<int32_t>(z.size());
        const auto computeThreads = static_cast<int32_t>(this->numberThreads());
        const auto range          = std::max(1000'000, (N+computeThreads-1)/computeThreads);
        auto       latch          = std::make_shared<Latch>((N+range-1)/range);
        const auto self           = static_cast<hh::core::abstraction::ReceiverAbstraction<SaxpyWorkUnit>*>(this->coreTask().get());
        for(int32_t i = range; i < N; i += range) {
            self->receive(std::make_shared<SaxpyWorkUnit>(data, std::make_tuple(i, std::min(i+range, N)), latch));
            this->coreTask()->wakeUp();
        }
        execute(std::make_shared<SaxpyWorkUnit>(data, std::make_tuple(0, range), latch));
        latch->wait();
        this->addResult(data);
    }

    void execute(std::shared_ptr<SaxpyWorkUnit> workUnit) override {
        auto [data, range, latch] = *workUnit;
        auto &[x, y, z, a] = *data;
        auto [start, end] = range;
        for(int32_t i = start; i < end; ++i) {
            z[i] = a*x[i] + y[i];
        }
        latch->count_down();
    }

    std::shared_ptr<AbstractTask> copy() override {
        return std::make_shared<SaxpyTask>(this->numberThreads());
    }
};

using ReductionResult   = float;
using ReductionData     = std::vector<ReductionResult>;
template<typename Container, typename Range, typename Result, typename Latch = Latch>
struct ReductionWorkUnit {
    std::shared_ptr<Container> data;
    Range                      range;
    Result&                    result;
    std::shared_ptr<Latch>     latch;
};
using MaxReductionWorkUnit = ReductionWorkUnit<ReductionData, std::tuple<int32_t, int32_t>, ReductionResult>;

class ReductionTask: public hh::AbstractTask<2, ReductionData, MaxReductionWorkUnit, ReductionData, ReductionResult> {
public:
    explicit ReductionTask(const int32_t computeThreads):
        AbstractTask("ReductionTask", computeThreads, false) {}

    void execute(std::shared_ptr<ReductionData> data) override {
        const auto N              = static_cast<int32_t>(data->size());
        const auto computeThreads = static_cast<int32_t>(this->numberThreads());
        const auto range          = std::max(1000'000, (N+computeThreads-1)/computeThreads);
        const auto chunks         = (N+range-1)/range;
        auto       latch          = std::make_shared<Latch>(chunks);
        auto       results        = std::vector(chunks, std::numeric_limits<ReductionResult>::min());
        const auto self           = static_cast<hh::core::abstraction::ReceiverAbstraction<MaxReductionWorkUnit>*>(this->coreTask().get());
        for(int32_t i = range, c = 0; i < N; i += range, ++c) {
            self->receive(std::make_shared<MaxReductionWorkUnit>(
                data,
                std::make_tuple(i, std::min(i+range, N)),
                results[c],
                latch
            ));
            this->coreTask()->wakeUp();
        }

        execute(std::make_shared<MaxReductionWorkUnit>(data, std::make_tuple(0, std::min(range, N)), results[0], latch));
        latch->wait();
        this->addResult(std::make_shared<ReductionResult>(*std::ranges::max_element(results)));
    }

    void execute(const std::shared_ptr<MaxReductionWorkUnit> workUnit) override {
        const auto [start, end] = workUnit->range;
        const auto &data        = *workUnit->data;
        auto       &value       = workUnit->result;
        for(int32_t i = start; i < end; ++i) {
            value = std::max(value, data[i]);
        }
        workUnit->latch->count_down();
    }

    std::shared_ptr<AbstractTask> copy() override {
        return std::make_shared<ReductionTask>(this->numberThreads());
    }
};

#endif //HEDGEHOG_FGP_BASELINE_H
