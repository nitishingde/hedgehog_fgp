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

// using Latch = std::latch;
// using Latch = LatchCV;
using Latch = LatchAtomics;

struct LatchDeleter {
    void operator()(const Latch *pLatch) const {
        pLatch->wait();
        delete pLatch;
    }
};

struct SaxpyData {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    float              a;
};

namespace hh {
    template<std::size_t Rank, std::integral Int = int>
    struct RangePolicy {
        using index_type = Int;
        static constexpr std::size_t rank = Rank;

        std::array<Int, Rank> begin = {};
        std::array<Int, Rank> end   = {};
        std::array<Int, Rank> step  = {}; // or 'tile'
    };

    template<std::integral Int>
    struct RangePolicy<1, Int> {
        using index_type = Int;
        static constexpr std::size_t rank = 1;

        Int begin = 0;
        Int end   = 0;
        Int step  = 1;
    };

    template<std::integral Int = int>
    using RangePolicy1D = RangePolicy<1, Int>;

    template<std::integral Int = int32_t>
    using RangePolicy2D = RangePolicy<2, Int>;

    template<std::integral Int = int32_t>
    using RangePolicy3D = RangePolicy<3, Int>;

    template<std::size_t Rank, std::integral Int = int>
    using MDRangePolicy = RangePolicy<Rank, Int>;

    namespace tool {
        template<typename T>
        concept IsRangePolicy1D = requires(T r) {
            typename T::index_type;
            requires std::integral<typename T::index_type>;
            requires T::rank == 1;
            { r.begin } -> std::same_as<typename T::index_type&>;
            { r.end   } -> std::same_as<typename T::index_type&>;
        };

        template <typename T>
        concept IsMDRangePolicy = requires(T r) {
            typename T::index_type;
            requires std::integral<typename T::index_type>;
            requires (T::rank > 1);
            { r.begin } -> std::same_as<std::array<typename T::index_type, T::rank>&>;
            { r.end   } -> std::same_as<std::array<typename T::index_type, T::rank>&>;
        };

        template<typename T>
        concept IsRangePolicy = IsRangePolicy1D<T> || IsMDRangePolicy<T>;

        template<typename T>
        concept IsRangePolicy2D = IsMDRangePolicy<T> and T::rank == 2;

        template<typename T>
        concept IsRangePolicy3D = IsMDRangePolicy<T> and T::rank == 3;
    }

    template<typename Data, tool::IsRangePolicy Range, typename Latch = Latch>
    struct WorkUnit {
        using InputType   = Data;
        using DataType    = Data;
        using RangePolicy = Range;

        std::shared_ptr<Data> data;
        Range                 range;
        Latch&                latch;

        ~WorkUnit() {
            latch.count_down();
        }

        [[nodiscard]] auto operator*() {
            return std::make_tuple(data, range);
        }
    };

    namespace tool {
        template <typename T>
        concept IsWorkUnit = requires(T w) {
            w.data;
            w.range;
            w.latch;

            requires IsRangePolicy1D<std::remove_cvref_t<decltype(w.range)>> or IsMDRangePolicy<std::remove_cvref_t<decltype(w.range)>>;

            { w.latch.count_down() };
            { *w                   } -> std::convertible_to<std::tuple<decltype(w.data), decltype(w.range)>>;
        };

        template <typename T>
        concept IsParallelForInput = requires(T p) {
            typename T::InputType;
            typename T::RangePolicy;
            typename T::WorkUnit;

            requires IsRangePolicy1D<typename T::RangePolicy> or IsMDRangePolicy<typename T::RangePolicy>;
            requires std::is_same_v<typename T::WorkUnit, WorkUnit<typename T::InputType, typename T::RangePolicy>>;
        };
    }

    template<typename Input, tool::IsRangePolicy Range>
    requires (not tool::IsWorkUnit<Input>)
    struct ParallelForInput {
        using InputType   = Input;
        using RangePolicy = Range;
        using WorkUnit    = WorkUnit<Input, Range>;
    };

    namespace tool {
        template<typename T>
        struct ExpandInput {
            using Type = std::tuple<T>;
        };

        template<IsParallelForInput T>
        struct ExpandInput<T> {
            using Type = std::tuple<typename T::InputType, typename T::WorkUnit>;
        };

        template<typename InputsTuple>
        struct ExpandAllInputs;

        template<typename ...Inputs>
        struct ExpandAllInputs<std::tuple<Inputs...>> {
            using Type = decltype(std::tuple_cat(
                std::declval<typename ExpandInput<Inputs>::Type>()...
            ));
        };

        template<size_t Separator, typename ExpandedInputsTuple, typename OutputsTuple>
        struct InstantiateTaskBase;

        template<size_t Separator, typename ...ExpandedInputs, typename ...Outputs>
        struct InstantiateTaskBase<Separator, std::tuple<ExpandedInputs...>, std::tuple<Outputs...>> {
            using Type = AbstractTask<Separator, ExpandedInputs..., Outputs...>;
        };

        template<size_t Separator, class ...AllTypes>
        using InstantiateTaskBase_t = InstantiateTaskBase<
            std::tuple_size_v<typename ExpandAllInputs<Inputs<Separator, AllTypes...>>::Type>,
            typename ExpandAllInputs<Inputs<Separator, AllTypes...>>::Type,
            Outputs<Separator, AllTypes...>
        >::Type;

        template<typename Target, typename Tuple, size_t CurrentIndex = 0>
        struct IndexOfType;

        template<typename Target, size_t CurrentIndex>
        struct IndexOfType<Target, std::tuple<>, CurrentIndex> {
            static_assert(CurrentIndex < 0, "Type not found in Tuple!");
        };

        template<typename Target, typename ...Rest, size_t CurrentIndex>
        struct IndexOfType<Target, std::tuple<Target, Rest...>, CurrentIndex> {
            static constexpr size_t Value = CurrentIndex;
        };

        template<typename Target, typename Head, typename ...Rest, size_t CurrentIndex>
        struct IndexOfType<Target, std::tuple<Head, Rest...>, CurrentIndex>
            : IndexOfType<Target, std::tuple<Rest...>, CurrentIndex + 1> {};

        template<typename Target, typename Tuple>
        constexpr size_t IndexOfType_v = IndexOfType<Target, Tuple>::Value;
    }

    template<size_t Separator, class ...AllTypes>
    class AbstractParallelForTask: public tool::InstantiateTaskBase_t<Separator, AllTypes...> {
    public:
        using ExpandedInputs = tool::ExpandAllInputs<tool::Inputs<Separator, AllTypes...>>::Type;
        using Outputs        = tool::Outputs<Separator, AllTypes...>;

        static constexpr auto TotalExpandedInputs = std::tuple_size_v<ExpandedInputs>;
        using Base           = tool::InstantiateTaskBase_t<Separator, AllTypes...>;

        explicit AbstractParallelForTask(const std::string &name = "ParallelForTask", const size_t numberThreads = 1):
            Base(name, numberThreads, false) {}

        template<tool::ContainsInTupleConcept<ExpandedInputs> InputType, std::integral Int>
        [[nodiscard]] auto executeWorkUnitsAsync(const std::shared_ptr<InputType> &data, const Int start, const Int end, const Int MIN_RANGE = 100'000) {
            constexpr size_t INP_POS = hh::tool::IndexOfType_v<InputType, ExpandedInputs>;
            static_assert(INP_POS + 1 < TotalExpandedInputs, "Selected input does not appear to be a ParallelForInput (no associated WorkUnit follows).");

            using WorkUnit    = std::tuple_element_t<INP_POS + 1, ExpandedInputs>;
            static_assert(tool::IsWorkUnit<WorkUnit>, "InputType does not have a valid WorkUnit (fix: use hh::ParallelForInput<InputType, RangePolicy>).");
            using RangePolicy = WorkUnit::RangePolicy;
            static_assert(tool::IsRangePolicy1D<RangePolicy>, "RangePolicy for this WorkUnit is not 1D!");

            const auto N              = end-start;
            const auto computeThreads = static_cast<Int>(this->numberThreads());
            const auto range          = std::max(MIN_RANGE, (N+computeThreads-1)/computeThreads);
            auto       latch          = std::unique_ptr<Latch, LatchDeleter>(new Latch((N+range-1)/range));
            auto       self           = static_cast<core::abstraction::ReceiverAbstraction<WorkUnit>*>(this->coreTask().get());
            for(Int i = range; i < N; i += range) {
                self->receive(std::make_shared<WorkUnit>(data, RangePolicy(i, std::min(i+range, N)), *latch));
                this->coreTask()->wakeUp();
            }
            static_cast<behavior::Execute<WorkUnit>*>(this)->execute(std::make_shared<WorkUnit>(data, RangePolicy(0, range), *latch));
            return latch;
        }

        template<tool::ContainsInTupleConcept<ExpandedInputs> InputType, std::integral Int>
        [[nodiscard]] auto executeWorkUnits(const std::shared_ptr<InputType> &data, const Int start, const Int end, const Int MIN_RANGE = 100'000) {
            (void)executeWorkUnitsAsync(data, start, end, MIN_RANGE);
        }
    };
}

struct MultiplierData {
    std::vector<float> data;
    float              factor;
};

class SaxpyTask final: public hh::AbstractParallelForTask<4, int32_t, hh::ParallelForInput<SaxpyData, hh::RangePolicy1D<int32_t>>, double, hh::ParallelForInput<MultiplierData, hh::RangePolicy1D<int32_t>>, SaxpyData, MultiplierData> {
public:
    explicit SaxpyTask(const int32_t computeThreads):
        AbstractParallelForTask("SaxpyTask", computeThreads) {}

    void execute(const std::shared_ptr<int32_t> data) override {}

    void execute(const std::shared_ptr<double> data) override {}

    void execute(const std::shared_ptr<SaxpyData> data) override {
        this->executeWorkUnits(data, int32_t{0}, static_cast<int32_t>(data->z.size()));
        this->addResult(data);
    }

    void execute(const std::shared_ptr<hh::ParallelForInput<SaxpyData, hh::RangePolicy1D<int32_t>>::WorkUnit> workUnit) override {
        const auto [data, range] = **workUnit;
        auto       &[x, y, z, a] = *data;
        for(int32_t i = range.begin; i < range.end; i += range.step) {
            z[i] = a*x[i] + y[i];
        }
    }

    void execute(const std::shared_ptr<MultiplierData> data) override {
        this->executeWorkUnits(data, int32_t{0}, static_cast<int32_t>(data->data.size()));
        this->addResult(data);
    }

    void execute(const std::shared_ptr<hh::ParallelForInput<MultiplierData, hh::RangePolicy1D<int32_t>>::WorkUnit> workUnit) override {
        const auto [data, range] = **workUnit;
        auto       &[arr, fact]  = *data;
        for(int32_t i = range.begin; i < range.end; i += range.step) {
            arr[i] *= fact;
        }
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
