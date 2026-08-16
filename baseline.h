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

        template<typename T>
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

    struct ForTag {
        struct Empty {};
        using StorageType = Empty;
    };

    template<typename ValueType>
    struct ReduceTag {
        using StorageType = ValueType&;
    };

    template<typename ValueType>
    struct ScanTag {
        struct ScanState {
            using value_type = ValueType;
            ValueType& chunkSum     = {};
            ValueType& prefixOffset = {};
            bool       finalPass    = {false};
        };
        using StorageType = ScanState;
    };

    namespace tool {
        template<typename StateTag>
        struct StateStorage {
            using Type = StateTag;
        };

        template<>
        struct StateStorage<ForTag> {
            struct Empty {};
            using Type = Empty;
        };
    }

    template<typename Data, tool::IsRangePolicy Range = RangePolicy1D<int32_t>, typename StateTag = ForTag>
    struct WorkUnit {
        using RangePolicy  = Range;
        using StorageType  = StateTag::StorageType;

        std::shared_ptr<Data> data;
        Range                 range;
        Latch&                latch;
        [[no_unique_address]]
        StorageType           state;

        ~WorkUnit() {
            latch.count_down();
        }

        [[nodiscard]] auto operator*() {
            return std::make_tuple(data, range);
        }
    };

    template<typename Data, tool::IsRangePolicy Range, typename ...ConstructTags>
    struct ParallelInput {
        using InputType   = Data;
        using RangePolicy = Range;
        using TagsTuple   = std::conditional_t<sizeof...(ConstructTags) == 0, std::tuple<ForTag>, std::tuple<ConstructTags...>>;
    };

    template<typename Input, tool::IsRangePolicy Range = RangePolicy1D<int32_t>>
    using ParallelForInput = ParallelInput<Input, Range, ForTag>;

    template<typename Input, typename ValueType, tool::IsRangePolicy Range = RangePolicy1D<int32_t>>
    using ParallelReduceInput = ParallelInput<Input, Range, ReduceTag<ValueType>>;

    template<typename Input, typename ValueType, tool::IsRangePolicy Range = RangePolicy1D<int32_t>>
    using ParallelScanInput = ParallelInput<Input, Range, ScanTag<ValueType>>;

    namespace tool {
        template<typename T>
        concept IsWorkUnit = requires(T w) {
            w.data;
            w.range;
            w.latch;

            requires IsRangePolicy1D<std::remove_cvref_t<decltype(w.range)>> or IsMDRangePolicy<std::remove_cvref_t<decltype(w.range)>>;

            { w.latch.count_down() };
            { *w                   } -> std::convertible_to<std::tuple<decltype(w.data), decltype(w.range)>>;
        };

        template<typename T>
        concept IsParallelInput = requires {
            typename T::InputType;
            typename T::RangePolicy;
            typename T::TagsTuple;
        };
    }

    namespace tool {
        template<typename InputDescriptor>
        struct ExpandDescriptor;

        template<typename Data, typename Range, typename ...Tags>
        struct ExpandDescriptor<ParallelInput<Data, Range, Tags...>> {
            using Type = std::tuple<Data, WorkUnit<Data, Range, Tags>...>;
        };

        template<typename T>
        struct ExpandInput {
            using Type = std::tuple<T>;
        };

        template<IsParallelInput T>
        struct ExpandInput<T> {
            using Type = ExpandDescriptor<T>::Type;
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
    class AbstractParallelTask: public tool::InstantiateTaskBase_t<Separator, AllTypes...> {
    public:
        using RawInputs      = tool::Inputs<Separator, AllTypes...>;
        using ExpandedInputs = tool::ExpandAllInputs<tool::Inputs<Separator, AllTypes...>>::Type;
        using Outputs        = tool::Outputs<Separator, AllTypes...>;

        static constexpr auto TotalExpandedInputs = std::tuple_size_v<ExpandedInputs>;
        using Base           = tool::InstantiateTaskBase_t<Separator, AllTypes...>;

        explicit AbstractParallelTask(const std::string &name = "ParallelTask", const size_t numberThreads = 1):
            Base(name, numberThreads, false) {}

        template<tool::ContainsInTupleConcept<ExpandedInputs> InputType, std::integral Int = int32_t>
        requires (not tool::IsWorkUnit<InputType>)
        [[nodiscard]] auto executeWorkUnitsAsync(const std::shared_ptr<InputType> &data, const Int start, const Int end, const Int MIN_RANGE = 100'000) {
            using WorkUnit    = std::tuple_element_t<tool::IndexOfType_v<WorkUnit<InputType, RangePolicy1D<Int>, ForTag>, ExpandedInputs>, ExpandedInputs>;
            using RangePolicy = WorkUnit::RangePolicy;

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

        template<tool::ContainsInTupleConcept<ExpandedInputs> InputType, std::integral Int = int32_t>
        [[nodiscard]] auto executeWorkUnits(const std::shared_ptr<InputType> &data, const Int start, const Int end, const Int MIN_RANGE = 100'000) {
            (void)executeWorkUnitsAsync(data, start, end, MIN_RANGE);
        }

        template<typename ValueType, typename InputType, std::integral Int, typename BinaryOp = std::plus<>>
        requires (not tool::IsWorkUnit<InputType>)
        [[nodiscard]] ValueType executeReduce(const std::shared_ptr<InputType> &data, const Int start, const Int end, ValueType identityValue, BinaryOp reductionOp, const Int MIN_RANGE = 100'000) {
            using WorkUnit     = std::tuple_element_t<tool::IndexOfType_v<WorkUnit<InputType, RangePolicy1D<Int>, ReduceTag<ValueType>>, ExpandedInputs>, ExpandedInputs>;
            using RangePolicy  = WorkUnit::RangePolicy;

            const auto N              = end-start;
            const auto computeThreads = static_cast<Int>(this->numberThreads());
            const auto range          = std::max(MIN_RANGE, (N+computeThreads-1)/computeThreads);
            const auto chunks         = (N+range-1)/range;
            auto       partialResults = std::vector(chunks, identityValue);
            auto       latch          = std::unique_ptr<Latch, LatchDeleter>(new Latch(chunks));
            auto       self           = static_cast<core::abstraction::ReceiverAbstraction<WorkUnit>*>(this->coreTask().get());

            for(Int i = range, c = 1; i < N; i += range, ++c) {
                self->receive(std::make_shared<WorkUnit>(data, RangePolicy(i, std::min(i + range, N)), *latch, partialResults[c]));
                this->coreTask()->wakeUp();
            }

            static_cast<behavior::Execute<WorkUnit>*>(this)->execute(std::make_shared<WorkUnit>(data, RangePolicy(0, std::min(range, N)), *latch, partialResults[0]));
            latch->wait();

            auto finalResult = identityValue;
            for(const auto &partialResult: partialResults) {
                finalResult = reductionOp(finalResult, partialResult);
            }

            return finalResult;
        }
    };
}

using ReductionResult   = float;
using ReductionData     = std::vector<ReductionResult>;

class ParallelTask final: public hh::AbstractParallelTask<4, int32_t, hh::ParallelForInput<SaxpyData>, double, hh::ParallelReduceInput<ReductionData, ReductionResult>, SaxpyData, ReductionResult> {
public:
    explicit ParallelTask(const int32_t computeThreads):
        AbstractParallelTask("ParallelTask", computeThreads) {}

    void execute(const std::shared_ptr<int32_t> data) override {}

    void execute(const std::shared_ptr<double> data) override {}

    void execute(const std::shared_ptr<SaxpyData> data) override {
        this->executeWorkUnits(data, int32_t{0}, static_cast<int32_t>(data->z.size()));
        this->addResult(data);
    }

    void execute(const std::shared_ptr<hh::WorkUnit<SaxpyData>> workUnit) override {
        const auto [data, range] = **workUnit;
        auto       &[x, y, z, a] = *data;
        for(int32_t i = range.begin; i < range.end; i += range.step) {
            z[i] = a*x[i] + y[i];
        }
    }

    void execute(const std::shared_ptr<ReductionData> data) override {
        auto result = this->executeReduce(
            data,
            int32_t{0},
            static_cast<int32_t>(data->size()),
            std::numeric_limits<ReductionResult>::min(),
            [](const ReductionResult a, const ReductionResult b) { return std::max(a, b); }
        );
        this->addResult(std::make_shared<ReductionResult>(result));
    }

    void execute(const std::shared_ptr<hh::WorkUnit<ReductionData, hh::RangePolicy1D<int32_t>, hh::ReduceTag<ReductionResult>>> workUnit) override {
        auto [data, range] = **workUnit;
        auto value = std::numeric_limits<ReductionResult>::min();
        for(int32_t i = range.begin; i < range.end; i += range.step) {
            value = std::max(value, data->at(i));
        }
        workUnit->state = value;
    }

    std::shared_ptr<AbstractTask> copy() override {
        return std::make_shared<ParallelTask>(this->numberThreads());
    }
};

#endif //HEDGEHOG_FGP_BASELINE_H
