#ifndef HEDGEHOG_FGP_BASELINE_H
#define HEDGEHOG_FGP_BASELINE_H

#include "lock_free_implementors/limited_lock_free_queue_receiver.hpp"
#include "lock_free_implementors/moodycamel_receiver.hpp"
#include "lock_free_implementors/split_slot.hpp"
#include "lock_free_implementors/group_slot.hpp"
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

namespace hh {
    template<std::size_t Rank, std::integral Int = int64_t>
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

    template<std::integral Int = int64_t>
    using RangePolicy1D = RangePolicy<1, Int>;

    template<std::integral Int = int64_t>
    using RangePolicy2D = RangePolicy<2, Int>;

    template<std::integral Int = int64_t>
    using RangePolicy3D = RangePolicy<3, Int>;

    template<std::size_t Rank, std::integral Int = int64_t>
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

    struct ScanTag {
        struct ScanState {
            using value_type = int64_t;

            int64_t& accumulator;
            bool     finalPass;
        };
        using StorageType = ScanState;
    };

    template<typename Data, typename StateTag = ForTag, tool::IsRangePolicy Range = RangePolicy1D<int64_t>>
    struct WorkUnit {
        using RangePolicy  = Range;
        using StorageType  = StateTag::StorageType;

        std::shared_ptr<Data> data;
        Range                 range;
        Latch&                latch;
        [[no_unique_address]]
        StorageType           state = {};

        ~WorkUnit() {
            latch.count_down();
        }

        WorkUnit& operator=(StorageType val) {
            state = val;
            return *this;
        }

        WorkUnit& operator=(const int64_t val) requires (std::is_same_v<StateTag, ScanTag>) {
            state.accumulator = val;
            return *this;
        }

        [[nodiscard]] auto operator*() {
            return std::make_tuple(data, range);
        }

        [[nodiscard]] int64_t accumulator() requires (std::is_same_v<StateTag, ScanTag>) {
            return state.accumulator;
        }

        void accumulator(const int64_t val) requires (std::is_same_v<StateTag, ScanTag>) {
            state.accumulator = val;
        }

        [[nodiscard]] bool isFinalPass() requires (std::is_same_v<StateTag, ScanTag>) {
            return state.finalPass;
        }
    };

    template<typename Data, tool::IsRangePolicy Range, typename ...ConstructTags>
    struct ParallelInput {
        using InputType   = Data;
        using RangePolicy = Range;
        using TagsTuple   = std::conditional_t<sizeof...(ConstructTags) == 0, std::tuple<ForTag>, std::tuple<ConstructTags...>>;
    };

    template<typename Input, tool::IsRangePolicy Range = RangePolicy1D<int64_t>>
    using ParallelForInput = ParallelInput<Input, Range, ForTag>;

    template<typename Input, typename ValueType, tool::IsRangePolicy Range = RangePolicy1D<int64_t>>
    using ParallelReduceInput = ParallelInput<Input, Range, ReduceTag<ValueType>>;

    template<typename Input, tool::IsRangePolicy Range = RangePolicy1D<int64_t>>
    using ParallelScanInput = ParallelInput<Input, Range, ScanTag>;

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

        template<typename InputDescriptor>
        struct ExpandDescriptor;

        template<typename Data, typename Range, typename ...Tags>
        struct ExpandDescriptor<ParallelInput<Data, Range, Tags...>> {
            using Type = std::tuple<Data, WorkUnit<Data, Tags, Range>...>;
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

        template<template <size_t, typename ...> class HHType, size_t Separator, typename ExpandedInputsTuple, typename OutputsTuple>
        struct InstantiateHHType;

        template<template <size_t, typename ...> class HHType, size_t Separator, typename ...ExpandedInputs, typename ...Outputs>
        struct InstantiateHHType<HHType, Separator, std::tuple<ExpandedInputs...>, std::tuple<Outputs...>> {
            using Type = HHType<Separator, ExpandedInputs..., Outputs...>;
        };

        template<template <size_t, typename ...> class HHType, size_t Separator, class ...AllTypes>
        using InstantiateHHType_t = typename InstantiateHHType<
            HHType,
            std::tuple_size_v<typename ExpandAllInputs<Inputs<Separator, AllTypes...>>::Type>,
            typename ExpandAllInputs<Inputs<Separator, AllTypes...>>::Type,
            Outputs<Separator, AllTypes...>
        >::Type;

        template <size_t Separator, class ...AllTypes>
        auto makeParallelTaskCore(auto task, std::string const &name, size_t numberThreads, bool automaticStart = false) {
            using CoreType = InstantiateHHType_t<hh::core::CoreTask, Separator, AllTypes...>;
            // using ReceiverType = InstantiateHHType_t<hh::core::implementor::MLLFQR, Separator, AllTypes...>;
            using ReceiverType = InstantiateHHType_t<hh::core::implementor::MCLR, Separator, AllTypes...>;
            using DMEType = InstantiateHHType_t<DME, Separator, AllTypes...>;
            using MDSType = InstantiateHHType_t<MDS, Separator, AllTypes...>;
            return std::make_shared<CoreType>(
                    task, name, numberThreads, false,
                    // std::make_shared<hh::core::implementor::DefaultSlot>(numberThreads),
                    std::make_shared<hh::core::implementor::SplitSlot>(numberThreads),
                    // std::make_shared<hh::core::implementor::GroupSlot>(numberThreads, 4),
                    std::make_shared<ReceiverType>(),
                    std::make_shared<DMEType>(task),
                    std::make_shared<hh::core::implementor::DefaultNotifier>(),
                    std::make_shared<MDSType>());
        }

        template<typename T>
        struct WorkUnitTraits {
            static constexpr bool is_work_unit = false;
            using Data = void;
            using Tag  = void;
        };

        template<typename Input, typename StateTag, typename Range>
        struct WorkUnitTraits<WorkUnit<Input, StateTag, Range>> {
            static constexpr bool is_work_unit = true;
            using Data = Input;
            using Tag  = StateTag;
        };

        template<typename TargetData, typename TargetTag, typename Tuple>
        struct FindWorkUnit;

        template<typename TargetData, typename TargetTag>
        struct FindWorkUnit<TargetData, TargetTag, std::tuple<>> {
            static_assert(sizeof(TargetData) == 0, "Requested parallel construct/tag is not registered for this InputType in ParallelInput!");
            using Type = TargetData;
        };

        template<typename TargetData, typename TargetTag, typename Head, typename... Tail>
        struct FindWorkUnit<TargetData, TargetTag, std::tuple<Head, Tail...>> {
        private:
            using Traits = WorkUnitTraits<Head>;

            static constexpr bool Matches = Traits::is_work_unit
                and std::is_same_v<typename Traits::Data, TargetData>
                and std::is_same_v<typename Traits::Tag, TargetTag>;

            template<bool Found, typename H, typename... T>
            struct Selector {
                using Type = H;
            };

            template<typename H, typename... T>
            struct Selector<false, H, T...> {
                using Type = FindWorkUnit<TargetData, TargetTag, std::tuple<T...>>::Type;
            };

        public:
            using Type = Selector<Matches, Head, Tail...>::Type;
        };
    }

    template<size_t Separator, class ...AllTypes>
    class AbstractParallelTask: public tool::InstantiateTaskBase_t<Separator, AllTypes...> {
    public:
        using RawInputs      = tool::Inputs<Separator, AllTypes...>;
        using ExpandedInputs = tool::ExpandAllInputs<tool::Inputs<Separator, AllTypes...>>::Type;
        using Outputs        = tool::Outputs<Separator, AllTypes...>;

        static constexpr auto TotalExpandedInputs = std::tuple_size_v<ExpandedInputs>;
        using Base           = tool::InstantiateTaskBase_t<Separator, AllTypes...>;

        // explicit AbstractParallelTask(const std::string &name = "ParallelTask", const size_t numberThreads = 1):
        //     Base(name, numberThreads, false) {}

        explicit AbstractParallelTask(const std::string &name = "ParallelTask", const size_t numberThreads = 1):
            Base(hh::tool::makeParallelTaskCore<Separator, AllTypes...>(this, name, numberThreads)) {}


        template<tool::ContainsInTupleConcept<ExpandedInputs> InputType, std::integral Int = int64_t>
        requires (not tool::IsWorkUnit<InputType>)
        [[nodiscard]] auto executeWorkUnitsAsync(const std::shared_ptr<InputType> &data, const Int start, const Int end, const Int MIN_RANGE = 100'000) {
            using WorkUnit    = tool::FindWorkUnit<InputType, ForTag, ExpandedInputs>::Type;
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
            static_cast<behavior::Execute<WorkUnit>*>(this)->execute(std::make_shared<WorkUnit>(data, RangePolicy(0, std::min(range, N)), *latch));
            return latch;
        }

        template<tool::ContainsInTupleConcept<ExpandedInputs> InputType, std::integral Int = int64_t>
        [[nodiscard]] auto executeWorkUnits(const std::shared_ptr<InputType> &data, const Int start, const Int end, const Int MIN_RANGE = 100'000) {
            (void)executeWorkUnitsAsync(data, start, end, MIN_RANGE);
        }

        template<tool::ContainsInTupleConcept<ExpandedInputs> InputType, std::integral Int, typename ValueType, typename BinaryOp>
        requires (not tool::IsWorkUnit<InputType>)
        [[nodiscard]] ValueType executeReductionWorkUnits(const std::shared_ptr<InputType> &data, const Int start, const Int end, ValueType identityValue, BinaryOp reductionOp, const Int MIN_RANGE = 100'000) {
            using WorkUnit    = tool::FindWorkUnit<InputType, ReduceTag<ValueType>, ExpandedInputs>::Type;
            using RangePolicy = WorkUnit::RangePolicy;

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

        template<tool::ContainsInTupleConcept<ExpandedInputs> InputType, std::integral Int, typename BinaryOp = std::plus<int64_t>>
        requires (not tool::IsWorkUnit<InputType>)
        void executeScanWorkUnits(const std::shared_ptr<InputType> &data, const Int start, const Int end, int64_t identityValue = 0, BinaryOp scanOp = {}, const Int MIN_RANGE = 100'000) {
            using WorkUnit    = tool::FindWorkUnit<InputType, ScanTag, ExpandedInputs>::Type;
            using RangePolicy = WorkUnit::RangePolicy;

            const auto N              = end-start;
            const auto computeThreads = static_cast<Int>(this->numberThreads());
            const auto range          = std::max(MIN_RANGE, (N+computeThreads-1)/computeThreads);
            const auto chunks         = (N+range-1)/range;
            auto       prefixSums     = std::vector(chunks, identityValue);
            auto       self           = static_cast<core::abstraction::ReceiverAbstraction<WorkUnit>*>(this->coreTask().get());
            if constexpr(true) {
                auto pass1Latch = std::unique_ptr<Latch, LatchDeleter>(new Latch(chunks));

                for(Int i = range, c = 1; i < N; i += range, ++c) {
                    self->receive(std::make_shared<WorkUnit>(
                        data,
                        RangePolicy(i, std::min(i + range, N)),
                        *pass1Latch,
                        ScanTag::ScanState {
                            .accumulator = prefixSums[c],
                            .finalPass   = false
                        }
                    ));
                    this->coreTask()->wakeUp();
                }

                static_cast<behavior::Execute<WorkUnit>*>(this)->execute(std::make_shared<WorkUnit>(
                    data,
                    RangePolicy(0, std::min(range, N)),
                    *pass1Latch,
                    ScanTag::ScanState {
                        .accumulator = prefixSums[0],
                        .finalPass   = false
                    }
                ));

                pass1Latch->wait();
            }

            auto prefixSum = identityValue;
            for(auto c = 0; c < chunks; ++c) {
                auto chunkPrefixSum = prefixSums[c];
                prefixSums[c]       = prefixSum;
                prefixSum           = scanOp(prefixSum, chunkPrefixSum);
            }

            if constexpr(true) {
                auto pass2Latch = std::unique_ptr<Latch, LatchDeleter>(new Latch(chunks));

                for(Int i = range, c = 1; i < N; i += range, ++c) {
                    self->receive(std::make_shared<WorkUnit>(
                        data,
                        RangePolicy(i, std::min(i + range, N)),
                        *pass2Latch,
                        ScanTag::ScanState {
                            .accumulator = prefixSums[c],
                            .finalPass   = true
                        }
                    ));
                    this->coreTask()->wakeUp();
                }

                static_cast<behavior::Execute<WorkUnit>*>(this)->execute(std::make_shared<WorkUnit>(
                    data,
                    RangePolicy(0, std::min(range, N)),
                    *pass2Latch,
                    ScanTag::ScanState {
                        .accumulator = prefixSums[0],
                        .finalPass   = true
                    }
                ));

                pass2Latch->wait();
            }
        }
    };
}

struct SaxpyData {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    float              a;
};

using ReductionResult   = float;
using ReductionData     = std::vector<ReductionResult>;

struct CopyIfData {
    std::vector<int32_t> src;
    std::vector<int32_t> dst;
};

class ParallelTask final: public hh::AbstractParallelTask<5, int32_t, hh::ParallelInput<SaxpyData, hh::RangePolicy1D<>, hh::ForTag>, double, hh::ParallelReduceInput<ReductionData, ReductionResult>, hh::ParallelScanInput<CopyIfData>, SaxpyData, ReductionResult, CopyIfData> {
public:
    explicit ParallelTask(const int32_t computeThreads):
        AbstractParallelTask("ParallelTask", computeThreads) {}

    void execute(const std::shared_ptr<int32_t> data) override {}

    void execute(const std::shared_ptr<double> data) override {}

    void execute(const std::shared_ptr<SaxpyData> data) override {
        this->executeWorkUnits(data, int64_t{0}, static_cast<int64_t>(data->z.size()));
        this->addResult(data);
    }

    void execute(const std::shared_ptr<hh::WorkUnit<SaxpyData>> workUnit) override {
        const auto [data, range] = **workUnit;
        auto       &[x, y, z, a] = *data;
        for(auto i = range.begin; i < range.end; i += range.step) {
            z[i] = a*x[i] + y[i];
        }
    }

    void execute(const std::shared_ptr<ReductionData> data) override {
        auto result = this->executeReductionWorkUnits(
            data,
            int64_t{0},
            static_cast<int64_t>(data->size()),
            std::numeric_limits<ReductionResult>::min(),
            [](const ReductionResult a, const ReductionResult b) { return std::max(a, b); }
        );
        this->addResult(std::make_shared<ReductionResult>(result));
    }

    void execute(const std::shared_ptr<hh::WorkUnit<ReductionData, hh::ReduceTag<ReductionResult>>> workUnit) override {
        auto [data, range] = **workUnit;
        auto value = std::numeric_limits<ReductionResult>::min();
        for(auto i = range.begin; i < range.end; i += range.step) {
            value = std::max(value, data->at(i));
        }
        *workUnit = value;
    }

    void execute(const std::shared_ptr<CopyIfData> data) override {
        this->executeScanWorkUnits(data, int64_t{0}, static_cast<int64_t>(data->src.size()));
        this->addResult(data);
    }

    void execute(const std::shared_ptr<hh::WorkUnit<CopyIfData, hh::ScanTag>> workUnit) override {
        auto [data, range] = **workUnit;
        auto &[src, dst]   = *data;

        auto       count       = workUnit->accumulator();
        const auto isFinalPass = workUnit->isFinalPass();
        for(auto i = range.begin; i < range.end; i += range.step) {
            if(src[i]%2 != 0) continue;

            if(isFinalPass) {
                dst[count] = src[i];
            }
            count++;
        }
        *workUnit = count;
    }

    std::shared_ptr<AbstractTask> copy() override {
        return std::make_shared<ParallelTask>(this->numberThreads());
    }
};

#endif //HEDGEHOG_FGP_BASELINE_H
