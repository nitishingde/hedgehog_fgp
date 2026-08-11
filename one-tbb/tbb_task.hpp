#ifndef TBB_TASK
#define TBB_TASK
#include <hedgehog.h>
#include <oneapi/tbb.h>
#include <oneapi/tbb/task_arena.h>

namespace hh {

template <size_t Separator, typename ...Types>
class TBBTask;

namespace core {

template <size_t Separator, typename ...Types>
class TBBCoreTask : public CoreTask<Separator, Types...> {
    oneapi::tbb::task_arena arena;

  public:
    TBBCoreTask(TBBTask<Separator, Types...> *const task,
            std::string const &name, size_t numberThreads, size_t numberSubThreads, bool const automaticStart) :
        CoreTask<Separator, Types...>(task, name, numberThreads, automaticStart),
        arena(numberThreads) {}

    void run() override {
        std::chrono::time_point<std::chrono::system_clock>
            start,
            finish;

        volatile bool canTerminate = false;

        this->isActive(true);
        this->nvtxProfiler()->initialize(this->threadId(), this->graphId());
        this->preRun();

        if (this->automaticStart()) {  this->callAllExecuteWithNullptr(); }

        // Actual computation loop
        while (!this->canTerminate()) {
            // Wait for a data to arrive or termination
            this->nvtxProfiler()->startRangeWaiting();
            start = std::chrono::system_clock::now();
            canTerminate = this->sleep();
            finish = std::chrono::system_clock::now();
            this->nvtxProfiler()->endRangeWaiting();
            this->incrementWaitDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));

            // If loop can terminate break the loop early
            if (canTerminate) { break; }

            // Operate the connectedReceivers to get a data and send it to execute
            arena.execute([this]() {
                this->operateReceivers();
            });
        }

        // Do the shutdown phase
        this->postRun();
        // Wake up a node that this node is linked to
        this->wakeUp();
    }
};
}

template <size_t Separator, typename ...Types>
class TBBTask : public AbstractTask<Separator, Types...> {
  public:
    TBBTask(std::string const &name, size_t numberThreads, size_t numberSubThreads, bool automaticStart = false)
        : AbstractTask<Separator, Types...>(std::make_shared<core::TBBCoreTask<Separator, Types...>>(
                    this, name, numberThreads, numberSubThreads, automaticStart)) {}
};

}

#endif
