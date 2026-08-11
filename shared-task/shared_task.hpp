#ifndef SHARED_TASK
#define SHARED_TASK
#include <hedgehog.h>
#include <thread>
#include <memory>
#include <atomic>
#include <semaphore>
#include <functional>

class ThreadPool {
    struct Worker {
        std::thread thread;
        std::atomic<bool> canTerminate{false};
        std::counting_semaphore<256> sem{0};
        size_t index;
        ThreadPool *pool = nullptr;

        void run() {
            for (;;) {
                this->sem.acquire();
                this->pool->parkedWorkers_ -= 1;
                if (this->canTerminate.load()) {
                    break;
                }
                if (!this->pool->fun_) {
                    continue;
                }
                this->pool->fun_(this->index, this->pool->workers_.size() + 1);
                if (this->pool->parkedWorkers_.fetch_add(1) == (this->pool->workers_.size() - 1)) {
                    this->pool->end_process_sem_.release(1);
                }
            }
        }
    };
    std::vector<std::unique_ptr<Worker>> workers_ = {};
    std::function<void(size_t, size_t)> fun_;
    std::atomic<size_t> parkedWorkers_{0};
    std::counting_semaphore<1> end_process_sem_{0};

  public:
    ThreadPool(size_t numberThreads) {
        initialize(numberThreads);
    }

    ~ThreadPool() {
        finalize();
    }

    void initialize(size_t numberThreads) {
        finalize();
        workers_.resize(numberThreads - 1);
        for (size_t i = 0; i < (numberThreads - 1); ++i) {
            workers_[i] = std::make_unique<Worker>();
            workers_[i]->pool = this;
            workers_[i]->index = i + 1;
            workers_[i]->thread = std::thread([this, i]() {
                workers_[i]->run();
            });
        }
        parkedWorkers_.store(numberThreads - 1);
    }

    void finalize() {
        if (workers_.size() == 0) {
            return;
        }
        for (auto &worker : workers_) {
            worker->canTerminate.store(true);
            worker->sem.release();
        }
        for (auto &worker : workers_) {
            worker->thread.join();
        }
        workers_.clear();
    }

    void run(std::function<void(size_t, size_t)> fun) {
        fun_ = fun;
        for (auto &worker : workers_) {
            worker->sem.release();
        }
        fun(0, workers_.size() + 1);
        end_process_sem_.acquire(); // wait for all the workers to finish
    }
};

template <size_t Separator, typename ...Types>
class SharedTask : public hh::AbstractTask<Separator, Types...> {
    ThreadPool pool_;

  public:
    SharedTask(std::string name, size_t numberThreads, size_t numberSubThreads)
        : hh::AbstractTask<Separator, Types...>(name, numberThreads),
          pool_(numberSubThreads) {}

    void process(std::function<void(size_t, size_t)> fun) {
        pool_.run(fun);
    }
};

#endif
