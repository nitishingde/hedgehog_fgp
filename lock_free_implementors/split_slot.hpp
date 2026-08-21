#ifndef LOCK_FREE_IMPLEMENTORS_SPLIT_SLOT_H
#define LOCK_FREE_IMPLEMENTORS_SPLIT_SLOT_H

#include <set>
#include <mutex>
#include <atomic>
#include <vector>
#include <semaphore>
#include <hedgehog.h>

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Default concrete implementation of slot interface
/// @details Utilise mutexes to protect its list of notifiers, and for the condition_variable / wait mechanism
class SplitSlot : public ImplementorSlot {
 private:
  std::unique_ptr<std::set<abstraction::NotifierAbstraction *>> const notifiers_ =
      std::make_unique<std::set<abstraction::NotifierAbstraction * >>(); ///< List of notifiers linked to this slot

  std::mutex mutexNotifierAccess_{}; ///< Mutex to protect the list of notifiers

  // TODO: not sure about the size of the semaphore
  struct Sem { // need a wrapper because C++ considers that ZII a semaphore is incorrect...
    alignas(CACHE_LINE_SIZE) std::counting_semaphore<1024> s{0};
  };
  std::vector<Sem> semaphores_;
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> currentThread_{0};
  alignas(CACHE_LINE_SIZE) std::atomic<bool> terminate_{false}; ///< used to force the termination

 public:
  SplitSlot(size_t numberThreads) : semaphores_(numberThreads) {}

  /// @brief Test if there is any notifiers connected
  /// @return True if there is any, else false
  [[nodiscard]] bool hasNotifierConnected() override {
    std::lock_guard<std::mutex> lck(mutexNotifierAccess_);
    return !notifiers_->empty();
  }

  /// @brief Accessor to the number of notifiers connected
  /// @return Number of notifiers connected
  size_t nbNotifierConnected() override {
    std::lock_guard<std::mutex> lck(mutexNotifierAccess_);
    return notifiers_->size();
  }

  /// @brief Accessor to the connected notifiers
  /// @return A set of connected notifiers
  [[nodiscard]] std::set<abstraction::NotifierAbstraction *> const &connectedNotifiers() const override {
    return *notifiers_;
  }

  /// @brief Add a notifier to the list of connected notifiers
  /// @param notifier Notifier to add to the list of connected notifiers
  void addNotifier(abstraction::NotifierAbstraction *const notifier) override {
    std::lock_guard<std::mutex> lck(mutexNotifierAccess_);
    notifiers_->insert(notifier);
  }

  /// @brief Remove a notifier to the list of connected notifiers
  /// @param notifier Notifier to remove from the list of connected notifiers
  void removeNotifier(abstraction::NotifierAbstraction *const notifier) override {
    std::lock_guard<std::mutex> lck(mutexNotifierAccess_);
    notifiers_->erase(notifier);
  }

  /// @brief Sleep mechanism used to make the thread enter in a sleep state
  /// @param slot Slot abstraction (core attache to the thread), used for callbacks
  /// @return True if the node can terminate, else false
  bool sleep(abstraction::SlotAbstraction *slot, SlotSleepOptions const &opts) override {
    assert(opts.threadId);
    semaphores_[opts.threadId.value()].s.acquire();
    return slot->canTerminate() || terminate_.load(std::memory_order_relaxed);
  }

  /// @brief Function used to wake up a thread attached to this condition variable
  void wakeUp(SlotWakeUpOptions const &opts) override {
    if (opts.count) {
      size_t startIdx = currentThread_.fetch_add(opts.count.value(), std::memory_order_release);
      size_t endIdx = startIdx + opts.count.value();
      size_t numberThreads = semaphores_.size();
      for (size_t i = startIdx; i < endIdx; ++i) {
        semaphores_[i % numberThreads].s.release();
      }
    } else {
      for (auto &sem : semaphores_) {
        sem.s.release();
      }
    }
  }

  /// @brief Function used to wake up and terminate a thread attached to this condition variable
  /// @param value Value of the termination flag.
  void terminate(bool value) override {
    terminate_.store(value, std::memory_order_release);
    if (value) {
      wakeUp({});
    }
  }

  /// @brief Return of the slot has been terminated.
  /// @return True if the slot has been terminated.
  bool terminate() const override {
    return terminate_.load(std::memory_order_relaxed);
  }
};
}
}
}

#endif
