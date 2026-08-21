#ifndef LOCK_FREE_IMPLEMENTORS_GROUP_SLOT_H
#define LOCK_FREE_IMPLEMENTORS_GROUP_SLOT_H

#include <set>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include <hedgehog.h>

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Default concrete implementation of slot interface
/// @details Utilise mutexes to protect its list of notifiers, and for the condition_variable / wait mechanism
class GroupSlot : public ImplementorSlot {
 private:
  std::unique_ptr<std::set<abstraction::NotifierAbstraction *>> const notifiers_ =
      std::make_unique<std::set<abstraction::NotifierAbstraction * >>(); ///< List of notifiers linked to this slot

  std::mutex mutexNotifierAccess_{}; ///< Mutex to protect the list of notifiers

  alignas(CACHE_LINE_SIZE)std::atomic<bool> terminate_ = false; ///< used to force the termination

  struct SemGroupData { // semaphore
      std::counting_semaphore<1024> sem{0};
      void sleep(auto) { sem.acquire(); }
      void wakeUp(size_t count) { sem.release(count); }
  };

  struct CondGroupData { // condition variable
      std::mutex mutex{};
      std::condition_variable cond{};

      void sleep(auto pred) {
        std::unique_lock<std::mutex> lck(mutex);
        cond.wait(lck, pred);
      }

      void wakeUp(size_t count) {
          std::lock_guard<std::mutex> lck(mutex);
          if (count == 1) {
            cond.notify_one();
          } else {
            cond.notify_all();
          }
      }
  };

  std::vector<SemGroupData> groupsDatas_ = {};
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> currentThread_{0};
  size_t threadsPerGroup_ = 0;

 public:
  GroupSlot(size_t numberThreads, size_t threadsPerGroup = 4)
      : threadsPerGroup_(threadsPerGroup),
        groupsDatas_(numberThreads / threadsPerGroup + (numberThreads % threadsPerGroup == 0 ? 0 : 1)) {}

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
    assert(opts.threadId.has_value());
    assert((opts.threadId.value() / threadsPerGroup_) < groupsDatas_.size());
    auto &groupData = groupsDatas_[opts.threadId.value() / threadsPerGroup_];
    groupData.sleep([this, &slot]() {
      return slot->waitTerminationCondition() || terminate_.load(std::memory_order_relaxed);
    });
    return slot->canTerminate() || terminate_.load(std::memory_order_relaxed);
  }

  /// @brief Function used to wake up a thread attached to this condition variable
  void wakeUp(SlotWakeUpOptions const &opts) override {
    if (opts.count && opts.count.value() == 1) {
      size_t count = opts.count.value();
      size_t numberGroupsToWakeUp = count / threadsPerGroup_ + (count % threadsPerGroup_ == 0 ? 0 : 1);
      size_t startIdx = currentThread_.fetch_add(numberGroupsToWakeUp, std::memory_order_release);
      size_t endIdx = startIdx + numberGroupsToWakeUp;
      size_t numberGroups = groupsDatas_.size();

      for (size_t i = startIdx; i < endIdx; ++i) {
        auto &groupData = groupsDatas_[i % numberGroups];
        groupData.wakeUp(std::min(count, threadsPerGroup_));
        count -= threadsPerGroup_;
      }
    } else {
      for (auto &groupData : groupsDatas_) {
          groupData.wakeUp(threadsPerGroup_);
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
