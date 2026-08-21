#ifndef LIMITED_LOCK_FREE_QUEUE_RECEIVER_H
#define LIMITED_LOCK_FREE_QUEUE_RECEIVER_H

#include <hedgehog.h>
#include <set>
#include <mutex>
#include <semaphore>
#include <memory>
#include "limited_lock_free_queue.hpp"

namespace hh {
namespace core {
namespace implementor {

template<class Input>
class LimitedLockFreeQueueReceiver : public ImplementorReceiver<Input> {
 private:
  std::unique_ptr<hh::LimitedLockFreeQueue<std::shared_ptr<Input>, 1024>> queue_;

  std::unique_ptr<std::set<abstraction::SenderAbstraction<Input> *>> const senders_ = nullptr;
  std::mutex sendersMutex_;

 public:
  /// @brief Default constructor
  /// @details Initialize the queue with a default node with no data (nullptr)
  LimitedLockFreeQueueReceiver()
      : queue_(std::make_unique<hh::LimitedLockFreeQueue<std::shared_ptr<Input>, 1024>>()),
        senders_(std::make_unique<std::set<abstraction::SenderAbstraction<Input> *>>()) {}

  /// @brief Default destructor
  virtual ~LimitedLockFreeQueueReceiver() {}

  /// @brief Accessor of the connected senders
  /// @return Set of connected senders
  std::set<abstraction::SenderAbstraction<Input> *> const &connectedSenders() const override { return *senders_; }

  /// @brief Add a sender to the set of connected senders
  /// @param sender Sender to add to the connected senders
  void addSender(abstraction::SenderAbstraction<Input> *sender) override {
    std::lock_guard<std::mutex> lck(sendersMutex_);
    senders_->insert(sender);
  }

  /// @brief Remove a sender to the set of connected senders
  /// @param sender Sender to remove from the connected senders
  void removeSender(abstraction::SenderAbstraction<Input> *sender) override {
    std::lock_guard<std::mutex> lck(sendersMutex_);
    senders_->erase(sender);
  }

  /// @brief Store a piece of data in the atomic queue
  /// @param data Data to store
  /// @return True
  bool receive(std::shared_ptr<Input> data, ReceiverReceiveOptions const &) override {
    while (!queue_->push(data)) {
        cross_platform_yield();
    }
    return true;
  }

  /// @brief Get a piece of data from the atomic queue
  /// @param data Reference uses to return the piece of data
  /// @return True
  bool getInputData(std::shared_ptr<Input> &data, ReceiverGetInputDataOptions const &) override {
      if (auto result = queue_->pop()) {
          data = *result;
          return true;
      }
      return false;
  }

  /// @brief Get the "current size" of the queue
  /// @return Current queue size
  size_t numberElementsReceived() override {
    return queue_->size();
  }

  /// @brief Accessor to the maximum filling size during the queue lifetime
  /// @return Maximum filling size during the queue lifetime
  [[nodiscard]] size_t maxNumberElementsReceived() const override {
    // this one will remain unimplemented because it is impossible to get an
    // accurate value. One top of that, it would add non negligible overhead.
    return 0;
  }

  /// @brief Test if the receiver is empty or not
  /// @return True if the receiver is empty, else false
  bool empty() override { return numberElementsReceived() == 0; }
};

/// @brief Concrete implementation of the receiver core abstraction for multiple types using LimitedLockFreeQueueReceiver
/// @tparam Inputs List of input types
template<class ...Inputs>
class MultiLimitedLockFreeQueueReceiver : public LimitedLockFreeQueueReceiver<Inputs> ... {
 public:
  /// @brief Default constructor
  explicit MultiLimitedLockFreeQueueReceiver() : LimitedLockFreeQueueReceiver<Inputs>()... {}

  /// Default destructor
  virtual ~MultiLimitedLockFreeQueueReceiver() = default;
};

/// @brief Base definition of the type deducer for MultiLimitedLockFreeQueueReceiver
/// @tparam Inputs Input types as tuple
template<class Inputs>
struct MultiLimitedLockFreeQueueReceiversTypeDeducer;

/// @brief Definition of the type deducer for MultiLimitedLockFreeQueueReceiver
/// @tparam Inputs Variadic of types
template<class ...Inputs>
struct MultiLimitedLockFreeQueueReceiversTypeDeducer<std::tuple<Inputs...>> {
  using type = core::implementor::MultiLimitedLockFreeQueueReceiver<Inputs...>; ///< Type accessor
};

/// @brief Helper to the deducer for MultiLimitedLockFreeQueueReceiver
/// @tparam TupleInputs Tuple of input types
template<class TupleInputs>
using MultiLimitedLockFreeQueueReceiversTypeDeducer_t = typename MultiLimitedLockFreeQueueReceiversTypeDeducer<TupleInputs>::type;

/// @brief Helper to the deducer for MultiLimitedLockFreeQueueReceiver from the nodes template parameters
/// @tparam Separator Separator of node template arg
/// @tparam AllTypes All types of node template arg
template<size_t Separator, class ...AllTypes>
using MLLFQR = MultiLimitedLockFreeQueueReceiversTypeDeducer_t<hh::tool::Inputs<Separator, AllTypes...>>;

}
}
}

#endif
