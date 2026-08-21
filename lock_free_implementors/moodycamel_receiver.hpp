#ifndef LOCK_FREE_IMPLEMENTORS_MOODYCAMEL_RECEIVER_H
#define LOCK_FREE_IMPLEMENTORS_MOODYCAMEL_RECEIVER_H

#include <hedgehog.h>
#include <set>
#include <mutex>
#include <semaphore>
#include <memory>
#include <concurrentqueue.h>

namespace hh {
namespace core {
namespace implementor {

template<class Input>
class MoodycamelReceiver : public ImplementorReceiver<Input> {
 private:
  std::unique_ptr<moodycamel::ConcurrentQueue<std::shared_ptr<Input>>> queue_;

  std::unique_ptr<std::set<abstraction::SenderAbstraction<Input> *>> const senders_ = nullptr;
  std::mutex sendersMutex_;

 public:
  /// @brief Default constructor
  /// @details Initialize the queue with a default node with no data (nullptr)
  MoodycamelReceiver()
      : queue_(std::make_unique<moodycamel::ConcurrentQueue<std::shared_ptr<Input>>>()),
        senders_(std::make_unique<std::set<abstraction::SenderAbstraction<Input> *>>()) {}

  /// @brief Default destructor
  virtual ~MoodycamelReceiver() {}

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
    while (!queue_->enqueue(data)) {
        cross_platform_yield();
    }
    return true;
  }

  /// @brief Get a piece of data from the atomic queue
  /// @param data Reference uses to return the piece of data
  /// @return True
  bool getInputData(std::shared_ptr<Input> &data, ReceiverGetInputDataOptions const &) override {
      if (auto result = queue_->try_dequeue(data)) {
          return true;
      }
      return false;
  }

  /// @brief Get the "current size" of the queue
  /// @return Current queue size
  size_t numberElementsReceived() override {
    return queue_->size_approx();
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

/// @brief Concrete implementation of the receiver core abstraction for multiple types using MoodycamelReceiver
/// @tparam Inputs List of input types
template<class ...Inputs>
class MultiMoodycamelReceivers : public MoodycamelReceiver<Inputs> ... {
 public:
  /// @brief Default constructor
  explicit MultiMoodycamelReceivers() : MoodycamelReceiver<Inputs>()... {}

  /// Default destructor
  virtual ~MultiMoodycamelReceivers() = default;
};

/// @brief Base definition of the type deducer for MultiMoodycamelReceivers
/// @tparam Inputs Input types as tuple
template<class Inputs>
struct MultiMoodycamelReceiversTypeDeducer;

/// @brief Definition of the type deducer for MultiMoodycamelReceivers
/// @tparam Inputs Variadic of types
template<class ...Inputs>
struct MultiMoodycamelReceiversTypeDeducer<std::tuple<Inputs...>> {
  using type = core::implementor::MultiMoodycamelReceivers<Inputs...>; ///< Type accessor
};

/// @brief Helper to the deducer for MultiMoodycamelReceivers
/// @tparam TupleInputs Tuple of input types
template<class TupleInputs>
using MultiMoodycamelReceiversTypeDeducer_t = typename MultiMoodycamelReceiversTypeDeducer<TupleInputs>::type;

/// @brief Helper to the deducer for MultiMoodycamelReceivers from the nodes template parameters
/// @tparam Separator Separator of node template arg
/// @tparam AllTypes All types of node template arg
template<size_t Separator, class ...AllTypes>
using MCLR = MultiMoodycamelReceiversTypeDeducer_t<hh::tool::Inputs<Separator, AllTypes...>>;

}
}
}

#endif
