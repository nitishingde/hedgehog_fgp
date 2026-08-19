#ifndef LIMITED_LOCK_FREE_QUEUE_H
#define LIMITED_LOCK_FREE_QUEUE_H

#include <atomic>
#include <optional>
#include <unistd.h>
#include <cstdint>

namespace hh {

// this queue is an implementation of Dmitry Vyukov bounded lock free queue.
template <typename T, size_t Size>
class alignas(64) LimitedLockFreeQueue {
 private:
  struct alignas(64) Slot {
      T data;
      std::atomic<size_t> index;
  };
  static constexpr size_t Mask = Size - 1;
  alignas(64) Slot slots_[Size];
  alignas(64) std::atomic<size_t> head_{0};
  alignas(64) std::atomic<size_t> tail_{0};

 public:
  LimitedLockFreeQueue() {
      static_assert(((Size - 1) & Size) == 0, "Bounded lock free queue Size must be a power of 2.");
      for (size_t i = 0; i < Size; ++i) {
          slots_[i].index = i;
      }
  }

  bool push(T data) {
      size_t t = tail_.load();

      for (;;) {
          size_t index = slots_[t & Mask].index.load(std::memory_order_acquire);
          int64_t diff = static_cast<int64_t>(index) - static_cast<int64_t>(t);

          if (diff == 0) {
              if (tail_.compare_exchange_weak(t, t + 1)) {
                  break;
              }
          } else if (diff < 0) {
              return false;
          } else {
              cross_platform_yield();
              t = tail_.load();
          }
      }
      slots_[t & Mask].data = data;
      slots_[t & Mask].index.store(t + 1, std::memory_order_release);
      return true;
  }

  std::optional<T> pop() {
      size_t h = head_.load();

      for (;;) {
          size_t index = slots_[h & Mask].index.load(std::memory_order_acquire);
          int64_t diff = static_cast<int64_t>(index) - static_cast<int64_t>(h + 1);

          if (diff == 0) {
              if (head_.compare_exchange_weak(h, h + 1, std::memory_order_relaxed)) {
                  break;
              }
          } else if (diff < 0) {
              return std::nullopt;
          } else {
              cross_platform_yield();
              h = head_.load();
          }
      }
      T result = slots_[h & Mask].data;
      slots_[h & Mask].data = T{};
      slots_[h & Mask].index.store(h + Size, std::memory_order_release);
      return result;
  }

  size_t size() const {
      return tail_.load() - head_.load();
  }
};

}

#endif
