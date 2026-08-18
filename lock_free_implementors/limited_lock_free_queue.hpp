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
  static constexpr size_t Mask = Size - 1;
  alignas(64) T datas_[Size];
  alignas(64) std::atomic<size_t> indices_[Size];
  alignas(64) std::atomic<size_t> head_{0};
  alignas(64) std::atomic<size_t> tail_{0};

 public:
  LimitedLockFreeQueue() {
      static_assert(((Size - 1) & Size) == 0, "Bounded lock free queue Size must be a power of 2.");
      for (size_t i = 0; i < Size; ++i) {
          indices_[i] = i;
      }
  }

  bool push(T data) {
      auto t = tail_.load(std::memory_order_acquire);

      for (;;) {
          auto index = indices_[t & Mask].load(std::memory_order_acquire);
          auto diff = static_cast<int64_t>(index) - static_cast<int64_t>(t);

          if (diff == 0) {
              if (tail_.compare_exchange_weak(t, t + 1, std::memory_order_relaxed)) {
                  break;
              }
          } else if (diff < 0) {
              return false;
          } else {
              cross_platform_yield();
              t = tail_.load(std::memory_order_acquire);
          }
      }
      datas_[t & Mask] = data;
      indices_[t & Mask].store(t + 1, std::memory_order_release);
      return true;
  }

  std::optional<T> pop() {
      auto h = head_.load(std::memory_order_acquire);

      for (;;) {
          auto index = indices_[h & Mask].load(std::memory_order_acquire);
          auto diff = static_cast<int64_t>(index) - static_cast<int64_t>(h + 1);

          if (diff == 0) {
              if (head_.compare_exchange_weak(h, h + 1, std::memory_order_relaxed)) {
                  break;
              }
          } else if (diff < 0) {
              return std::nullopt;
          } else {
              cross_platform_yield();
              h = tail_.load(std::memory_order_acquire);
          }
      }
      auto result = datas_[h & Mask];
      indices_[h & Mask].store(h + Size, std::memory_order_release);
      return result;
  }

  // TODO: using relaxed order may be dangerous when used in hedgehog (this should be properly tested on ARM).
  size_t size() const {
      return tail_.load(std::memory_order_relaxed) - head_.load(std::memory_order_relaxed);
  }
};

}

#endif
