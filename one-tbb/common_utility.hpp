#ifndef HEDGEHOG_FGP_COMMON_UTILITY_H
#define HEDGEHOG_FGP_COMMON_UTILITY_H

#include <chrono>

namespace detail {
  template<typename>
  struct is_duration: std::false_type {};

  template<typename Rep, typename Period>
  struct is_duration<std::chrono::duration<Rep, Period>>: std::true_type {};
}

template<typename T>
concept StdChronoDuration = detail::is_duration<T>::value;

template<StdChronoDuration Duration>
[[nodiscard]] static auto toSeconds(Duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

template<StdChronoDuration Duration>
[[nodiscard]] static auto toMilliSeconds(Duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count();
}

template<StdChronoDuration Duration>
[[nodiscard]] static auto toMicroSeconds(Duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(duration).count();
}

template<StdChronoDuration Duration>
[[nodiscard]] static auto toNanoSeconds(Duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(duration).count();
}

#endif //HEDGEHOG_FGP_COMMON_UTILITY_H
