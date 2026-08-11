#ifndef TBB_REDUCE
#define TBB_REDUCE
#include <vector>
#include <span>
#include <oneapi/tbb.h>

template <typename T>
T tbb_reduce(std::span<T> const &v) {
    T sum = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<std::size_t>(0, v.size()), T{0},
        [&](oneapi::tbb::blocked_range<std::size_t> const& r, T init) -> T {
            for (std::size_t i = r.begin(); i != r.end(); i++) {
                init += v[i];
            }
            return init;
        },
        [](T lhs, T rhs) -> T {
            return lhs + rhs;
        }
    );
    return sum;
}

#endif
