#ifndef TBB_REDUCE
#define TBB_REDUCE
#include <vector>
#include <span>
#include <oneapi/tbb.h>

template <typename T>
T tbb_reduce(std::span<T> const &v) {
    T sum = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<int>(0, v.size()), 0,
        [&](oneapi::tbb::blocked_range<int> const& r, T init) -> T {
            for (int i = r.begin(); i != r.end(); i++) {
                init += v[i];
            }
            return init;
        },
        [](T lhs, int rhs) -> T {
            return lhs + rhs;
        }
    );
    return sum;
}

#endif
