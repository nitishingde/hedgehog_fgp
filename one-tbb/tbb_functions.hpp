#ifndef TBB_FUNCTIONS
#define TBB_FUNCTIONS
#include <vector>
#include <span>
#include <oneapi/tbb.h>
#include <oneapi/tbb/parallel_for.h>

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

template <typename T>
void tbb_saxpy(std::span<T> z, T a, std::span<T> x, std::span<T> y) {
    oneapi::tbb::parallel_for(0UL, z.size(), 1UL, [&](size_t i) {
        z[i] = a * x[i] + y[i];
    });
}

#endif
