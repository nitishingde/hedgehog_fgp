#ifndef TBB_REDUCE
#define TBB_REDUCE
#include <vector>
#include <oneapi/tbb.h>

inline int tbb_reduce(std::vector<int> const &v) {
    int sum = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<int>(0, v.size()), 0,
        [&](oneapi::tbb::blocked_range<int> const& r, int init) -> int {
            for (int i = r.begin(); i != r.end(); i++) {
                init += v[i];
            }
            return init;
        },
        [](int lhs, int rhs) -> int {
            return lhs + rhs;
        }
    );
    return sum;
}

#endif
