#include <iostream>
#include <oneapi/tbb.h>

int main(){
    int sum = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<int>(1,101), 0,
        [](oneapi::tbb::blocked_range<int> const& r, int init) -> int {
            for (int v = r.begin(); v != r.end(); v++) {
                init += v;
            }
            return init;
        },
        [](int lhs, int rhs) -> int {
            return lhs + rhs;
        }
    );

    printf("Sum: %d\n", sum);
    return 0;
}
