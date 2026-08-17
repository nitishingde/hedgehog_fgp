#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cassert>
#include "Kokkos_TBB_Parallel_For.hpp"
#include "Kokkos_TBB_Parallel_Reduce.hpp"
#include "Kokkos_TBB_Execution_Space.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::OneTBB tbb_space(2);
    // Allocate a 1-dimensional view of integers
    Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", 100);
    // Fill view with sequentially increasing values v=[0,1,2,3,4]
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OneTBB>(tbb_space, 0, 100),
        KOKKOS_LAMBDA(size_t n) {
          v(n) = n + 1;
        }
    );
    int r;
    Kokkos::parallel_reduce(
      "accumulate",
      Kokkos::RangePolicy<Kokkos::OneTBB>(tbb_space, 0, 100),
      KOKKOS_LAMBDA(size_t i, int& partial_r) {
        partial_r += v(i);
      }, r);
    // Check the result
    assert(r == (100 * 101) / 2);
  }
  Kokkos::printf("Goodbye World\n");
  Kokkos::finalize();
  return 0;
}
