#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include "Kokkos_TBB_Execution_Space.hpp"
#include "Kokkos_TBB_Parallel_For.hpp"
#include "Kokkos_TBB_Parallel_Reduce.hpp"
#include "Kokkos_TBB_Parallel_Scan.hpp"

void test_for_reduce(size_t num_threads, size_t N) {
  Kokkos::OneTBB tbb_space(num_threads);
  // Allocate a 1-dimensional view of integers
  Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", N);
  // Fill view with sequentially increasing values v=[0,1,2,3,4]
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::OneTBB>(tbb_space, 0, N),
      KOKKOS_LAMBDA(size_t n) {
        v(n) = n + 1;
      }
  );
  int r;
  Kokkos::parallel_reduce(
    "accumulate",
    Kokkos::RangePolicy<Kokkos::OneTBB>(tbb_space, 0, N),
    KOKKOS_LAMBDA(size_t i, int& partial_r) {
      partial_r += v(i);
    }, r);
  // Check the result
  assert(r == (100 * 101) / 2);
}

void test_scan(size_t num_threads, size_t N) {
  Kokkos::OneTBB tbb_space(num_threads);
  int64_t result;
  Kokkos::View<int64_t*> in_scan("inclusive_scan", N);
  Kokkos::View<int64_t*> ex_scan("exclusive_scan", N);

  Kokkos::parallel_scan("Loop1",
    Kokkos::RangePolicy<Kokkos::OneTBB>(tbb_space, 0, N),
    KOKKOS_LAMBDA(int64_t i, int64_t& partial_sum, bool is_final) {
    if (is_final) ex_scan(i) = partial_sum;
    partial_sum += i;
    if (is_final) in_scan(i) = partial_sum;
  }, result);

  // exclusive scan: 0,0,1,3,6,10,...
  // inclusive scan: 0,1,3,6,10,...
  // result: N*(N-1)/2
  printf("Result: %i %li (expected: %li)\n", N, result, N * (N - 1)/2);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
      test_for_reduce(2, 100);
      test_scan(2, 100);
  }
  Kokkos::printf("Goodbye World\n");
  Kokkos::finalize();
  return 0;
}
