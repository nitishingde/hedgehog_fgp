#ifndef KOKKOS_TBB_PARALLEL_SCAN_HPP
#define KOKKOS_TBB_PARALLEL_SCAN_HPP
#include "Kokkos_TBB_Execution_Space.hpp"
#include <Kokkos_Core.hpp>
#include <sstream>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <typename FunctorType, typename Policy, typename Analysis>
class TBBParallelScanBody {
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using Reducer   = typename Analysis::Reducer;

  using value_type     = typename Analysis::value_type;
  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const FunctorType m_functor;
  const Reducer m_reducer;
  value_type m_result;

 public:
  TBBParallelScanBody(FunctorType functor) : m_functor(functor), m_reducer(m_functor) {
    m_reducer.init(&m_result);
  }

  TBBParallelScanBody(TBBParallelScanBody& body, oneapi::tbb::split) : TBBParallelScanBody(body.m_functor) {}

  template<typename Tag>
  void operator()(const oneapi::tbb::blocked_range<Member>& r, Tag) {
      bool is_final = Tag::is_final_scan();

      if constexpr (std::is_void_v<WorkTag>) {
          for(Member iwork = r.begin(); iwork < r.end(); ++iwork) {
              m_functor(iwork, m_result, is_final);
          }
      } else {
          const WorkTag t{};
          for(Member iwork = r.begin(); iwork < r.end(); ++iwork) {
              m_functor(t, iwork, m_result, is_final);
          }
          if (Tag::is_final_scan()) {
              m_reducer.final(&m_result);
          }
      }
  }

  void reverse_join(TBBParallelScanBody& body) {
      m_reducer.join(&m_result, &body.m_result);
  }

  void assign(TBBParallelScanBody& body) {
      m_result = body.m_result;
  }

  void get_final_result(value_type *result) {
    *result = m_result;
  }
};

template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::OneTBB> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;

  using Analysis = FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType, void>;

  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  OneTBBImpl* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;

 public:
  inline void execute() const {
    TBBParallelScanBody<FunctorType, Policy, Analysis> body(m_functor);
    oneapi::tbb::parallel_scan(oneapi::tbb::blocked_range<Member>(m_policy.begin(), m_policy.end()), body);
  }

  //----------------------------------------

  inline ParallelScan(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_instance(nullptr), m_functor(arg_functor), m_policy(arg_policy) {
    m_instance = arg_policy.space().impl_internal_space_instance();
  }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>,
                            ReturnType, Kokkos::OneTBB> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;

  using Analysis = FunctorAnalysis<FunctorPatternInterface::SCAN, Policy,
                                   FunctorType, ReturnType>;

  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using value_type     = typename Analysis::value_type;
  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  OneTBBImpl* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
  const pointer_type m_result_ptr;

 public:
  inline void execute() const {
    TBBParallelScanBody<FunctorType, Policy, Analysis> body(m_functor);
    oneapi::tbb::parallel_scan(oneapi::tbb::blocked_range<Member>(m_policy.begin(), m_policy.end()), body);
    body.get_final_result(m_result_ptr);
  }

  //----------------------------------------

  template <class ViewType>
  ParallelScanWithTotal(const FunctorType& arg_functor,
                        const Policy& arg_policy,
                        const ViewType& arg_result_view)
      : m_instance(nullptr),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_result_ptr(arg_result_view.data()) {
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename ViewType::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "Kokkos::OneTBB parallel_scan result must be host-accessible!");
    m_instance = arg_policy.space().impl_internal_space_instance();
  }

  //----------------------------------------
};

}  // namespace Impl
}  // namespace Kokkos

#endif /* KOKKOS_TBB_PARALLEL_REDUCE_SCAN_HPP */
