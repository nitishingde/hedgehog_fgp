#ifndef KOKKOS_TBB_PARALLEL_REDUCE_HPP
#define KOKKOS_TBB_PARALLEL_REDUCE_HPP
#include "Kokkos_TBB_Execution_Space.hpp"
#include <Kokkos_Core.hpp>
#include <sstream>

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <typename CombinedFunctorReducerType, typename Policy>
class TBBRangeReducerBody {
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  using value_type = typename ReducerType::value_type;

  const FunctorType &m_functor;
  const ReducerType &m_reducer;
  value_type m_result;

 public:
  TBBRangeReducerBody(const FunctorType &f, const ReducerType &r)
    : m_functor(f), m_reducer(r) {
      m_reducer.init(&m_result);
  }

  TBBRangeReducerBody(const TBBRangeReducerBody &body, tbb::split)
    : TBBRangeReducerBody(body.m_functor, body.m_reducer) {}

  void operator()(const tbb::blocked_range<Member> &range) {
    if constexpr (std::is_void_v<WorkTag>) {
        for (Member i = range.begin(); i < range.end(); ++i) { // += range.chunk_size()?
          m_functor(i, m_result);
        }
    } else {
        const WorkTag t{};
        for (Member i = range.begin(); i < range.end(); ++i) { // += range.chunk_size()?
          m_functor(t, i, m_result);
        }
    }
  }

  void join(TBBRangeReducerBody &body) {
    m_reducer.join(&m_result, &body.m_result);
  }

  void get_final_result(value_type *result) {
    m_reducer.final(&m_result);
    *result = m_result;
  }
};


template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType, Kokkos::RangePolicy<Traits...>, Kokkos::OneTBB> {
 private:
  using Policy      = Kokkos::RangePolicy<Traits...>;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using Member  = typename Policy::member_type;

  using pointer_type   = typename ReducerType::pointer_type;
  using value_type     = typename ReducerType::value_type;
  using reference_type = typename ReducerType::reference_type;

  OneTBBImpl* m_instance;
  const CombinedFunctorReducerType m_functor_reducer;
  const Policy m_policy;
  const pointer_type m_result_ptr;

 public:
  inline void execute() const {
    const ReducerType& reducer = m_functor_reducer.get_reducer();
    const FunctorType& functor = m_functor_reducer.get_functor();

    if (m_policy.end() <= m_policy.begin()) {
      if (m_result_ptr) {
        reducer.init(m_result_ptr);
        reducer.final(m_result_ptr);
      }
      return;
    }
    enum {
      is_dynamic = std::is_same_v<typename Policy::schedule_type::type, Kokkos::Dynamic>
    };
    TBBRangeReducerBody<CombinedFunctorReducerType, Policy> body(functor, reducer);
    tbb::blocked_range<Member> range(m_policy.begin(), m_policy.end());
    if (is_dynamic) {
        m_instance->arena.execute([&] {
            tbb::parallel_reduce(range, body);
        });
    } else {
        m_instance->arena.execute([&] {
            tbb::parallel_reduce(range, body, tbb::static_partitioner());
        });
    }
    body.get_final_result(m_result_ptr);
  }

  //----------------------------------------

  template <class ViewType>
  inline ParallelReduce(const CombinedFunctorReducerType& arg_functor_reducer,
                        Policy arg_policy, const ViewType& arg_view)
      : m_instance(nullptr),
        m_functor_reducer(arg_functor_reducer),
        m_policy(arg_policy),
        m_result_ptr(arg_view.data()) {
    m_instance = arg_policy.space().impl_internal_space_instance();
    assert(m_instance != nullptr);
    static_assert(Kokkos::Impl::MemorySpaceAccess<typename ViewType::memory_space, Kokkos::HostSpace>::accessible,
                  "Kokkos::OneTBB reduce result must be a View accessible from HostSpace");
  }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// TODO: not tested
template <typename CombinedFunctorReducerType, typename Policy>
class TBBMDRangeReducerBody {
  using FunctorType   = typename CombinedFunctorReducerType::functor_type;
  using ReducerType   = typename CombinedFunctorReducerType::reducer_type;

  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  using value_type     = typename ReducerType::value_type;
  using pointer_type   = typename ReducerType::pointer_type;
  using reference_type = typename ReducerType::reference_type;

  using iterate_type = typename Kokkos::Impl::HostIterateTile<
      Policy, CombinedFunctorReducerType, WorkTag, reference_type>;

  const iterate_type &m_iter;
  value_type m_result;

 public:
  TBBMDRangeReducerBody(const iterate_type &iter)
    : m_iter(iter) {
      const ReducerType& reducer = iter.m_func.get_reducer();
      reducer.init(&m_result);
  }

  TBBMDRangeReducerBody(const TBBMDRangeReducerBody &body, tbb::split)
    : TBBMDRangeReducerBody(body.m_iter) {}

  void operator()(const tbb::blocked_range<Member> &range) {
    if constexpr (std::is_void_v<WorkTag>) {
        for (Member i = range.begin(); i < range.end(); ++i) { // += range.chunk_size()?
          m_iter(i, m_result);
        }
    } else {
        const WorkTag t{};
        for (Member i = range.begin(); i < range.end(); ++i) { // += range.chunk_size()?
          m_iter(t, i, m_result);
        }
    }
  }

  void join(TBBMDRangeReducerBody &body) {
    const ReducerType& reducer = m_iter.m_func.get_reducer();
    reducer.join(&m_result, &body.m_result);
  }

  void get_final_result(value_type *result) {
    const ReducerType& reducer = m_iter.m_func.get_reducer();
    reducer.final(&m_result);
    *result = m_result;
  }
};


// MDRangePolicy impl
template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType, Kokkos::MDRangePolicy<Traits...>, Kokkos::OneTBB> {
 private:
  using Policy      = Kokkos::MDRangePolicy<Traits...>;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  using pointer_type   = typename ReducerType::pointer_type;
  using value_type     = typename ReducerType::value_type;
  using reference_type = typename ReducerType::reference_type;

  using iterate_type = typename Kokkos::Impl::HostIterateTile<
      Policy, CombinedFunctorReducerType, WorkTag, reference_type>;

  OneTBBImpl* m_instance;
  const iterate_type m_iter;
  const pointer_type m_result_ptr;

  inline void exec_range(const Member ibeg, const Member iend,
                         reference_type update) const {
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      m_iter(iwork, update);
    }
  }

 public:
  inline void execute() const {
    TBBMDRangeReducerBody<CombinedFunctorReducerType, Policy> body(m_iter);
    tbb::blocked_range<Member> range(0, m_iter.m_rp.m_num_tiles);
    enum {
      is_dynamic =
          std::is_same_v<typename Policy::schedule_type::type, Kokkos::Dynamic>
    };

    if (is_dynamic) {
        m_instance->arena.execute([&] {
            tbb::parallel_reduce(range, body);
        });
    } else {
        m_instance->arena.execute([&] {
            tbb::parallel_reduce(range, body, tbb::static_partitioner());
        });
    }
  }

  //----------------------------------------

  template <class ViewType>
  ParallelReduce(const CombinedFunctorReducerType& arg_functor_reducer,
                 Policy arg_policy, const ViewType& arg_view)
      : m_instance(nullptr),
        m_iter(arg_policy, arg_functor_reducer),
        m_result_ptr(arg_view.data()) {
    m_instance = arg_policy.space().impl_internal_space_instance();
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename ViewType::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "Kokkos::OneTBB reduce result must be a View accessible from "
        "HostSpace");
  }

  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy&, const Functor&) {
    /**
     * 1024 here is just our guess for a reasonable max tile size,
     * it isn't a hardware constraint. If people see a use for larger
     * tile size products, we're happy to change this.
     */
    return 1024;
  }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

}  // namespace Impl
}  // namespace Kokkos

#endif /* KOKKOS_TBB_PARALLEL_REDUCE_HPP */
