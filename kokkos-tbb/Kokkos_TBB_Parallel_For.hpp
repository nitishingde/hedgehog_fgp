#ifndef KOKKOS_TBB_PARALLEL_FOR_HPP
#define KOKKOS_TBB_PARALLEL_FOR_HPP
#include <oneapi/tbb.h>
#include <Kokkos_Core.hpp>
#include <sstream>
#include <unistd.h>
#include "Kokkos_TBB_Execution_Space.hpp"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define KOKKOS_PRAGMA_IVDEP_IF_ENABLED
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#undef KOKKOS_PRAGMA_IVDEP_IF_ENABLED
#define KOKKOS_PRAGMA_IVDEP_IF_ENABLED _Pragma("ivdep")
#endif

#ifndef KOKKOS_COMPILER_NVHPC
#define KOKKOS_TBB_OPTIONAL_CHUNK_SIZE , m_policy.chunk_size()
#else
#define KOKKOS_TBB_OPTIONAL_CHUNK_SIZE
#endif

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::OneTBB> {
 private:
  using Policy  = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  using index_type = typename Policy::index_type;

  const FunctorType m_functor;
  const Policy m_policy;
  OneTBBImpl *m_instance;

  inline static void exec_range(const FunctorType& functor, const Member ibeg,
                                const Member iend) {
    KOKKOS_PRAGMA_IVDEP_IF_ENABLED
    for (auto iwork = ibeg; iwork < iend; ++iwork) {
      exec_work(functor, iwork);
    }
  }

  inline static void exec_work(const FunctorType& functor, const Member iwork) {
    if constexpr (std::is_void_v<WorkTag>) {
      functor(iwork);
    } else {
      functor(WorkTag{}, iwork);
    }
  }

  template <class Policy>
  std::enable_if_t<std::is_same_v<typename Policy::schedule_type::type, Kokkos::Dynamic>>
  execute_parallel() const {
    if (m_policy.begin() >= m_policy.end()) return;
    m_instance->arena.execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<index_type>(m_policy.begin(), m_policy.end()),
            [this](const tbb::blocked_range<index_type>& r) {
                exec_range(m_functor, r.begin(), r.end());
            }
        );
    });
  }

  template <class Policy>
  std::enable_if_t<!std::is_same_v<typename Policy::schedule_type::type, Kokkos::Dynamic>>
  execute_parallel() const {
    m_instance->arena.execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<index_type>(m_policy.begin(), m_policy.end()),
            [this](const tbb::blocked_range<index_type>& r) {
                exec_range(m_functor, r.begin(), r.end());
            },
            tbb::static_partitioner()
        );
    });
  }

 public:
  inline void execute() const {
    execute_parallel<Policy>();
  }

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_functor(arg_functor),
        m_policy(std::move(arg_policy)) {
    m_instance = m_policy.space().impl_internal_space_instance();
  }
};

// TODO: this is untested
// MDRangePolicy impl
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>, Kokkos::OneTBB> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;
  using WorkTag       = typename MDRangePolicy::work_tag;

  using Member = typename Policy::member_type;

  using index_type   = typename Policy::index_type;
  using iterate_type = typename Kokkos::Impl::HostIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;

  OneTBBImpl *m_instance;
  const iterate_type m_iter;

  template <class Policy>
  typename std::enable_if_t<std::is_same_v<typename Policy::schedule_type::type, Kokkos::Dynamic>>
  execute_parallel() const {
    m_instance->arena.execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<index_type>(0, m_iter.m_rp.m_num_tiles),
            [this](const tbb::blocked_range<index_type>& r) {
                for (index_type iwork = r.begin(); iwork < r.end(); ++iwork) {
                    m_iter(iwork);
                }
            }
        );
    });
  }

  template <class Policy>
  std::enable_if_t<!std::is_same_v<typename Policy::schedule_type::type, Kokkos::Dynamic>>
  execute_parallel() const {
    m_instance->arena.execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<index_type>(0, m_iter.m_rp.m_num_tiles),
            [this](const tbb::blocked_range<index_type>& r) {
                for (index_type iwork = r.begin(); iwork < r.end(); ++iwork) {
                    m_iter(iwork);
                }
            },
            tbb::static_partitioner()
        );
    });
  }

 public:
  inline void execute() const {
    execute_parallel<Policy>();
  }

  inline ParallelFor(const FunctorType& arg_functor, MDRangePolicy arg_policy)
      : m_instance(nullptr), m_iter(arg_policy, arg_functor) {
    m_instance = arg_policy.space().impl_internal_space_instance();
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

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#undef KOKKOS_PRAGMA_IVDEP_IF_ENABLED
#undef KOKKOS_TBB_OPTIONAL_CHUNK_SIZE

#endif /* KOKKOS_TBB_PARALLEL_FOR_HPP */
