#ifndef KOKKOS_TBB_EXECUTION_SPACE
#define KOKKOS_TBB_EXECUTION_SPACE
#include <oneapi/tbb/task_arena.h>
#include <cstdint>
#include <memory>

namespace Kokkos {

struct OneTBBImpl {
  oneapi::tbb::task_arena arena;
  OneTBBImpl(unsigned int num_threads) : arena(num_threads) {}
};

// This is not an actual class, it just describes the concept in shorthand
class OneTBB {
  std::shared_ptr<OneTBBImpl> m_impl = nullptr;

public:
    typedef OneTBB execution_space;
    typedef HostSpace memory_space;
    typedef Device<execution_space, memory_space> device_type;
    typedef LayoutRight array_layout;
    typedef memory_space::size_type size_type;
    // typedef memory_space::index_type index_type;
    typedef size_t index_type;

    OneTBB(unsigned int num_threads = std::thread::hardware_concurrency())
          : m_impl(std::make_shared<OneTBBImpl>(num_threads)) {}
    OneTBB(const OneTBB& src) : m_impl(src.m_impl) {}

    const char* name() const { return "TBBSpace"; }
    uint32_t impl_instance_id() const noexcept { return 1; }

    void print_configuration([[maybe_unused]] std::ostream &ostr) const { /* TODO */ }
    void print_configuration([[maybe_unused]] std::ostream &ostr, bool details) const { /* TODO */ }

    int concurrency() const { return m_impl->arena.max_concurrency(); }
    oneapi::tbb::task_arena &arena() const { return m_impl->arena; }

    OneTBBImpl *impl_internal_space_instance() const { return m_impl.get(); }

    /* TODO: we will need to use the task groups to be able to wait (shouldn't be necessary) */
    void fence(const std::string&) const {}
    void fence() const {}

    friend bool operator==(const execution_space& lhs, const execution_space& rhs);
};

inline bool operator==(const OneTBB& lhs, const OneTBB& rhs) {
    return lhs.m_impl == rhs.m_impl;
}

// template<class MS>
// struct is_execution_space {
// enum { value = false };
// };

// template<>
// struct is_execution_space<OneTBB> {
// enum { value = true };
// };

namespace Tools {
namespace Experimental {
template <>
struct DeviceTypeTraits<OneTBB> {
  static constexpr DeviceType id = DeviceType::Unknown; // we could also use Threads, HPX or OpenMP??
  static int device_id(const OneTBB&) { return 0; }
};
}  // namespace Experimental
}  // namespace Tools

}  // namespace Kokkos

#endif
