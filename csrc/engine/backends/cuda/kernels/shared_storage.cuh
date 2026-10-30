#pragma once

#include <cstddef>
#include <type_traits>

namespace mfq::cuda {

template <typename T, std::size_t Count = 1>
struct SharedStorage {
    static_assert(Count > 0 && std::is_aggregate_v<T> && std::is_trivially_copyable_v<T>);
    alignas(T) unsigned char bytes[sizeof(T) * Count];
    __device__ T* data() { return reinterpret_cast<T*>(bytes); }
};

}
