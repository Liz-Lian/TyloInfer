#pragma once

#include <cstdint>
#include <optional>

namespace mfq {
struct HostMemorySample {
    std::uint64_t resident_bytes = 0;
    std::uint64_t available_bytes = 0;
    std::optional<std::uint64_t> commit_available_bytes;
    std::uint64_t available_host_bytes() const noexcept {
        return commit_available_bytes && *commit_available_bytes < available_bytes
            ? *commit_available_bytes : available_bytes;
    }
};
HostMemorySample sample_host_memory();
}
