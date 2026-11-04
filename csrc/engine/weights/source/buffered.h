#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>

namespace tyloi {

void read_buffered(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::byte* destination,
    std::size_t size);
std::function<void(std::uint64_t, std::byte*, std::size_t)> buffered_reader(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::uint64_t bytes);

} // namespace tyloi
