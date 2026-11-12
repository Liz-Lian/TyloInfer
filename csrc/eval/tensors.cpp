#include "eval/tensors.h"
#include "engine/backends/cuda/native/tensor/backend.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace tyloi::eval {

DenseTensor read_tensor(const std::string& path, bool required) {
    if (!std::filesystem::is_regular_file(path)) {
        if (required) throw std::runtime_error("MiniCPM-o tensor file is missing: " + path);
        return {};
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("failed to open MiniCPM-o tensor file: " + path);
    const auto end = input.tellg();
    if (end <= 0) throw std::runtime_error("MiniCPM-o tensor file is empty: " + path);
    std::vector<char> bytes(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!input) throw std::runtime_error("failed to read MiniCPM-o tensor file: " + path);
    auto value = cuda::pickle_load(bytes);
    if (!value.isTensor()) throw std::runtime_error("MiniCPM-o input is not a tensor: " + path);
    return value.toTensor();
}

void write_tensor(const DenseTensor& value, const std::string& path) {
    if (!value.defined()) return;
    const std::filesystem::path output(path);
    if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
    auto bytes = cuda::pickle_save(value.detach().to(kCPU).contiguous());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("failed to create MiniCPM-o tensor file: " + path);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("failed to write MiniCPM-o tensor file: " + path);
}

std::int64_t read_scalar(const std::string& path, std::int64_t fallback) {
    auto value = read_tensor(path, false);
    if (!value.defined()) return fallback;
    if (value.numel() != 1) throw std::runtime_error("MiniCPM-o scalar tensor must contain one value: " + path);
    return value.to(kCPU, kInt64).item<std::int64_t>();
}

} // namespace tyloi::eval
