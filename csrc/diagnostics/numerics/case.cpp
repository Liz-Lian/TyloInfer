#include "case.h"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::diagnostics::numerics {
void cuda_check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
Buffer::Buffer(std::size_t bytes) : size_(bytes) {
    if (bytes)
        cuda_check(cudaMalloc(&data_, bytes), "cudaMalloc");
}
Buffer::~Buffer() {
    if (data_)
        cudaFree(data_);
}
Buffer::Buffer(Buffer &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
Buffer &Buffer::operator=(Buffer &&other) noexcept {
    if (this != &other) {
        if (data_)
            cudaFree(data_);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}
namespace {
std::filesystem::path safe(const std::filesystem::path &root, const std::string &name) {
    const std::filesystem::path path(name);
    if (path.empty() || path.is_absolute() || path.has_root_name())
        throw std::invalid_argument("invalid artifact path");
    for (const auto &component : path)
        if (component == ".." || component == ".")
            throw std::invalid_argument("invalid artifact path");
    const auto resolved = std::filesystem::weakly_canonical(root / path);
    const auto relative = resolved.lexically_relative(std::filesystem::canonical(root));
    if (relative.empty() || *relative.begin() == "..")
        throw std::invalid_argument("artifact escapes case root");
    return resolved;
}
std::size_t element_bytes(const std::string &dtype) {
    if (dtype == "float16")
        return 2;
    if (dtype == "float32" || dtype == "bfloat16")
        return 4;
    if (dtype == "int64")
        return 8;
    throw std::invalid_argument("unsupported tensor dtype");
}
std::string encoding(const std::string &dtype) {
    if (dtype == "float16")
        return "<f2";
    if (dtype == "float32" || dtype == "bfloat16")
        return "<f4";
    if (dtype == "int64")
        return "<i8";
    throw std::invalid_argument("unsupported tensor dtype");
}
std::size_t tensor_size(const Json &spec) {
    auto count = std::size_t{1};
    for (const auto &axis : spec.at("shape")) {
        const auto n = axis.get<std::int64_t>();
        if (n < 0 || (n && count > std::numeric_limits<std::size_t>::max() / std::size_t(n)))
            throw std::invalid_argument("invalid tensor shape");
        count *= std::size_t(n);
    }
    const auto bytes = element_bytes(spec.at("dtype"));
    if (count > std::numeric_limits<std::size_t>::max() / bytes)
        throw std::overflow_error("tensor byte count overflow");
    return count * bytes;
}
} // namespace
Case::Case(const std::filesystem::path &input, const std::filesystem::path &output)
    : root_(std::filesystem::canonical(input).parent_path()), output_(output) {
    const std::uint16_t endian = 1;
    if (*reinterpret_cast<const std::uint8_t *>(&endian) != 1)
        throw std::runtime_error("numerics requires a little-endian host");
    std::ifstream stream(input, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open case");
    stream >> data_;
    if (data_.at("kind") != "case" || data_.at("version") != 1 || !data_.at("inputs").is_object() ||
        !data_.at("outputs").is_object())
        throw std::invalid_argument("invalid case manifest");
    if (!std::filesystem::create_directory(output_))
        throw std::runtime_error("raw output directory already exists or cannot be created");
}
std::vector<std::uint8_t> Case::read(const Json &spec) const {
    const auto path = safe(root_, spec.at("file").get<std::string>());
    const auto bytes = spec.at("nbytes").get<std::size_t>();
    if (std::filesystem::file_size(path) != bytes)
        throw std::runtime_error("artifact byte count mismatch");
    std::vector<std::uint8_t> result(bytes);
    std::ifstream stream(path, std::ios::binary);
    if (!stream || (bytes && !stream.read(reinterpret_cast<char *>(result.data()),
                                          static_cast<std::streamsize>(bytes))))
        throw std::runtime_error("cannot read artifact");
    return result;
}
std::vector<std::uint8_t> Case::input(const std::string &name) const {
    const auto &spec = data_.at("inputs").at(name);
    if (spec.at("order") != "C" || spec.at("encoding") != encoding(spec.at("dtype")) ||
        tensor_size(spec) != spec.at("nbytes").get<std::size_t>())
        throw std::invalid_argument("invalid input tensor encoding");
    return read(spec);
}
std::vector<std::uint8_t> Case::asset(const std::string &name) const {
    return read(data_.at("assets").at(name));
}
void Case::output(const std::string &name, const Json &shape, const std::string &dtype,
                  const void *values, std::size_t bytes) {
    const auto &contract = data_.at("outputs").at(name);
    if (shape != contract.at("shape") ||
        dtype != contract.value("actual_dtype", contract.at("dtype").get<std::string>()))
        throw std::invalid_argument("native output shape or dtype differs: " + name);
    Json spec = {{"shape", shape}, {"dtype", dtype}, {"encoding", encoding(dtype)},
                 {"layout", "contiguous"},        {"order", "C"},   {"nbytes", bytes},
                 {"file", name + ".bin"}};
    if (name.empty() ||
        name.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
            std::string::npos ||
        outputs_.contains(name) || tensor_size(spec) != bytes)
        throw std::invalid_argument("invalid output observation");
    std::ofstream stream(output_ / spec.at("file").get<std::string>(), std::ios::binary);
    if (!stream || (bytes && !stream.write(static_cast<const char *>(values),
                                           static_cast<std::streamsize>(bytes))))
        throw std::runtime_error("cannot write output");
    outputs_[name] = std::move(spec);
}
void Case::finish(Json execution, Json checks) {
    if (outputs_.size() != data_.at("outputs").size())
        throw std::logic_error("missing output observations");
    std::ofstream stream(output_ / "raw.json", std::ios::binary);
    if (!stream || !(stream << Json{{"outputs", outputs_},
                                    {"execution", std::move(execution)},
                                    {"checks", std::move(checks)}}
                                   .dump(2)))
        throw std::runtime_error("cannot publish raw result");
}
} // namespace tyloi::diagnostics::numerics
