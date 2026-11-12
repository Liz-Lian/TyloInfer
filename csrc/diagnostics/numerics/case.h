#pragma once

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace tyloi::diagnostics::numerics {
using Json = nlohmann::json;

/** Report a CUDA failure with its operation. */
void cuda_check(cudaError_t status, const char *operation);

/** Own a device allocation until all submitted work has completed. */
class Buffer {
  public:
    /** Allocate device storage for the supplied byte count. */
    explicit Buffer(std::size_t bytes);
    ~Buffer();
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    /** Transfer allocation ownership without moving device contents. */
    Buffer(Buffer &&other) noexcept;
    /** Replace this allocation with the moved allocation. */
    Buffer &operator=(Buffer &&other) noexcept;
    void *data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

  private:
    void *data_ = nullptr;
    std::size_t size_ = 0;
};

/** Load one verified-by-caller case and publish its uncompressed observations. */
class Case {
  public:
    /** Open a case manifest and create a fresh raw output directory. */
    Case(const std::filesystem::path &input, const std::filesystem::path &output);
    const Json &data() const noexcept { return data_; }
    /** Read a named tensor after checking its encoding and byte count. */
    std::vector<std::uint8_t> input(const std::string &name) const;
    /** Read a named opaque asset after checking its byte count. */
    std::vector<std::uint8_t> asset(const std::string &name) const;
    /** Check actual native metadata and write one contiguous observation. */
    void output(const std::string &name, const Json &shape, const std::string &dtype,
                const void *values, std::size_t bytes);
    /** Publish complete raw metadata and actual execution evidence. */
    void finish(Json execution, Json checks);

  private:
    std::vector<std::uint8_t> read(const Json &spec) const;
    std::filesystem::path root_, output_;
    Json data_, outputs_ = Json::object();
};
} // namespace tyloi::diagnostics::numerics
