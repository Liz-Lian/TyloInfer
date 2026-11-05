#pragma once

#include "engine/tensor.h"
#include "engine/weights/bias.h"
#include "engine/weights/moe/options.h"
#include "engine/weights/placement.h"
#include "engine/weights/source/read.h"
#include "engine/weights/tensor_parallel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class TqExec;
namespace tyloi { class NintRows; }
namespace tyloi::cuda { struct Experts; struct RoutedCache; }

struct MfqDropFileCacheGuard {
    bool& setting;
    bool previous;

    MfqDropFileCacheGuard(bool enabled, bool& value)
        : setting(value), previous(value) {
        setting = enabled;
    }

    ~MfqDropFileCacheGuard() {
        setting = previous;
    }
};

std::shared_ptr<tyloi::NintRows> load_nint_row_table(
    const tyloi::source::Source& source, const std::string& name, bool prefer_mapping=false);

namespace tyloi::weights {
// The source and cache owner outlive all loading, including streamed media layers.
struct Loader {
    using DenseTensor = ::tyloi::DenseTensor;
    struct Options {
        MoeLoadOptions quant;
        bool grouped = true;
        bool shard_dense = false;
        bool ranges = true;
        bool preload = false;
        bool mapped_embedding = false;
        int min_slots = 8;
    };
    struct Placement {
        int device = 0;
        bool cpu = false;
        bool drop_cache = false;
        ParallelConfig tensor;
        ParallelConfig experts;
    };

    const tyloi::source::Source& source;
    Options options;
    Placement placement;
    const char* role = "";

    Loader for_layer(int device, bool cpu) const {
        auto result = *this;
        result.placement.device = device;
        result.placement.cpu = cpu;
        return result;
    }
    bool has(const std::string& name) const { return has_tensor(source, name); }
    bool has_prefix(const std::string& prefix) const { return has_tensor_prefix(source, prefix); }
    // Dense parameters promote half precision to F32; native/host preserve stored dtype.
    std::shared_ptr<const ::tyloi::Tensor> tensor(const std::string& name) const;
    std::shared_ptr<const ::tyloi::Tensor> native(const std::string& name) const;
    std::shared_ptr<const ::tyloi::Tensor> host(const std::string& name) const;
    std::shared_ptr<const ::tyloi::Tensor> norm(const std::string& name, std::int64_t width) const;
    BiasWeight biased(const std::string& weight_name, const std::string& bias_name = {}) const;
    BiasWeight affine(const std::string& weight_name, const std::string& bias_name) const;
    std::shared_ptr<const ::tyloi::Tensor> weight(const std::string& name,
        std::optional<TensorParallelAxis> axis = std::nullopt,
        const std::vector<tyloi::TensorParallelSlice>* slices = nullptr) const;
    // preserve_boundaries controls storage packing only.
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> group(const std::vector<std::string>& names, size_t prefix = 0,
        const std::vector<tyloi::TensorParallelSlice>* slices = nullptr, bool preserve_boundaries = false) const;
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> gate_up(const std::vector<std::string>& names,
        const ::tyloi::Tensor& down, size_t prefix = 2, bool preserve_boundaries = false) const;
    std::shared_ptr<const ::tyloi::Tensor> embedding(const std::string& name, bool independent_output) const;
    std::shared_ptr<const ::tyloi::Tensor> experts(const std::string& name, bool cacheable = false,
        int layer = -1, const std::string& role = {}) const;
    std::shared_ptr<const ::tyloi::Tensor> dense(const std::string& name) const;
    std::shared_ptr<const ::tyloi::Tensor> routed(const std::string& name, int layer, int64_t experts, int64_t output,
                  int64_t input, const std::string& role = {}) const;
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> routed_gate_up(const std::string& prefix, int layer, int64_t experts, int64_t width,
                         int64_t input, const std::string& role = {}) const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    Loader(const tyloi::source::Source& source, Options options, Placement placement,
           cuda::RoutedCache& cache, std::shared_ptr<TqExec> tq);
    friend struct cuda::RoutedCache;
    DenseTensor host_value(const std::string& name) const;
    cuda::Experts build_experts(const std::string& name, bool cacheable, int layer,
                                const std::string& role) const;
};
} // namespace tyloi::weights
