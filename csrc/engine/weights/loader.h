#pragma once

#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/weights/bias.h"
#include "engine/weights/moe/experts.h"
#include "engine/weights/source/read.h"
#include "engine/weights/placement.h"
#include "engine/weights/moe/load.h"
#include "engine/weights/routed_cache.h"
#include "engine/weights/tq/routed.h"

#include <functional>

namespace tyloi::cuda { class Projector; }

std::vector<tyloi::TensorParallelSlice> plan_moe_expert_parallel_slices(
    const ParallelConfig& parallel, int64_t extent, const std::string& name);

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

bool is_quant_dtype(const std::string& dtype);
std::shared_ptr<tyloi::NintRows> load_nint_row_table(
    const tyloi::source::Source& source, const std::string& name, bool prefer_mapping=false);
std::shared_ptr<::tyloi::MfeTensor> load_mfe_cpu_offloaded(
    const tyloi::source::Source& source, const std::string& name);
::tyloi::DenseTensor mfe_dense_reference(
    const tyloi::source::Source& source, const std::string& name,
    ::tyloi::DenseTensor input, const std::vector<std::int32_t>& expert_ids,
    int tokens, int routes, bool routed_input);
::tyloi::DenseTensor materialize_mfe_dense(
    const tyloi::source::Source& source, const std::string& name);

namespace tyloi::cuda::weight_loader {
using DenseTensor = ::tyloi::DenseTensor;
struct Routed {
    using Function=std::function<DenseTensor(const Projector&,const DenseTensor&,const DenseTensor&)>;
    Function function;
    std::vector<std::shared_ptr<tyloi::cuda::Experts>> projections;
    std::vector<std::shared_ptr<TqRoutedWeight>> tq_projections;
    Routed()=default;
    template<class Callable, std::enable_if_t<!std::is_same_v<std::decay_t<Callable>,Routed>,int> =0>
    Routed(Callable&& call):function(std::forward<Callable>(call)) {}
    DenseTensor operator()(const Projector& project,const DenseTensor& input,const DenseTensor& ids)const {
        return function(project,input,ids);
    }
    explicit operator bool()const{return bool(function);}
};

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
    RoutedCache& cache;
    std::shared_ptr<TqExec> tq;
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
    DenseTensor tensor(const std::string& name) const;
    DenseTensor native(const std::string& name) const;
    DenseTensor host(const std::string& name) const;
    DenseTensor norm(const std::string& name, std::int64_t width) const;
    weights::BiasWeight biased(const std::string& weight_name, const std::string& bias_name = {}) const;
    weights::BiasWeight affine(const std::string& weight_name, const std::string& bias_name) const;
    QuantLinearGroup build_group(const std::vector<std::string>& names, size_t prefix = 0,
        const std::vector<tyloi::TensorParallelSlice>* slices = nullptr,
        bool preserve_boundaries = false) const;
    QuantLinearGroup build_gate_up(const std::vector<std::string>& names, const ::tyloi::Tensor& down,
        size_t prefix = 2, bool preserve_boundaries = false) const;
    Experts build_experts(const std::string& name, bool cacheable = false, int layer = -1,
                    const std::string& role = {}) const;
    std::shared_ptr<const ::tyloi::Tensor> weight(const std::string& name,
        std::optional<TensorParallelAxis> axis = std::nullopt,
        const std::vector<tyloi::TensorParallelSlice>* slices = nullptr) const;
    // preserve_boundaries controls storage packing only.
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> group(const std::vector<std::string>& names, size_t prefix = 0,
        const std::vector<tyloi::TensorParallelSlice>* slices = nullptr, bool preserve_boundaries = false) const;
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> gate_up(const std::vector<std::string>& names,
        const ::tyloi::Tensor& down, size_t prefix = 2, bool preserve_boundaries = false) const;
    std::shared_ptr<const ::tyloi::Tensor> embedding(const std::string& name, bool independent_output) const;
    std::shared_ptr<const tyloi::cuda::Experts> experts(const std::string& name, bool cacheable = false,
        int layer = -1, const std::string& role = {}) const;
    DenseTensor dense(const std::string& name) const;
    Routed routed(const std::string& name, int layer, int64_t experts, int64_t output,
                  int64_t input, const std::string& role = {}) const;
    Routed routed_gate_up(const std::string& prefix, int layer, int64_t experts, int64_t width,
                         int64_t input, const std::string& role = {}) const;
    static DenseTensor fp32(const DenseTensor& value) { return value.to(::tyloi::kFloat32).contiguous(); }
    static auto shape(const DenseTensor& value) { return value.sizes().vec(); }
    static int64_t elements(const DenseTensor& value) { return value.numel(); }
};
} // namespace tyloi::cuda::weight_loader
