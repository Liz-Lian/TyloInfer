#pragma once

#include "quant_linear.h"
#include "engine/weights/moe/experts.h"
#include "source/read.h"
#include "backends/cuda/engine/weights/placement.h"
#include "backends/cuda/engine/weights/moe/load.h"
#include "backends/cuda/engine/weights/routed_cache.h"
#include "backends/cuda/engine/weights/tq/routed.h"

#include <functional>

namespace mfq::cuda { class Projector; }

std::vector<mfq::TensorParallelSlice> plan_moe_expert_parallel_slices(
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
std::shared_ptr<mfq::NintRows> load_nint_row_table(
    const mfq::ModelSource& source, const std::string& name, bool prefer_mapping=false);
std::shared_ptr<MixedMoeRuntime> load_mfe_cpu_offloaded(
    const mfq::ModelSource& source, const std::string& name);
mfq_tensor_backend::Tensor mfe_dense_reference(
    const mfq::ModelSource& source, const std::string& name,
    mfq_tensor_backend::Tensor input, const std::vector<std::int32_t>& expert_ids,
    int tokens, int routes, bool routed_input);
mfq_tensor_backend::Tensor materialize_mfe_dense(
    const mfq::ModelSource& source, const std::string& name);

namespace mfq::cuda::weight_loader {
using Tensor = mfq_tensor_backend::Tensor;
using Linear = std::shared_ptr<const QuantLinear>;
struct Routed {
    using Function=std::function<Tensor(const Projector&,const Tensor&,const Tensor&)>;
    Function function;
    std::vector<std::shared_ptr<mfq::cuda::Experts>> projections;
    std::vector<std::shared_ptr<TqRoutedWeight>> tq_projections;
    Routed()=default;
    template<class Callable, std::enable_if_t<!std::is_same_v<std::decay_t<Callable>,Routed>,int> =0>
    Routed(Callable&& call):function(std::forward<Callable>(call)) {}
    Tensor operator()(const Projector& project,const Tensor& input,const Tensor& ids)const {
        return function(project,input,ids);
    }
    explicit operator bool()const{return bool(function);}
};

// The source and cache owner outlive all loading, including streamed media layers.
struct Loader {
    using Tensor = mfq_tensor_backend::Tensor;
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

    const mfq::ModelSource& source;
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
    Tensor tensor(const std::string& name) const;
    Tensor native(const std::string& name) const;
    Tensor host(const std::string& name) const;
    QuantLinear projection(const std::string& name,
        std::optional<TensorParallelAxis> axis = std::nullopt,
        const std::vector<mfq::TensorParallelSlice>* slices = nullptr) const;
    QuantLinearGroup group(const std::vector<std::string>& names, size_t prefix = 0,
        const std::vector<mfq::TensorParallelSlice>* slices = nullptr,
        bool preserve_boundaries = false) const;
    QuantLinearGroup gate_up(const std::vector<std::string>& names, const QuantLinear& down,
        size_t prefix = 2, bool preserve_boundaries = false) const;
    QuantLinear embedding(const std::string& name, bool independent_output) const;
    Experts experts(const std::string& name, bool cacheable = false, int layer = -1,
                    const std::string& role = {}) const;
    Linear linear(const std::string& name) const;
    Tensor dense(const std::string& name) const;
    Routed routed(const std::string& name, int layer, int64_t experts, int64_t output,
                  int64_t input, const std::string& role = {}) const;
    Routed routed_gate_up(const std::string& prefix, int layer, int64_t experts, int64_t width,
                         int64_t input, const std::string& role = {}) const;
    static Tensor fp32(const Tensor& value) { return value.to(mfq_tensor_backend::kFloat32).contiguous(); }
    static auto shape(const Tensor& value) { return value.sizes().vec(); }
    static int64_t elements(const Tensor& value) { return value.numel(); }
};
} // namespace mfq::cuda::weight_loader
