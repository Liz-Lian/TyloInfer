#pragma once
#include "formats/format_features.h"
#include "engine/executor/run.h"
#include "engine/executor/ffn.h"
#include "engine/executor/moe_ffn_pipeline.h"
#include "ops/common/include/gated_residual.h"
namespace mfq::cuda { class KvOffloadStore; class KvMemoryManager; }

#include "backend.h"
#include "native/cuda/branches.h"
#include "native/cuda/collectives.h"
#include "native/cuda/profile.h"
#include "diagnostics/route_stats.h"
#include "ops/common/include/mmq.h"
#include "ops/common/include/parallel.h"
#include "ops/common/include/projection.h"
#include "ops/attention/scratch.h"
#include "engine/weights/tensor_parallel.h"
#include "engine/models/common/yarn.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef TYLOINFER_CUDA_CHECK
#define TYLOINFER_CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
    } \
} while (0)
#endif

class MoeExpertCache;
class TqCache;
class TqExec;

// One load owner outlives its execution context and every model borrowing it.
struct CudaStores {
    CudaStores() = default;
    CudaStores(const CudaStores&) = delete;
    CudaStores& operator=(const CudaStores&) = delete;
    std::shared_ptr<mfq::cuda::KvMemoryManager> kv_memory;
    std::shared_ptr<MoeExpertCache> moe_expert_cache;
    std::shared_ptr<TqCache> tq_expert_cache;
    std::shared_ptr<mfq::cuda::KvOffloadStore> qsa_kv_store;

    // Drain execution before retiring backing; leases may retain individual handles.
    void reset() noexcept {
        qsa_kv_store.reset();
        kv_memory.reset();
        moe_expert_cache.reset();
        tq_expert_cache.reset();
    }
};

namespace mfq::cuda::internal {

class PrefillCudaTimer {
public:
    PrefillCudaTimer();
    ~PrefillCudaTimer();

    PrefillCudaTimer(const PrefillCudaTimer&) = delete;
    PrefillCudaTimer& operator=(const PrefillCudaTimer&) = delete;

    cudaEvent_t finished_event() const;
    double elapsed_ms() const;

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t started_ = nullptr;
    cudaEvent_t finished_ = nullptr;
};

} // namespace mfq::cuda::internal

struct CudaPreparedPrompt {
    std::vector<std::int64_t> token_ids;
    mfq_tensor_backend::Tensor embeddings;
    mfq_tensor_backend::Tensor positions;
    int decode_position_delta = 0;
    std::string cache_key;
    mfq_tensor_backend::Tensor image_mask;

    bool transformed() const noexcept {
        return embeddings.defined() || positions.defined() ||
            decode_position_delta != 0;
    }
};

// Image participation is scoped to one prepared prefill chunk. It must not
// survive cancellation, a failed forward, ordinary decode or graph capture.
template<class Model> class CudaImageMaskScope {
    Model& model_;
    bool active_=false;
public:
    CudaImageMaskScope(Model& model,const CudaPreparedPrompt* prepared,
                       int64_t offset,int64_t count):model_(model) {
        if(!prepared || !prepared->image_mask.defined())return;
        if constexpr(requires {model_.adapter_set_image_mask(prepared->image_mask);}) {
            model_.adapter_set_image_mask(prepared->image_mask.narrow(1,offset,count).contiguous());
            active_=true;
        } else throw std::invalid_argument("model cannot apply an image participation mask");
    }
    ~CudaImageMaskScope() {
        if constexpr(requires {model_.adapter_set_image_mask(mfq_tensor_backend::Tensor{});})
            if(active_)model_.adapter_set_image_mask({});
    }
    CudaImageMaskScope(const CudaImageMaskScope&)=delete;
    CudaImageMaskScope& operator=(const CudaImageMaskScope&)=delete;
};

struct ParallelConfig {
    std::vector<int> devices;
    std::vector<double> split;
    bool allow_duplicate_devices = false;

    bool enabled() const {
        return devices.size() > 1;
    }

    int primary_device() const {
        return devices.empty() ? 0 : devices.front();
    }
};

struct LayerPlacementConfig {
    std::vector<int> devices;
    std::vector<double> split;
    std::vector<int> layer_devices;
    int load_device = -1;

    bool enabled() const {
        return devices.size() > 1;
    }

    int primary_device() const {
        return devices.empty() ? 0 : devices.front();
    }

    void prepare(int64_t layers);
    int device_for_layer(int64_t layer) const;
};

struct CudaExecutionConfig {
    mfq::cuda::Executor::Options tasks;
    MoeFfnOptions moe_ffn;
    std::string profile_filter;
    bool report_cuda_memory = false;
    bool model_parallel_cuda_graph = true;
    bool tensor_parallel_cuda_graph = true;
    bool expert_parallel_cuda_graph = true;
    bool tensor_parallel_shared_linear_attention_input = true;
    bool tensor_parallel_mirror_linear_attention_scalars = true;
    bool tensor_parallel_mirror_qwen35_attention_kv = true;
    bool tensor_parallel_shard_native_float = false;
    bool nvq2_exec = false;
    bool nvq_extended_group_exec = false;
    mfq::cuda::FfnExec::Options ffn;
    bool gemma4_fused_norms = true;
    bool dsv4_groupwise_output_a = true;
    bool diagnostic_nint_group = true;
    mfq::cuda::GatedResidual::Options residual;
    bool gdn_transposed_state = true;
    bool gdn_fused_output = true;
    bool gdn_fused_preparation = true;
    bool gdn_fused_core = true;
    bool linear_conv_prefill_fused = true;
    std::size_t deepseek_v41_engram_cache_rows = 16'384;
    bool moe_mapped_gather = false;
    int moe_mapped_copy_blocks = 64;
    int moe_ssd_io_workers = 8;
    bool moe_ssd_overlap = true;
    bool moe_ssd_ranges = true;
    std::optional<std::size_t> moe_device_cache_bytes;
    std::size_t moe_host_cache_bytes = 0;
    bool moe_pipeline = false;
    bool moe_hybrid_cpu = false;
    bool moe_preload_all = false;
    bool moe_assert_resident = false;
    bool moe_ram_pcie = false;
    bool moe_direct_ram = true;
    // Diagnostics may hold primary residency constant across paired passes.
    bool moe_residency_adapt = true;
    bool moe_residency_warm = true;
    bool moe_residency_projection_heat = true;
    std::size_t moe_host_physical_bytes = 0;
    std::optional<double> moe_ram_pcie_fraction;
};

CudaExecutionConfig load_cuda_execution_config();

struct CudaExecutionContext {
    explicit CudaExecutionContext(CudaStores& stores);
    ~CudaExecutionContext();
    CudaExecutionContext(const CudaExecutionContext&) = delete;
    CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mfq::cuda::ContextSet native_contexts;
#endif
    bool yarn_context_extension=false;
    mfq::models::YarnScaling yarn;

    CudaProfiler profiler;
    CudaExecutionConfig config;
    bool force_moe_unfused_reduce = false;
    bool force_moe_materialized_swiglu = false;
    KlMmqState kl_mmq;
    std::unordered_set<int> dsv4_cpu_offload_layers;
    int64_t dsv4_cpu_offload_host_bytes = 0;
    int n_gpu_layers = -1;
    int dense_cpu_layer_count = 0;
    bool loading_cpu_layer = false;
    bool drop_file_cache = false;
    bool decode_graph_serial_branches = false;
    bool decode_graph_tp_projection_major = false;
    bool continuous_batch_cache_serial = false;
    int continuous_batch_ffn_tile = 0;
    int continuous_batch_ffn_tiles = 1;
    int moe_cache_registration_min_slots = 8;
    int gemma_trace_layer = -1;
    mfq::cuda::Trace* gemma_stage_trace = nullptr;
    ParallelConfig tensor_parallel;
    ParallelConfig expert_parallel;
    mfq::cuda::Collectives collectives;
    LayerPlacementConfig layer_placement;
    CudaStores& stores;
    std::shared_ptr<TqExec> tq;
    std::vector<std::shared_ptr<MoeFfnPipeline>> moe_pipelines;
    mfq::cuda::Branches branches;
    // Layer addresses identify reusable workspace; only execution retires it after drain.
    std::unordered_map<const void*, mfq::cuda::AttentionScratch> attention_scratch;
    mfq::cuda::RouteStats routes;

    mfq::cuda::Projector projector();
    mfq::cuda::FfnExec ffn();
    void drain();
    void reset();
    void register_moe_pipeline(std::shared_ptr<MoeFfnPipeline> pipeline);
};

void mfq_release_host_allocator_cache() noexcept;
struct CudaScope {
    MfqCudaGuard device;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mfq::cuda::ContextGuard context;
#endif
    explicit CudaScope(CudaExecutionContext& execution)
        : device(execution.layer_placement.primary_device())
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        , context(execution.native_contexts)
#endif
    {}
};

bool model_parallel_enabled(const CudaExecutionContext& execution);
const ParallelConfig& model_parallel_config(
    const CudaExecutionContext& execution);
const ParallelConfig& moe_parallel_config(
    const CudaExecutionContext& execution);
int model_parallel_primary_device(const CudaExecutionContext& execution);
bool model_parallel_cuda_graph_enabled(const CudaExecutionContext& execution);
int active_weight_load_device(const CudaExecutionContext& execution);
void trace_gemma_stage(
    CudaExecutionContext& execution,
    int layer,
    const char* name,
    const mfq_tensor_backend::Tensor& value);
void report_cuda_memory(
    const CudaExecutionConfig& config,
    const char* stage);

struct DecodeGraphBranchScope {
    bool& setting;
    bool previous;

    explicit DecodeGraphBranchScope(bool& value)
        : setting(value), previous(value) {
        setting = true;
    }
    ~DecodeGraphBranchScope() {
        setting = previous;
    }
};

struct DecodeGraphTpProjectionScope {
    bool& setting;
    bool previous;

    explicit DecodeGraphTpProjectionScope(bool& value)
        : setting(value), previous(value) {
        setting = true;
    }
    ~DecodeGraphTpProjectionScope() {
        setting = previous;
    }
};

namespace mfq::cuda::internal {
void prepare_sampling_runtime(int64_t vocabulary);
}
