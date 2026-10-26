#include "env.h"
#include "backends/cuda/engine/executor/options.h"
#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/executor/tq.h"
#include "backends/cuda/engine/weights/moe/expert_cache.h"
#include "backends/cuda/engine/state/kv_offload.h"

#include "mfq_cuda_quant_ops.h"
#include "mfe_ffn_token_groups.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(TYLOINFER_NATIVE_CUDA_RUNTIME) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#endif

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace {

std::optional<std::string> environment(const char* name) {
    const char* value = mfq::cuda::env_lookup(name);
    if (value == nullptr || value[0] == '\0') return std::nullopt;
    return value;
}

bool environment_flag(const char* name, bool fallback) {
    const auto value = environment(name);
    if (!value) return fallback;
    int parsed = 0;
    const auto [end, error] = std::from_chars(
        value->data(), value->data() + value->size(), parsed);
    if (error != std::errc{} || end != value->data() + value->size()) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return parsed != 0;
}

int environment_int(
        const char* name, int fallback, int minimum, int maximum) {
    const auto value = environment(name);
    if (!value) return fallback;
    int parsed = 0;
    const auto [end, error] = std::from_chars(
        value->data(), value->data() + value->size(), parsed);
    if (error != std::errc{} || end != value->data() + value->size() ||
            parsed < minimum || parsed > maximum) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return parsed;
}

std::size_t environment_size(
        const char* name, std::size_t fallback) {
    const auto value = environment(name);
    if (!value) return fallback;
    std::size_t parsed = 0;
    const auto [end, error] = std::from_chars(
        value->data(), value->data() + value->size(), parsed);
    if (error != std::errc{} || end != value->data() + value->size()) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return parsed;
}

bool enabled_unless_disabled(const char* name) {
    return !environment_flag(name, false);
}

} // namespace

namespace mfq::cuda::runtime_options {

MoeOptions moe(MoeOptions options) {
    options.row_workspace = nint_route_row_workspace();
    options.direct_pool = nint_direct_pool();
    options.group_dot = nint_group_dot();
    options.route_hint = nint_route_hint();
    options.route_warps = nint_route_warps();
    return options;
}

bool kv_vram_preallocated() {
    const auto* value = mfq::cuda::env_lookup("TYLOINFER_KV_VRAM_PREALLOCATED");
    return value && value[0] == '1';
}
bool vision_resident() {
    const auto* value = mfq::cuda::env_lookup("TYLOINFER_CUDA_VISION_RESIDENT");
    return value && value[0] == '1';
}
namespace {
bool exact_one(const char* name, bool fallback = false) {
    const auto* value = mfq::cuda::env_lookup(name);
    return value ? std::string_view(value) == "1" : fallback;
}
bool first_nonzero(const char* name, bool fallback) {
    const auto* value = mfq::cuda::env_lookup(name);
    return value ? value[0] != '0' : fallback;
}
}
bool resident_plan_overlap() {
    const auto* value = mfq::cuda::env_lookup("TYLOINFER_MFE_RESIDENT_PLAN_OVERLAP");
    if(value)return std::atoi(value)==1;
    // The GPU-only RAM/VRAM path has exact ordinary-generation coverage. Its
    // small early-window gain is retained; steady-tail gain is not established.
    // Keep mixed CPU scheduling explicit until it has the same coverage.
    return exact_one("TYLOINFER_MOE_RAM_PCIE_FRACTION");
}
std::optional<bool> rotary_fusion() {
    const auto* value = mfq::cuda::env_lookup("TYLOINFER_ROTARY_FUSED");
    return value ? std::optional<bool>(std::string_view(value) == "1") : std::nullopt;
}
bool mhc_timings() { return exact_one("TYLOINFER_TRACE_MHC_TIMINGS"); }
bool layer_timings() {
    const auto* value = mfq::cuda::env_lookup("TYLOINFER_TRACE_LAYER_TIMINGS");
    return value && std::string_view(value) != "0";
}
int route_poll_interval_us() {
    return environment_int("TYLOINFER_MOE_ROUTE_POLL_US", 2000, 0, 1'000'000);
}
bool single_window_flush() { return exact_one("TYLOINFER_MOE_SINGLE_WINDOW_FLUSH",true); }
bool early_no_cpu_ready() { return exact_one("TYLOINFER_MOE_EARLY_NO_CPU_READY",true); }
bool skip_gpu_only_cpu_wait() { return exact_one("TYLOINFER_MOE_SKIP_GPU_ONLY_CPU_WAIT",true); }
bool skip_fixed_gpu_timing() { return exact_one("TYLOINFER_MOE_SKIP_FIXED_GPU_TIMING",true); }
bool window_input_views() { return exact_one("TYLOINFER_MFE_WINDOW_INPUT_VIEWS",true); }
bool decode_host_pin() { return exact_one("TYLOINFER_MOE_DECODE_HOST_PIN"); }
bool dma_graph_batch() { return exact_one("TYLOINFER_MOE_DMA_GRAPH_BATCH"); }
bool warp_multi_sum() { return exact_one("TYLOINFER_CUDA_WARP_MULTI_SUM"); }
float residency_heat_retention() {
    const auto* setting=mfq::cuda::env_lookup("TYLOINFER_MOE_RESIDENCY_HEAT_RETENTION");
    // A four-token adaptation interval needs to remember experts reused every
    // dozen tokens. With 0.7 decay their score never reaches the admission
    // threshold; 0.9 retains that evidence across roughly 26 tokens. Prefill
    // still replaces the heat seed when a new request changes the working set.
    if(!setting)return 0.9f;
    char* end=nullptr;const float value=std::strtof(setting,&end);
    if(end==setting || *end || !std::isfinite(value) || value<0.0f || value>1.0f)
        throw std::invalid_argument("TYLOINFER_MOE_RESIDENCY_HEAT_RETENTION must be between zero and one");
    return value;
}
int residency_interval() {
    return environment_int("TYLOINFER_MOE_RESIDENCY_INTERVAL",1,1,1024);
}
bool router_lookahead() { return exact_one("TYLOINFER_MOE_ROUTER_LOOKAHEAD"); }
std::optional<bool> router_lookahead_override() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_ROUTER_LOOKAHEAD");
    return value?std::optional<bool>(std::string_view(value)=="1"):std::nullopt;
}
bool batch_attention_projections() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_BATCH_ATTENTION_PROJECTIONS");
    return !value || value[0]!='0';
}
bool gdn_independent_batch() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_GDN_INDEPENDENT_BATCH");
    return !value || value[0]!='0';
}
bool gdn_independent_batch_trace() { return exact_one("TYLOINFER_TRACE_GDN_BATCH"); }
bool batch_packed_prefill() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_BATCH_PACKED_PREFILL");
    return !value || value[0]!='0';
}
int batch_packed_prefill_token_budget() {
    return environment_int("TYLOINFER_BATCH_PACKED_PREFILL_TOKEN_BUDGET",512,16,8192);
}
bool nint_compressed_decode_batch() {
    // Keep the faster GEMM dispatch while tuning the online Tensor Core path.
    return exact_one("TYLOINFER_NINT_COMPRESSED_DECODE_BATCH");
}
bool nint_compressed_large_batch(bool vocabulary) {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_NINT_COMPRESSED_LARGE_BATCH");
    if(!value || std::strcmp(value,"head")==0)return vocabulary;
    return value[0]!='0';
}
bool nint_large_output_tiling() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_NINT_LARGE_OUTPUT_TILING");
    return !value || value[0]!='0';
}
int continuous_batch_kv_initial_capacity() {
    return environment_int("TYLOINFER_CONTINUOUS_BATCH_KV_INITIAL_CAPACITY",256,16,4096);
}
std::size_t workspace_reserve_bytes() {
    std::size_t reserve = std::size_t(4) * 1024 * 1024 * 1024;
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_CUDA_WORKSPACE_RESERVE_GIB")) {
        char* end = nullptr;
        const double gib = std::strtod(value, &end);
        const long double bytes = static_cast<long double>(gib) * 1024 * 1024 * 1024;
        if (end == value || *end || !std::isfinite(gib) || gib < 0 ||
                bytes > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("TYLOINFER_CUDA_WORKSPACE_RESERVE_GIB must be finite and non-negative");
        reserve = static_cast<std::size_t>(bytes);
    }
    return reserve;
}
std::size_t resident_budget_bytes() {
    const auto value=environment("TYLOINFER_SERVER_RESIDENT_BUDGET_BYTES");
    if(!value)return 0;
    std::size_t bytes=0;
    const auto [end,error]=std::from_chars(value->data(),value->data()+value->size(),bytes);
    if(error!=std::errc{} || end!=value->data()+value->size())
        throw std::invalid_argument("TYLOINFER_SERVER_RESIDENT_BUDGET_BYTES must be a non-negative integer");
    return bytes;
}
bool phased_transfer() { return exact_one("TYLOINFER_MOE_FFN_TRANSFER_PHASES", true); }
bool ram_field_packing() { return exact_one("TYLOINFER_MOE_RAM_FIELD_PACKING", true); }
bool mapped_copy_overlap() { return exact_one("TYLOINFER_MOE_MAPPED_COPY_OVERLAP"); }
bool background_calibration() { return exact_one("TYLOINFER_MOE_CPU_BACKGROUND_CALIBRATION", true); }
bool early_gate_up() { return first_nonzero("TYLOINFER_MFE_EARLY_GU", true); }
bool parallel_gate_up() { return first_nonzero("TYLOINFER_MFE_PARALLEL_GU", true); }
bool cpu_transfer_budget() { return first_nonzero("TYLOINFER_MOE_CPU_TRANSFER_BUDGET", true); }
bool transfer_cache() { return first_nonzero("TYLOINFER_MOE_TRANSFER_CACHE", false); }
bool mapped_copy() { return first_nonzero("TYLOINFER_MOE_MAPPED_COPY", false); }
MoeDiagnostics moe_diagnostics() {
    MoeDiagnostics result;
    result.cpu_rows = exact_one("TYLOINFER_TRACE_CPU_ROWS");
    result.cpu_rows_fail_alloc = exact_one("TYLOINFER_TRACE_CPU_ROWS_FAIL_ALLOC");
    result.dma = exact_one("TYLOINFER_TRACE_MOE_DMA");
    result.dma_fail_alloc = exact_one("TYLOINFER_TRACE_MOE_DMA_FAIL_ALLOC");
    result.shared_cpu_cost = exact_one("TYLOINFER_MOE_SHARED_CPU_COST");
    if (const auto* path = mfq::cuda::env_lookup("TYLOINFER_MOE_DISPATCH_RECORD")) result.dispatch_record = path;
    if (const auto* path = mfq::cuda::env_lookup("TYLOINFER_MOE_MISS_RECORD")) result.miss_record = path;
    if (const auto* path = mfq::cuda::env_lookup("TYLOINFER_MOE_DISPATCH_REPLAY")) result.dispatch_replay = path;
    return result;
}
std::optional<int> embedding_mapped() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_EMBEDDING_MAPPED");
    return value ? std::optional<int>(value[0] - '0') : std::nullopt;
}
std::optional<bool> gdn_inplace_state() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_GDN_INPLACE_STATE");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<bool> gdn_prefill_fused() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_GDN_PREFILL_FUSED");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> gr_dense_vector() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_GR_DENSE_VECTOR");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<bool> mfe_group_dot() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MFE_GROUP_DOT");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<bool> mfe_nint_late_scale() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MFE_NINT_LATE_SCALE");
    return value ? std::optional<bool>(std::string_view(value) == "1") : std::nullopt;
}
std::optional<bool> mfe_nvq_word_signs() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MFE_NVQ_WORD_SIGNS");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<bool> mfe_shared_dense_math() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MFE_SHARED_DENSE_MATH");
    return value ? std::optional<bool>(std::string_view(value) == "1") : std::nullopt;
}
std::optional<bool> moe_assert_resident() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_ASSERT_RESIDENT");
    return value ? std::optional<bool>(std::string_view(value) == "1") : std::nullopt;
}
std::optional<bool> moe_nvq_dense() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_NVQ_DENSE");
    return value ? std::optional<bool>(std::string_view(value) == "1") : std::nullopt;
}
std::optional<bool> moe_plan_reuse() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PLAN_REUSE");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_plan_reuse_check() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PLAN_REUSE_CHECK");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_compact_mma() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_COMPACT_MMA");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_direct_ram() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_DIRECT_RAM");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_expert_batch() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_EXPERT_BATCH");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<std::int64_t> moe_prefill_expert_batch_tokens() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_EXPERT_BATCH_TOKENS");
    return value ? std::optional<std::int64_t>(std::atoll(value)) : std::nullopt;
}
std::optional<bool> moe_prefill_graph() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_GRAPH");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_groups() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_GROUPS");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_layer() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_LAYER");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_layer_async() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_LAYER_ASYNC");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_layer_flush() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_LAYER_FLUSH");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_layer_major() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_LAYER_MAJOR");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_layer_overlap() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_LAYER_OVERLAP");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_layer_phased() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_LAYER_PHASED");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_layer_reuse_async() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_LAYER_REUSE_ASYNC");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_prefill_pool_retain() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_POOL_RETAIN");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> moe_preload_all() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PRELOAD_ALL");
    return value ? std::optional<bool>(std::string_view(value) == "1") : std::nullopt;
}
bool nint_dense_reference() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_NINT_DENSE_REFERENCE");
    return value != nullptr;
}
std::optional<bool> nint_direct_pool() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_NINT_DIRECT_POOL");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<bool> nint_group_dot() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_NINT_GROUP_DOT");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<bool> nint_route_hint() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_NINT_ROUTE_HINT");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<bool> nint_route_row_workspace() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_NINT_ROUTE_ROW_WORKSPACE");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<int> nint_route_warps() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_NINT_ROUTE_WARPS");
    return value ? std::optional<int>(std::atoi(value)) : std::nullopt;
}
std::optional<bool> nvq1_group_records() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_NVQ1_GROUP_RECORDS");
    return value ? std::optional<bool>(value[0] == '1') : std::nullopt;
}
std::optional<bool> nvq1_integer_delta() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_NVQ1_INTEGER_DELTA");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
bool nvq_dense_reference() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_NVQ_DENSE_REFERENCE");
    return value != nullptr;
}
std::optional<bool> ple_file_map() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_PLE_FILE_MAP");
    return value ? std::optional<bool>(value[0] == '1') : std::nullopt;
}
std::optional<bool> prefill_expert_batch_audit() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_PREFILL_EXPERT_BATCH_AUDIT");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> prefill_layer_major_trace() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_PREFILL_LAYER_MAJOR_TRACE");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> qsa_prefill_fused() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_QSA_PREFILL_FUSED");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}

QsaKvOffloadOptions qsa_kv_offload() {
    return {environment_size("TYLOINFER_QSA_KV_BUDGET_BYTES", 0),
        environment_size("TYLOINFER_QSA_KV_RAM_BUDGET_BYTES", 4ULL << 30),
        environment("TYLOINFER_RUNTIME_PREFIX_CACHE_DIR").value_or("")};
}
std::optional<bool> qsa_select_fused() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_QSA_SELECT_FUSED");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<bool> qsa_sparse_gate_fused() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_QSA_SPARSE_GATE_FUSED");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<bool> shared_gate_fused() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_SHARED_GATE_FUSED");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<bool> trace_mfe_dispatch() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_TRACE_MFE_DISPATCH");
    return value ? std::optional<bool>(std::atoi(value) != 0) : std::nullopt;
}
std::optional<bool> trace_mfe_dispatch_requested() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_TRACE_MFE_DISPATCH");
    return value ? std::optional<bool>(value[0] != '0') : std::nullopt;
}
std::optional<bool> trace_moe_serve() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_TRACE_MOE_SERVE");
    return value ? std::optional<bool>(value[0] != '\0' && value[0] != '0') : std::nullopt;
}
std::optional<bool> trace_ple_timings() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_TRACE_PLE_TIMINGS");
    return value ? std::optional<bool>(value[0] == '1') : std::nullopt;
}
int moe_prefill_group_size() {
    const auto* value=mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_GROUP_SIZE");
    return value ? std::atoi(value) : 16;
}
} // namespace mfq::cuda::runtime_options

CudaExecutionConfig load_cuda_execution_config() {
    CudaExecutionConfig result;
    result.profile_filter = environment("TYLOINFER_PROFILE_CUDA_FILTER").value_or("");
    result.report_cuda_memory = environment_flag("TYLOINFER_REPORT_CUDA_MEMORY", false);

    const auto parallel_graph = environment("TYLOINFER_MODEL_PARALLEL_CUDA_GRAPH");
    if (parallel_graph) {
        result.model_parallel_cuda_graph = environment_flag(
            "TYLOINFER_MODEL_PARALLEL_CUDA_GRAPH", true);
        result.tensor_parallel_cuda_graph = result.model_parallel_cuda_graph;
        result.expert_parallel_cuda_graph = result.model_parallel_cuda_graph;
    } else {
        result.tensor_parallel_cuda_graph = environment_flag(
            "TYLOINFER_TP_CUDA_GRAPH", true);
        result.expert_parallel_cuda_graph = environment_flag(
            "TYLOINFER_EP_CUDA_GRAPH", true);
    }
    result.tasks.projection.grouped = environment_flag(
        "TYLOINFER_TP_GROUPED_PROJECTIONS", true);
    result.tensor_parallel_shared_linear_attention_input = environment_flag(
        "TYLOINFER_TP_SHARED_LINEAR_ATTENTION_INPUT", true);
    result.tensor_parallel_mirror_linear_attention_scalars = environment_flag(
        "TYLOINFER_TP_MIRROR_LINEAR_ATTENTION_SCALARS", true);
    result.tensor_parallel_mirror_qwen35_attention_kv = environment_flag(
        "TYLOINFER_TP_MIRROR_QWEN35_ATTENTION_KV", true);
    result.tasks.projection.reduce_to_primary = environment(
        "TYLOINFER_MODEL_PARALLEL_REDUCE_TO_PRIMARY")
        ? environment_flag("TYLOINFER_MODEL_PARALLEL_REDUCE_TO_PRIMARY", true)
        : environment_flag("TYLOINFER_TP_REDUCE_TO_PRIMARY", true);
    result.tasks.projection.fp16_reduce = environment(
        "TYLOINFER_MODEL_PARALLEL_FP16_REDUCE")
        ? environment_flag("TYLOINFER_MODEL_PARALLEL_FP16_REDUCE", true)
        : environment_flag("TYLOINFER_TP_FP16_REDUCE", true);
    result.tasks.projection.peer_first = environment(
        "TYLOINFER_MODEL_PARALLEL_PEER_FIRST_LAUNCH")
        ? environment_flag("TYLOINFER_MODEL_PARALLEL_PEER_FIRST_LAUNCH", true)
        : environment_flag("TYLOINFER_TP_PEER_FIRST_LAUNCH", true);

    result.tasks.projection.parallel = enabled_unless_disabled(
        "TYLOINFER_DISABLE_DECODE_BRANCH_PARALLEL");
    result.tensor_parallel_shard_native_float = environment_flag(
        "TYLOINFER_TP_SHARD_NATIVE_FLOAT", false);
    result.tasks.projection.nvq_fusion = environment("TYLOINFER_DISABLE_NVQ_FUSION")
        ? enabled_unless_disabled("TYLOINFER_DISABLE_NVQ_FUSION")
        : enabled_unless_disabled("TYLOINFER_DISABLE_NIQ_FUSION");
    result.nvq2_exec = environment_flag("TYLOINFER_NVQ2_EXEC", false) &&
        (environment("TYLOINFER_DISABLE_NVQ2_EXEC")
            ? enabled_unless_disabled("TYLOINFER_DISABLE_NVQ2_EXEC")
            : enabled_unless_disabled("TYLOINFER_DISABLE_NIQ2_EXEC"));
    result.nvq_extended_group_exec = environment_flag(
        "TYLOINFER_NVQ_EXTENDED_GROUP_EXEC", false);
    result.tasks.projection.moe.heterogeneous = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_NVQ_HETERO");
    result.tasks.projection.moe.nvq_decode =
        enabled_unless_disabled("TYLOINFER_DISABLE_MOE_NVQ_HETERO_DECODE") &&
        !environment_flag("TYLOINFER_NVQ_MOE_EXACT_REDUCTION", false) &&
        !environment("TYLOINFER_NVQ_MOE_ROWS_PER_BLOCK") &&
        !environment("TYLOINFER_NVQ_MOE_WARPS") &&
        !environment("TYLOINFER_NVQ_MOE_SHARE_GROUP_STATE");
    result.tasks.projection.moe.active_decode = environment_flag("TYLOINFER_MOE_NVQ_ACTIVE_DECODE",false);
    result.tasks.projection.moe.prefill = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_PREFILL_MMA");
    result.tasks.projection.moe.min_tokens = environment_int(
        "TYLOINFER_MOE_PREFILL_MMA_MIN_TOKENS", 9, 9,
        std::numeric_limits<int>::max());
    result.moe_small_heterogeneous = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_SMALL_HETERO");
    result.moe_delayed_route_readback = environment_flag(
        "TYLOINFER_MOE_DELAYED_ROUTE_READBACK", true);
    result.moe_projection_bundle_prefetch = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_PROJECTION_BUNDLE_PREFETCH");
    result.moe_ffn.reuse_input = enabled_unless_disabled(
        "TYLOINFER_DISABLE_SPLIT_MOE_ACTIVATION_REUSE");
    result.moe_ffn.resident_dispatch = environment_flag(
        "TYLOINFER_MOE_GPU_RESIDENT_DISPATCH", false);
    result.moe_ffn.fused_activation = environment_flag(
        "TYLOINFER_MOE_FFN_FUSED_ACTIVATION", true);
    result.moe_ffn.two_stage = environment_flag("TYLOINFER_MOE_TWO_STAGE_FFN", false);
    result.moe_swiglu_quant_fusion = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_SWIGLU_QUANT_FUSION");
    result.moe_reduce_gate_fusion = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_REDUCE_GATE_FUSION");
    result.tasks.dense.mlp.gelu_fusion = enabled_unless_disabled(
        "TYLOINFER_DISABLE_FFN_GEGLU_FUSION");
    result.tasks.dense.mlp.silu_fusion = enabled_unless_disabled(
        "TYLOINFER_DISABLE_FFN_SWIGLU_FUSION");
    result.tasks.dense.important_parallel = enabled_unless_disabled(
        "TYLOINFER_DISABLE_IN_BRANCH_PARALLEL");
    result.gemma4_fused_norms = environment_flag(
        "TYLOINFER_GEMMA4_FUSED_NORMS", true);
    result.dsv4_groupwise_output_a = environment_flag(
        "TYLOINFER_DSV4_GROUPWISE_OUTPUT_A", true);
    result.diagnostic_nint_group = enabled_unless_disabled(
        "TYLOINFER_DIAGNOSTIC_DISABLE_NINT_GROUP");
    result.tasks.dense.fp32_residual = environment_flag(
        "TYLOINFER_DIAGNOSTIC_FP32_RESIDUAL", false);
    result.tasks.dense.important_fp32 = environment_flag(
        "TYLOINFER_DIAGNOSTIC_IN_F32_DOWN", false);
    result.tasks.attention.write_aten = environment_flag(
        "TYLOINFER_KV_CACHE_WRITE_ATEN", false);
    result.gdn_transposed_state = environment_flag(
        "TYLOINFER_GDN_TRANSPOSED_STATE", true);
    result.gdn_fused_output = environment_flag("TYLOINFER_GDN_FUSED_OUTPUT",true);
    result.linear_conv_prefill_fused = environment_flag(
        "TYLOINFER_LINEAR_CONV_PREFILL_FUSED", true);
    result.tasks.attention.fused_rope = environment_flag(
        "TYLOINFER_MINICPM_FUSED_BF16_ROPE", true);
    result.tasks.attention.fused_qk_rope_kv = environment_flag(
        "TYLOINFER_MINICPM_FUSED_QK_NORM_ROPE_KV", true);
    result.tasks.attention.fused_norm = environment_flag(
        "TYLOINFER_MINICPM_FUSED_BF16_RMSNORM", true);
    result.tasks.attention.fused_rope_kv = environment_flag(
        "TYLOINFER_MINICPM_FUSED_ROPE_KV", true);
    result.tasks.attention.kernels.bf16_flash128 = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MINICPM_BF16_FLASH128");
    result.tasks.attention.kernels.flash128_casts = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MINICPM_FLASH128_SPECIALIZED_CASTS");
    result.tasks.attention.kernels.bf16_gqa_decode = environment_flag(
        "TYLOINFER_MINICPM_BF16_GQA_DECODE", true);
    result.tasks.dense.bf16_swiglu = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MINICPM_BF16_SWIGLU_FUSION");
    result.tasks.dense.bf16_residual = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MINICPM_BF16_RESIDUAL_ACC");
    result.tasks.attention.kernels.mma_prefill = environment_flag("TYLOINFER_MMA_ATTENTION", true);
    result.tasks.attention.kernels.mma_decode = environment_flag(
        "TYLOINFER_MMA_ATTENTION_DECODE", true);
    result.tasks.attention.kernels.decode_aten = environment_flag(
        "TYLOINFER_ATTENTION_DECODE_ATEN", false);
    result.tasks.attention.kernels.split_k = environment_flag(
        "TYLOINFER_ATTENTION_DECODE_SPLITK", true);
    result.deepseek_v41_engram_cache_rows = environment_size(
        "TYLOINFER_DEEPSEEK_V41_ENGRAM_CACHE_ROWS", 16'384);
    result.moe_mapped_gather = environment_flag("TYLOINFER_MOE_MAPPED_GATHER", false);
    result.moe_mapped_copy_blocks = environment_int(
        "TYLOINFER_MOE_MAPPED_COPY_BLOCKS", 64, 4, 128);
    result.moe_ssd_io_workers = environment_int(
        "TYLOINFER_MOE_SSD_IO_WORKERS", 8, 1, 64);
    result.moe_ssd_overlap = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_SSD_OVERLAP");
    result.moe_ssd_ranges = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_SSD_RANGES");
    result.moe_host_cache_bytes = environment_size("TYLOINFER_MOE_HOST_CACHE_BYTES",0);
    result.moe_pipeline = environment_flag("TYLOINFER_MOE_PIPELINE",false);
    result.moe_hybrid_cpu = environment_flag("TYLOINFER_MOE_HYBRID_CPU",false);
    result.moe_preload_all = environment_flag("TYLOINFER_MOE_PRELOAD_ALL",false);
    result.moe_assert_resident = environment_flag("TYLOINFER_MOE_ASSERT_RESIDENT",false);
    result.moe_ram_pcie = environment_flag("TYLOINFER_MOE_RAM_PCIE",false);
    result.moe_direct_ram = environment_flag("TYLOINFER_MOE_DIRECT_RAM",true);
    result.moe_residency_adapt = environment_flag("TYLOINFER_MOE_RESIDENCY_ADAPT",true);
    result.moe_residency_warm = environment_flag("TYLOINFER_MOE_RESIDENCY_WARM",true);
    result.moe_residency_projection_heat = environment_flag("TYLOINFER_MOE_RESIDENCY_PROJECTION_HEAT",true);
    result.moe_host_physical_bytes = environment_size("TYLOINFER_MOE_HOST_PHYSICAL_BYTES",0);
    result.residual.fused_projections = environment_flag("TYLOINFER_GR_FUSED_PROJECTIONS",true);
    result.residual.fused_activation = environment_flag("TYLOINFER_GR_FUSED_PROJECTION_ACTIVATION",true);
    result.residual.fused_post_norm = environment_flag("TYLOINFER_GR_FUSED_POST_NORM",true);
    result.gdn_fused_preparation = environment_flag("TYLOINFER_GDN_FUSED_PREPARATION",true);
    result.gdn_fused_core = environment_flag("TYLOINFER_GDN_FUSED_CORE",true);
    result.residual.prepared_dense = environment_flag("TYLOINFER_GR_PREPARED_DENSE_PROJECTION",false);
    result.residual.native_input = environment_flag("TYLOINFER_GR_NATIVE_PROJECTION_INPUT",true);
    result.residual.fixed_group = environment_flag("TYLOINFER_GR_FIXED_GROUP_PROJECTION",true);
    result.residual.compact_mix = environment_flag("TYLOINFER_GR_COMPACT_MIX",true);
    result.residual.two_stage = environment_flag("TYLOINFER_GR_TWO_STAGE",true);
    result.residual.dense_injection = environment_flag("TYLOINFER_GR_TWO_STAGE_DENSE_INJECTION",true);
    result.tasks.projection.moe.nint_decode = environment_flag("TYLOINFER_MOE_NINT_HETEROGENEOUS_DECODE",true);
    if (const auto value = environment("TYLOINFER_MOE_RAM_PCIE_FRACTION")) {
        std::size_t end = 0;
        const auto fraction = std::stod(*value,&end);
        if (end != value->size() || !std::isfinite(fraction) || fraction < 0 || fraction > 1)
            throw std::invalid_argument("TYLOINFER_MOE_RAM_PCIE_FRACTION must be in [0,1]");
        result.moe_ram_pcie_fraction = fraction;
    }
    return result;
}

void mfq_release_host_allocator_cache() noexcept {
#if defined(__GLIBC__)
    (void)malloc_trim(0);
#endif
}

using mfq_tensor_backend::indexing::Slice;

namespace mfq::cuda::internal {

PrefillCudaTimer::PrefillCudaTimer()
    : stream_(mfq_get_current_cuda_stream()) {
    TYLOINFER_CUDA_CHECK(cudaEventCreate(&started_));
    try {
        TYLOINFER_CUDA_CHECK(cudaEventCreate(&finished_));
        TYLOINFER_CUDA_CHECK(cudaEventRecord(started_, stream_));
    } catch (...) {
        if (finished_ != nullptr) cudaEventDestroy(finished_);
        cudaEventDestroy(started_);
        finished_ = nullptr;
        started_ = nullptr;
        throw;
    }
}

PrefillCudaTimer::~PrefillCudaTimer() {
    if (finished_ != nullptr) cudaEventDestroy(finished_);
    if (started_ != nullptr) cudaEventDestroy(started_);
}

cudaEvent_t PrefillCudaTimer::finished_event() const {
    return finished_;
}

double PrefillCudaTimer::elapsed_ms() const {
    TYLOINFER_CUDA_CHECK(cudaEventSynchronize(finished_));
    float elapsed = 0.0f;
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed, started_, finished_));
    return static_cast<double>(elapsed);
}

} // namespace mfq::cuda::internal

void LayerPlacementConfig::prepare(int64_t layers) {
    layer_devices.assign(static_cast<size_t>(layers), primary_device());
    if (!enabled()) return;
    const auto slices = mfq::plan_tensor_parallel_slices(
        layers, 1, devices, split);
    for (const auto& slice : slices) {
        for (int64_t layer = slice.begin; layer < slice.end; ++layer) {
            layer_devices.at(static_cast<size_t>(layer)) = slice.device;
        }
    }
}

int LayerPlacementConfig::device_for_layer(int64_t layer) const {
    if (layer < 0 || layer >= static_cast<int64_t>(layer_devices.size())) {
        throw std::runtime_error("layer-placement index is outside the model");
    }
    return layer_devices.at(static_cast<size_t>(layer));
}

CudaExecutionContext::CudaExecutionContext(CudaStores& borrowed)
    : config(load_cuda_execution_config()), stores(borrowed),
      routes(environment("TYLOINFER_MOE_ROUTE_STATS").value_or(""),
             environment_flag("TYLOINFER_MOE_ROUTE_OUTPUT_ENERGY", false)) {
    profiler.filter = config.profile_filter;
    tq = std::make_shared<TqExec>();
}

mfq::cuda::Projector CudaExecutionContext::projector() {
    auto options = config.tasks.projection;
    options.primary = model_parallel_primary_device(*this);
    options.nint_group_quantize = !decode_graph_serial_branches ||
        mfq::cuda::runtime_options::nint_group_quantize().value_or(true);
    options.moe = mfq::cuda::runtime_options::moe(options.moe);
    return {profiler, kl_mmq, *tq, branches, collectives,
            decode_graph_serial_branches, decode_graph_tp_projection_major, options};
}

CudaExecutionContext::~CudaExecutionContext() {
    try { drain(); }
    catch (const std::exception& error) {
        std::cerr << "runtime_execution_cleanup error=" << error.what() << '\n';
    }
    // Release pipelines while their borrowed branch streams still exist.
    moe_pipelines.clear();
    attention_scratch.clear();
}

void CudaExecutionContext::register_moe_pipeline(std::shared_ptr<MoeFfnPipeline> pipeline) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if(!pipeline)throw std::invalid_argument("cannot register an empty MoE FFN pipeline");
    moe_pipelines.push_back(pipeline);
    try { activate_moe_ffn_pipeline(std::move(pipeline)); }
    catch(...) { moe_pipelines.pop_back(); throw; }
#else
    throw std::invalid_argument("MoE FFN pipeline support is disabled");
#endif
}

void CudaExecutionContext::drain() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (native_contexts.empty() && !stores.moe_expert_cache &&
        !stores.tq_expert_cache && !stores.qsa_kv_store &&
        (!tq || tq->empty()) && moe_pipelines.empty() && branches.empty() &&
        attention_scratch.empty() && collectives.streams().empty()) return;
#endif
    CudaScope scope(*this);
    std::exception_ptr failure;
    const auto finish = [&](auto&& work) {
        try { work(); } catch (...) { if (!failure) failure = std::current_exception(); }
    };
    finish([&] { finish_moe_expert_exchanges(stores.moe_expert_cache); });
#if TYLOINFER_ENABLE_MFQ_COMPAT
    for (const auto& pipeline : moe_pipelines)
        finish([&] { drain_moe_ffn_pipeline(pipeline); });
#endif
    finish([&] { collectives.drain(); });
    if (stores.qsa_kv_store) finish([&] { stores.qsa_kv_store->flush(); });
    // shortcut: retirement fences the whole device; use events when every consumer has a completion handle.
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    for (const auto& [device, context] : native_contexts) finish([&] {
        MfqCudaGuard guard(device);
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
    });
#else
    std::unordered_set<int> devices{layer_placement.primary_device()};
    for (const auto* group : {&tensor_parallel.devices, &expert_parallel.devices, &layer_placement.devices})
        devices.insert(group->begin(), group->end());
    for (int device : devices) finish([&] {
        MfqCudaGuard guard(device);
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
    });
#endif
    if (tq) finish([&] { tq->drain(); });
    finish([&] { branches.drain(); });
    if (failure) std::rethrow_exception(failure);
}

void CudaExecutionContext::reset() {
    drain();
    moe_pipelines.clear();
    attention_scratch.clear();
    tq = std::make_shared<TqExec>();
    branches.clear();
    profiler.reset();
    collectives.reset();
    tensor_parallel = {};
    expert_parallel = {};
    layer_placement = {};
    dsv4_cpu_offload_layers.clear();
    dsv4_cpu_offload_host_bytes = 0;
    n_gpu_layers = -1;
    dense_cpu_layer_count = 0;
    loading_cpu_layer = false;
    drop_file_cache = false;
    decode_graph_serial_branches = false;
    decode_graph_tp_projection_major = false;
    continuous_batch_cache_serial = false;
    routes.clear();
    kl_mmq.reset();
    force_moe_unfused_reduce = false;
    force_moe_materialized_swiglu = false;
    moe_cache_registration_min_slots = 8;
    gemma_trace_layer = -1;
    gemma_stage_trace = nullptr;
}

bool model_parallel_enabled(const CudaExecutionContext& execution) {
    return execution.tensor_parallel.enabled() ||
        execution.expert_parallel.enabled();
}

const ParallelConfig & model_parallel_config(
        const CudaExecutionContext& execution) {
    return execution.tensor_parallel.enabled()
        ? execution.tensor_parallel
        : execution.expert_parallel;
}

const ParallelConfig & moe_parallel_config(
        const CudaExecutionContext& execution) {
    return execution.expert_parallel.enabled()
        ? execution.expert_parallel
        : execution.tensor_parallel;
}

int model_parallel_primary_device(const CudaExecutionContext& execution) {
    if (!execution.tensor_parallel.devices.empty()) {
        return execution.tensor_parallel.primary_device();
    }
    if (!execution.expert_parallel.devices.empty()) {
        return execution.expert_parallel.primary_device();
    }
    return 0;
}

bool model_parallel_cuda_graph_enabled(
        const CudaExecutionContext& execution) {
    if(execution.stores.tq_expert_cache && moe_parallel_config(execution).enabled())return false;
    if (!execution.tensor_parallel.enabled() &&
            !execution.expert_parallel.enabled()) {
        return true;
    }
    const bool enabled = execution.expert_parallel.enabled() &&
            !execution.tensor_parallel.enabled()
        ? execution.config.expert_parallel_cuda_graph
        : execution.config.tensor_parallel_cuda_graph;
    return execution.collectives.enabled() && enabled;
}

int active_weight_load_device(const CudaExecutionContext& execution) {
    const auto& placement = execution.layer_placement;
    return placement.load_device >= 0
        ? placement.load_device
        : model_parallel_primary_device(execution);
}

void trace_gemma_stage(
        CudaExecutionContext& execution,
        int layer,
        const char * name,
        const mfq_tensor_backend::Tensor & value) {
    if (execution.gemma_stage_trace != nullptr &&
            layer == execution.gemma_trace_layer) {
        execution.gemma_stage_trace->emplace_back(
            name, value.to(mfq_tensor_backend::kFloat32).contiguous().clone());
    }
}

void report_cuda_memory(
        const CudaExecutionConfig& config,
        const char * stage) {
    if (!config.report_cuda_memory) return;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const auto stats = mfq_cuda_memory_stats(mfq_current_cuda_device());
    constexpr double mib = 1024.0 * 1024.0;
    std::cout << "cuda_memory_stage=" << stage
              << " free_mib=" << free_bytes / mib
              << " used_mib=" << (total_bytes - free_bytes) / mib
              << " total_mib=" << total_bytes / mib
              << " allocated_mib=" << stats.allocated_bytes / mib
              << " active_mib=" << stats.active_bytes / mib
              << " reserved_mib=" << stats.reserved_bytes / mib
              << " inactive_split_mib="
              << stats.inactive_split_bytes / mib
              << " requested_mib=" << stats.requested_bytes / mib
              << " allocations=" << stats.allocations
              << " segments=" << stats.segments
              << " retries=" << stats.retries
              << " ooms=" << stats.ooms << "\n";
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    // cudaMemGetInfo does not describe this process's allocation ownership or
    // WDDM budget. Report those separately when diagnosing residency pressure.
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto native=context->memory_stats();
    const auto [pool_used,pool_reserved]=context->pool_memory_stats();
    std::cout<<"native_memory_stage="<<stage<<" owned_mib="<<native.allocated/mib
             <<" peak_mib="<<native.peak/mib<<" limit_mib="<<native.limit/mib
             <<" local_usage_mib="<<context->local_memory_usage()/mib
             <<" pool_used_mib="<<pool_used/mib<<" pool_reserved_mib="<<pool_reserved/mib<<"\n";
#ifdef _WIN32
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    cudaDeviceProp properties{};TYLOINFER_CUDA_CHECK(cudaGetDeviceProperties(&properties,mfq_current_cuda_device()));
    if(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))for(UINT index=0;;++index) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> item;
        if(factory->EnumAdapters1(index,&item)==DXGI_ERROR_NOT_FOUND)break;
        DXGI_ADAPTER_DESC1 description{};
        if(FAILED(item->GetDesc1(&description)) || std::memcmp(properties.luid,&description.AdapterLuid,8))continue;
        Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;
        if(SUCCEEDED(item.As(&adapter)))for(auto segment:{DXGI_MEMORY_SEGMENT_GROUP_LOCAL,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL}) {
            DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
            if(SUCCEEDED(adapter->QueryVideoMemoryInfo(0,segment,&memory)))
                std::cout<<"wddm_memory_stage="<<stage<<" segment="<<(segment==DXGI_MEMORY_SEGMENT_GROUP_LOCAL?"local":"nonlocal")
                         <<" budget_mib="<<memory.Budget/mib<<" usage_mib="<<memory.CurrentUsage/mib
                         <<" reservation_mib="<<memory.CurrentReservation/mib<<"\n";
        }
        break;
    }
#endif
#endif
}

namespace mfq::cuda::runtime_options {
std::optional<bool> mfe_token_groups() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MFE_TOKEN_GROUPS")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> mfe_token_compact_plan() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MFE_TOKEN_COMPACT_PLAN")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> mfe_token_aligned_gu() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MFE_TOKEN_ALIGNED_GU")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> mfe_shared_small_whole() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MFE_SHARED_SMALL_WHOLE")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> mfe_nvq_bank_map() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MFE_NVQ_BANK_MAP")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> nint_fused_fixed_bits() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_NINT_FUSED_FIXED_BITS")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> nint_group_quantize() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_NINT_GROUP_QUANTIZE")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> ple_native_fusion() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_PLE_NATIVE_FUSION")) return value[0] == '1';
    return std::nullopt;
}
std::optional<bool> gdn_strided_projections() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GDN_STRIDED_PROJECTIONS")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> gdn_short_inplace() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GDN_SHORT_INPLACE")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> gdn_short_fusion() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GDN_SHORT_FUSION")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> gdn_batch_projection() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GDN_BATCH_PROJECTION")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> nint_prefill_decode_reuse() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_NINT_PREFILL_DECODE_REUSE")) return value[0] == '1';
    return std::nullopt;
}
std::optional<bool> moe_prefill_append_tail() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_APPEND_TAIL")) return value[0] == '1';
    return std::nullopt;
}
std::optional<bool> gdn_prefill_layer_batch() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GDN_PREFILL_LAYER_BATCH")) return value[0] == '1';
    return std::nullopt;
}
std::optional<bool> gr_prefill_layer_batch() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GR_PREFILL_LAYER_BATCH")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> gr_chain_ffn_post() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GR_CHAIN_FFN_POST")) return value[0] == '1';
    return std::nullopt;
}
std::optional<bool> mtp_predictor_graph() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MTP_PREDICTOR_GRAPH")) return value[0] != '0';
    return std::nullopt;
}
std::optional<std::int64_t> continuous_ffn_tile() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_CONTINUOUS_FFN_TILE")) return std::atoll(value);
    return std::nullopt;
}
std::optional<bool> gdn_packed_qkv() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GDN_PACKED_QKV")) return value[0] != '0';
    return std::nullopt;
}
std::optional<int> moe_prefill_shared_chain_streams() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_SHARED_CHAIN_STREAMS")) return std::atoi(value);
    return std::nullopt;
}
std::optional<int> moe_prefill_shared_streams() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_SHARED_STREAMS")) return std::atoi(value);
    return std::nullopt;
}
std::optional<int> gdn_prefill_projection_streams() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GDN_PREFILL_PROJECTION_STREAMS")) return std::atoi(value);
    return std::nullopt;
}
std::optional<bool> trace_gr_dispatch() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_TRACE_GR_DISPATCH")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> gr_batch_two_stage() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GR_BATCH_TWO_STAGE")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> gr_continuous_two_stage() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GR_CONTINUOUS_TWO_STAGE")) return value[0] != '0';
    return std::nullopt;
}
std::optional<std::int64_t> gr_prefill_work_chunk() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GR_PREFILL_WORK_CHUNK")) return std::atoll(value);
    return std::nullopt;
}
std::optional<bool> position_buffer(const char* name) {
    if (const auto* value = mfq::cuda::env_lookup(name)) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> single_output_view() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_SINGLE_OUTPUT_VIEW")) return value[0] != '0';
    return std::nullopt;
}
std::optional<bool> moe_direct_cache_dma() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MOE_DIRECT_CACHE_DMA")) return value[0] == '1';
    return std::nullopt;
}
std::optional<bool> moe_compact_host_stage() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MOE_COMPACT_HOST_STAGE")) return value[0] == '1';
    return std::nullopt;
}
std::optional<bool> moe_prefill_host_maps() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MOE_PREFILL_HOST_MAPS")) return value[0] != '0';
    return std::nullopt;
}
std::optional<char> gr_up_group_coefficients() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GR_UP_GROUP_COEFFICIENTS")) return value[0];
    return std::nullopt;
}
std::optional<bool> gr_dense_bf16_direct() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_GR_DENSE_BF16_DIRECT")) return value[0] != '0';
    return std::nullopt;
}
std::optional<int> mfe_token_group_capacity() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MFE_TOKEN_GROUP_CAPACITY")) {
        const std::string_view mode(value);
        if (mode == "2") return 2;
        if (mode == "4") return 4;
        if (mode == "8") return 8;
        if (mode == "shared4") return mfq::cuda::kMfeSharedGroupEncoding + 4;
        if (mode == "auto") return 0;
    }
    return std::nullopt;
}
std::optional<std::int64_t> moe_ram_pin_bytes() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MOE_RAM_PIN_GIB")) {
        char* end=nullptr;
        const double gib=std::strtod(value,&end);
        if(end==value || *end || !std::isfinite(gib) || gib<0 || gib>double(INT64_MAX)/(1ull<<30))
            throw std::invalid_argument("TYLOINFER_MOE_RAM_PIN_GIB must be finite and non-negative");
        return gib>0?std::max<std::int64_t>(1,std::int64_t(gib*(1ull<<30))):0;
    }
    return std::nullopt;
}
std::optional<std::int64_t> moe_ram_pin_reserve_bytes() {
    if (const auto* value = mfq::cuda::env_lookup("TYLOINFER_MOE_RAM_PIN_RESERVE_GIB")) {
        char* end=nullptr;
        const double gib=std::strtod(value,&end);
        if(end==value || *end || !std::isfinite(gib) || gib<0 || gib>=double(INT64_MAX)/(1ull<<30))
            throw std::invalid_argument("TYLOINFER_MOE_RAM_PIN_RESERVE_GIB must be finite and non-negative");
        return std::int64_t(gib*(1ull<<30));
    }
    return std::nullopt;
}
} // namespace mfq::cuda::runtime_options
