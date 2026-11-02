#include "options.h"

#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#include "engine/backends/cuda/kernels/mfe_ffn_token_groups.h"

#include <charconv>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

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
    return env_int("TYLOINFER_MOE_ROUTE_POLL_US", 2000, 0, 1'000'000);
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
    return env_int("TYLOINFER_MOE_RESIDENCY_INTERVAL",1,1,1024);
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
    return env_int("TYLOINFER_BATCH_PACKED_PREFILL_TOKEN_BUDGET",512,16,8192);
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
    return env_int("TYLOINFER_CONTINUOUS_BATCH_KV_INITIAL_CAPACITY",256,16,4096);
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
    const auto value=env_value("TYLOINFER_SERVER_RESIDENT_BUDGET_BYTES");
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
    return {mfq::cuda::env_size("TYLOINFER_QSA_KV_BUDGET_BYTES", 0),
        mfq::cuda::env_size("TYLOINFER_QSA_KV_RAM_BUDGET_BYTES", 4ULL << 30),
        env_value("TYLOINFER_RUNTIME_PREFIX_CACHE_DIR").value_or("")};
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
