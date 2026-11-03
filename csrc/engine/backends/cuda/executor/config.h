#pragma once

#include "engine/weights/encoding/format_features.h"
#include "engine/weights/loader.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/ffn/gated.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

// Read once at owner construction; controls read on access stay in options.h.
struct CudaExecutionConfig {
    mfq::cuda::weight_loader::Loader::Options weights;
    bool report_cuda_memory = false;
    bool model_parallel_cuda_graph = true;
    bool tensor_parallel_cuda_graph = true;
    bool expert_parallel_cuda_graph = true;
    bool tensor_parallel_shared_linear_attention_input = true;
    bool tensor_parallel_mirror_linear_attention_scalars = true;
    bool gemma4_fused_norms = true;
    bool dsv4_groupwise_output_a = true;
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
    bool moe_pipeline = false;
    bool moe_hybrid_cpu = false;
    bool moe_assert_resident = false;
    bool moe_ram_pcie = false;
    bool moe_direct_ram = true;
    // Diagnostics may hold primary residency constant across paired passes.
    bool moe_residency_adapt = true;
    bool moe_residency_warm = true;
    bool moe_residency_projection_heat = true;
    std::optional<double> moe_ram_pcie_fraction;
};

std::pair<CudaExecutionConfig, mfq::cuda::Executor::Options> load_cuda_execution_config();
