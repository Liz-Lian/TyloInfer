#pragma once

// Read-only weights and configuration shared by one Qwen4 layer.

#include "engine/models/qwen4_exp/model.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops/ffn.h"
#else
#include "engine/models/qwen4_exp/moe.h"
#endif
#include "engine/models/qwen4_exp/config.h"
#include "engine/backends/cuda/ops/ffn/gated.h"

#include <cstdint>
#include <memory>

namespace tyloi::cuda::qwen4_exp {
struct LayerState;

struct Layer final {
    // Placement, as the layer loop sets it for CPU-offloaded layers.
    int cuda_device = 0;
    bool cpu_offloaded = false;
    tyloi::models::qwen4_exp::Config config;
    GatedResidual attention_gr, ffn_gr;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::FfnWeights ffn;
#else
    Moe ffn;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    std::unique_ptr<tyloi::ops::Gdn> gdn;
#else
    std::unique_ptr<Gdn> gdn;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    std::unique_ptr<tyloi::ops::Qsa> qsa;
#else
    std::unique_ptr<Qsa> qsa;
#endif
    std::unique_ptr<Ple> ple;
};

DenseTensor forward(const Layer& weights, LayerState& request,
                    DenseTensor x, const DenseTensor& ids,
                    const DenseTensor& positions, const DenseTensor& full_positions,
                    int64_t confirmed = 0, const DenseTensor& host_ids = {});

DenseTensor prefill_ple(const Layer& weights, LayerState& request,
                        DenseTensor x, const DenseTensor& ids, const DenseTensor& host_ids);

DenseTensor prefill_attention_core(const Layer& weights, LayerState& request,
                                   const DenseTensor& input, const DenseTensor& positions,
                                   const DenseTensor& full_positions);

GatedResidual::Result prefill_attention(const Layer& weights, LayerState& request,
                                        DenseTensor x, const DenseTensor& ids,
                                        const DenseTensor& host_ids, const DenseTensor& positions,
                                        const DenseTensor& full_positions);

} // namespace tyloi::cuda::qwen4_exp
