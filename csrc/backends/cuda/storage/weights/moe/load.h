#pragma once

#include "moe.h"

namespace mfq { class ModelSource; }

struct MfeCpuPool {
    std::vector<int32_t> expert_ids;
    std::string dtype;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> runtime_payload;
    NintCpu weight;
    Nint8ZeroCpu q8_zero;
    Mxfp4Cpu mxfp4;
};

struct MfeCpu {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<MfeCpuPool> pools;
};

MfeCpu unpack_mfe(const std::vector<uint8_t>& blob);
MfeCpu load_mfe_cpu(
    const mfq::ModelSource& source, const std::string& name);
MfeWeight to_gpu_mfe(const MfeCpu& source);
std::shared_ptr<MixedMoeRuntime> make_mixed_moe_runtime(
    const MfeCpu& source,
    bool cuda,
    const CudaExecutionConfig& config = {});
int64_t mixed_moe_storage_bytes(const MixedMoeRuntime& runtime);
mfq_tensor_backend::Tensor copy_cpu_weight_to_cuda(
    const mfq_tensor_backend::Tensor& source);
MfeWeight to_gpu_mixed_moe(
    const MfeCpu& source,
    const CudaExecutionConfig& config = {});
MfeWeight to_cuda_device_moe_expert_slice(
    const MfeCpu& source,
    int64_t expert_begin,
    int64_t expert_end,
    int device,
    const CudaExecutionConfig& config = {});
MfeWeight cpu_mixed_moe_metadata(
    const std::shared_ptr<MixedMoeRuntime>& runtime);

MfeWeight stage_cpu_mixed_moe(
    const std::shared_ptr<MixedMoeRuntime>& runtime,
    const CudaExecutionConfig& config = {});
