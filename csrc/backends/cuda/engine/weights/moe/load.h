#pragma once

#include "moe.h"
#include "engine/weights/moe/experts.h"
#include "engine/executor/execution.h"
#include "formats/quant/mfe_blob.h"

namespace mfq { class ModelSource; }

mfq::mfe::Data load_mfe_cpu(
    const mfq::ModelSource& source, const std::string& name);
mfq::cuda::Experts to_gpu_mfe(const mfq::mfe::Data& source);
std::shared_ptr<MixedMoeRuntime> make_mixed_moe_runtime(
    const mfq::mfe::Data& source,
    bool cuda,
    const CudaExecutionConfig& config = {});
int64_t mixed_moe_storage_bytes(const MixedMoeRuntime& runtime);
mfq_tensor_backend::Tensor copy_cpu_weight_to_cuda(
    const mfq_tensor_backend::Tensor& source);
mfq::cuda::Experts to_gpu_mixed_moe(
    const mfq::mfe::Data& source,
    const CudaExecutionConfig& config = {});
mfq::cuda::Experts to_cuda_device_moe_expert_slice(
    const mfq::mfe::Data& source,
    int64_t expert_begin,
    int64_t expert_end,
    int device,
    const CudaExecutionConfig& config = {});
mfq::cuda::Experts cpu_mixed_moe_metadata(
    const std::shared_ptr<MixedMoeRuntime>& runtime);
