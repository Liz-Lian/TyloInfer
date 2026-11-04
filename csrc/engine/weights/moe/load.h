#pragma once

#include "engine/weights/moe/experts.h"
#include "engine/weights/encoding/quant/mfe_blob.h"

#include <cstdint>
#include <memory>
#include <string>

namespace tyloi { namespace source { class Source; } }

struct MoeLoadOptions {
    bool nvq2 = false;
    bool nvq_groups = false;
    bool heterogeneous = true;
};

tyloi::mfe::Data load_mfe_cpu(
    const tyloi::source::Source& source, const std::string& name);
tyloi::cuda::Experts to_gpu_mfe(const tyloi::mfe::Data& source);
std::shared_ptr<MixedMoeRuntime> make_mixed_moe_runtime(
    const tyloi::mfe::Data& source,
    bool cuda,
    const MoeLoadOptions& options = {});
int64_t mixed_moe_storage_bytes(const MixedMoeRuntime& runtime);
::tyloi::Tensor copy_cpu_weight_to_cuda(
    const ::tyloi::Tensor& source);
tyloi::cuda::Experts to_gpu_mixed_moe(
    const tyloi::mfe::Data& source,
    const MoeLoadOptions& options = {});
tyloi::cuda::Experts to_cuda_device_moe_expert_slice(
    const tyloi::mfe::Data& source,
    int64_t expert_begin,
    int64_t expert_end,
    int device,
    const MoeLoadOptions& options = {});
tyloi::cuda::Experts cpu_mixed_moe_metadata(
    const std::shared_ptr<MixedMoeRuntime>& runtime);
