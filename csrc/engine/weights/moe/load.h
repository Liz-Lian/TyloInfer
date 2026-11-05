#pragma once

#include "engine/weights/moe/experts.h"
#include "engine/weights/moe/options.h"
#include "engine/weights/encoding/quant/mfe_blob.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tyloi { namespace source { class Source; } }

tyloi::mfe::Data load_mfe_cpu(
    const tyloi::source::Source& source, const std::string& name);
std::shared_ptr<::tyloi::MfeTensor> load_mfe_cpu_offloaded(
    const tyloi::source::Source& source, const std::string& name);
::tyloi::DenseTensor mfe_dense_reference(
    const tyloi::source::Source& source, const std::string& name,
    ::tyloi::DenseTensor input, const std::vector<std::int32_t>& expert_ids,
    int tokens, int routes, bool routed_input);
::tyloi::DenseTensor materialize_mfe_dense(
    const tyloi::source::Source& source, const std::string& name);
tyloi::cuda::Experts to_gpu_mfe(const tyloi::mfe::Data& source);
std::shared_ptr<::tyloi::MfeTensor> make_mfe(
    const tyloi::mfe::Data& source,
    bool cuda,
    const MoeLoadOptions& options = {});
int64_t mixed_moe_storage_bytes(const ::tyloi::MfeTensor& runtime);
::tyloi::DenseTensor copy_cpu_weight_to_cuda(
    const ::tyloi::DenseTensor& source);
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
    const std::shared_ptr<::tyloi::MfeTensor>& runtime);
