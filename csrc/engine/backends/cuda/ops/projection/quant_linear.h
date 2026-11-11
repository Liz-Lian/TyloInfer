#pragma once

#include "engine/backends/cuda/ops/projection/fp8_sq.h"
#include "engine/backends/cuda/ops/projection/mx.h"
#include "engine/backends/cuda/ops/projection/mxfp4_sq.h"
#include "engine/backends/cuda/ops/projection/nint.h"
#include "engine/backends/cuda/ops/projection/vq.h"
#include "engine/backends/cuda/ops/projection/tq.h"
#include "engine/weights/encoding/format_features.h"
#include "engine/weights/encoding/tq.h"
#include "engine/weights/placement.h"
#include "engine/weights/sharded.h"

#include "mmq.h"
#include "parallel.h"
#include "backend.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

enum class InputGateMode : int { None = 0, Sigmoid = 1, Silu = 2 };

struct QuantLinearGroup;
struct DenseLinearGroup;

::tyloi::DenseTensor dense_projection(
    ::tyloi::DenseTensor input, const ::tyloi::DenseTensor& weight);

class TqExec;
namespace tyloi::cuda { class Collectives; }

::tyloi::DenseTensor run_nint_linear(
    CudaProfiler& profiler, KlMmqState& kl_mmq,
    const tyloi::NintTensor& weight, ::tyloi::DenseTensor input,
    MfqOptional<::tyloi::DenseTensor> gate = mfq_nullopt, InputGateMode mode = InputGateMode::None);
::tyloi::DenseTensor run_nvq_linear(
    CudaProfiler& profiler, KlMmqState& kl_mmq,
    const tyloi::NvqTensor& weight, ::tyloi::DenseTensor input,
    MfqOptional<::tyloi::DenseTensor> gate = mfq_nullopt, InputGateMode mode = InputGateMode::None);

::tyloi::DenseTensor run_quant_linear_shard(
    CudaProfiler& profiler,
    KlMmqState& kl_mmq,
    TqExec& tq,
    const tyloi::ShardedTensor::Shard& shard,
    ::tyloi::DenseTensor input,
    MfqOptional<::tyloi::DenseTensor> gate = mfq_nullopt,
    InputGateMode gate_mode = InputGateMode::None);


struct NintLinearGroup {
    tyloi::NintTensor w;
    std::vector<tyloi::NintTensor> projection_w;
    std::vector<tyloi::NintTensor> split_w;
    std::vector<std::vector<int64_t>> split_outs;
    std::vector<int64_t> outs;
};

// Reads the concrete representation a projection was prepared with, and
// rejects a weight that is not one.
template <class Format>
const Format& payload_as(const ::tyloi::Tensor& weight) {
    const auto* found = weight.representation<Format>();
    TYLOINFER_RUNTIME_CHECK(found, "projection weight has an unexpected representation");
    return *found;
}


void check_projection_width(const ::tyloi::Tensor& weight, int64_t width);
void check_projection_device(const ::tyloi::Tensor& weight, ::tyloi::Device device);
void check_projection(const ::tyloi::Tensor& weight, const ::tyloi::DenseTensor& input);

void check_device(const ::tyloi::DenseTensor& weight, const ::tyloi::DenseTensor& input);
void check_device(const ::tyloi::TensorImpl& weight, const ::tyloi::DenseTensor& input);
inline void check_device(const ::tyloi::Tensor& weight, const ::tyloi::DenseTensor& input) {
    check_device(weight.implementation(), input);
}

tyloi::weights::Placement projection_placement(const ::tyloi::Tensor& weight, std::size_t index = 0);

// Where a weight's buffers actually live, matched against the running device.
// Every buffer of a representation is defined and resident on `device`.
void check_fields(const ::tyloi::TensorImpl& weight, ::tyloi::Device device, bool check_device = true);
inline void check_fields(const ::tyloi::Tensor& weight, ::tyloi::Device device, bool check_device = true) {
    check_fields(weight.implementation(), device, check_device);
}

bool tensor_parallel_output_projections_compatible(
    const std::vector<::tyloi::Tensor>& projections);

struct QuantLinearGroup {
    bool nint_grouped = false;
    bool nvq_prefix2 = false;
    bool decode_branch_parallel = true;
    NintLinearGroup nint;
    TqProjectionGroup tq;
    std::vector<::tyloi::Tensor> layers;
    std::vector<int64_t> outs;
    std::vector<int64_t> cols;
    std::vector<tyloi::weights::Placement> placements;

    std::size_t size() const { return outs.size(); }
    // The projected width and input width of one matrix in the group.
    tyloi::weights::Shape shape(std::size_t index) const {
        return {outs.at(index), cols.at(index)};
    }

    std::vector<::tyloi::Tensor> tensor_parallel_output_projections() const;

    bool tensor_parallel_output_compatible() const;
};

void check_projection(const QuantLinearGroup& group, const ::tyloi::DenseTensor& input);

struct DenseLinearGroup {
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor w;
#else
    ::tyloi::DenseTensor w;
#endif
    std::vector<int64_t> outs;

    std::vector<::tyloi::DenseTensor> forward(::tyloi::DenseTensor x) const;
};

::tyloi::DenseTensor nint_kld(
    CudaProfiler& profiler, KlMmqState& mmq, const ::tyloi::Tensor& weight,
    ::tyloi::DenseTensor input, ::tyloi::DenseTensor gate, InputGateMode mode);

::tyloi::DenseTensor quant_embedding_lookup(
    const ::tyloi::Tensor& weight, ::tyloi::DenseTensor token_ids);

::tyloi::DenseTensor nvq_ffn_swiglu_down(
    CudaProfiler& profiler,
    const tyloi::NvqTensor& gate,
    const tyloi::NvqTensor& up,
    const tyloi::NvqTensor& down,
    ::tyloi::DenseTensor input,
    MfqOptional<::tyloi::DenseTensor> residual = mfq_nullopt);

bool nvq_fused_residual_format(std::int64_t kernel_format);

::tyloi::DenseTensor nint_matmul_groupwise_u8(
    CudaProfiler& profiler,
    const tyloi::NintTensor& weight,
    ::tyloi::DenseTensor input,
    std::int64_t groups);

::tyloi::DenseTensor dequant_nint_dense_f32(const tyloi::NintTensor& weight);

::tyloi::DenseTensor quant_linear_reference_weight(
    const ::tyloi::Tensor& linear);
