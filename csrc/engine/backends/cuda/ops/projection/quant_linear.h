#pragma once

#include "engine/backends/cuda/ops/projection/fp8_sq.h"
#include "engine/backends/cuda/ops/projection/mx.h"
#include "engine/backends/cuda/ops/projection/mxfp4_sq.h"
#include "engine/backends/cuda/ops/projection/nint.h"
#include "engine/backends/cuda/ops/projection/vq.h"
#include "engine/backends/cuda/ops/projection/tq.h"
#include "engine/weights/encoding/format_features.h"
#include "engine/weights/encoding/tq.h"
#include "engine/weights/weights.h"

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

struct QuantLinear;
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
    const struct QuantLinearShard& shard,
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

enum class QuantLinearKind {
    Nint,
    Nvq,
    Mxfp4,
    Mxfp4Sq,
    Fp8Sq,
    Mxfp8,
    Dense,
    Tq,
};


struct QuantLinearShard {
    int device = 0;
    int64_t input_begin = 0;
    int64_t input_end = 0;
    int64_t output_begin = 0;
    int64_t output_end = 0;
    QuantLinearKind kind = QuantLinearKind::Nint;
    tyloi::NintTensor nint;
    tyloi::NvqTensor nvq;
    tyloi::Mxfp4Tensor mxfp4;
    tyloi::Mxfp4SqTensor mxfp4_sq;
    tyloi::Fp8SqTensor fp8_sq;
    tyloi::Mxfp8Tensor mxfp8;
    ::tyloi::DenseTensor dense;
    tyloi::encoding::TqTensor tq;
};

struct QuantLinear : tyloi::weights::Weights {
    QuantLinearKind kind = QuantLinearKind::Nint;
    tyloi::NintTensor nint;
    tyloi::NvqTensor nvq;
    tyloi::Mxfp4Tensor mxfp4;
    tyloi::Mxfp4SqTensor mxfp4_sq;
    tyloi::Fp8SqTensor fp8_sq;
    tyloi::Mxfp8Tensor mxfp8;
    ::tyloi::DenseTensor dense;
    tyloi::encoding::TqTensor tq;
    TensorParallelAxis tensor_parallel_axis =
        TensorParallelAxis::Mirrored;
    std::vector<QuantLinearShard> tensor_parallel_shards;
    int64_t logical_out = 0;
    int64_t logical_neuron_len = 0;

    bool tensor_parallel() const {
        return !tensor_parallel_shards.empty();
    }

    bool is_nint() const { return tyloi::mfq_compat_enabled && kind == QuantLinearKind::Nint; }
    bool is_nvq() const { return tyloi::mfq_compat_enabled && kind == QuantLinearKind::Nvq; }
    bool is_mxfp4() const { return kind == QuantLinearKind::Mxfp4; }
    bool is_mxfp4_sq() const { return tyloi::mfq_compat_enabled && kind == QuantLinearKind::Mxfp4Sq; }
    bool is_fp8_sq() const { return tyloi::mfq_compat_enabled && kind == QuantLinearKind::Fp8Sq; }
    bool is_mxfp8() const { return kind == QuantLinearKind::Mxfp8; }
    bool is_dense() const { return kind == QuantLinearKind::Dense; }
    bool is_tq() const { return kind == QuantLinearKind::Tq; }

    int64_t out() const;
    int64_t neuron_len() const;
    tyloi::weights::Layout layout() const override { return tyloi::weights::Layout::matrix; }
    tyloi::DeviceType backend() const override { return tyloi::DeviceType::cuda; }
    std::size_t size() const override { return 1; }
    tyloi::weights::Shape shape(std::size_t index = 0) const override {
        if (index != 0) throw std::out_of_range("matrix index out of range");
        return {out(), neuron_len()};
    }
    tyloi::weights::Placement placement(std::size_t index = 0) const override;
};

const QuantLinear& as_quant_linear(const tyloi::weights::Weights& weight);

void check_projection_width(const QuantLinear& weight, int64_t width);
void check_projection_device(const QuantLinear& weight, ::tyloi::Device device);
void check_projection(const QuantLinear& weight, const ::tyloi::DenseTensor& input);

void check_device(const ::tyloi::DenseTensor& weight, const ::tyloi::DenseTensor& input);
void check_device(const ::tyloi::Tensor& weight, const ::tyloi::DenseTensor& input);

using QuantLinearProjectionRefs =
    std::vector<const QuantLinear*>;

bool tensor_parallel_output_projections_compatible(
    const QuantLinearProjectionRefs& projections);

struct QuantLinearGroup : tyloi::weights::Weights {
    bool nint_grouped = false;
    bool nvq_prefix2 = false;
    bool decode_branch_parallel = true;
    NintLinearGroup nint;
    TqProjectionGroup tq;
    std::vector<QuantLinear> layers;
    std::vector<int64_t> outs;
    std::vector<int64_t> cols;
    std::vector<tyloi::weights::Placement> placements;

    tyloi::weights::Layout layout() const override { return tyloi::weights::Layout::group; }
    tyloi::DeviceType backend() const override { return tyloi::DeviceType::cuda; }
    std::size_t size() const override { return outs.size(); }
    tyloi::weights::Shape shape(std::size_t index = 0) const override {
        return {outs.at(index), cols.at(index)};
    }
    tyloi::weights::Placement placement(std::size_t index = 0) const override {
        return placements.at(index);
    }

    QuantLinearProjectionRefs tensor_parallel_output_projections() const;

    bool tensor_parallel_output_compatible() const;
};

const QuantLinearGroup& as_quant_linear_group(const tyloi::weights::Weights& group);
void check_projection(const QuantLinearGroup& group, const ::tyloi::DenseTensor& input);

struct DenseLinearGroup {
    ::tyloi::DenseTensor w;
    std::vector<int64_t> outs;

    std::vector<::tyloi::DenseTensor> forward(::tyloi::DenseTensor x) const;
};

::tyloi::DenseTensor nint_kld(
    CudaProfiler& profiler, KlMmqState& mmq, const QuantLinear& weight,
    ::tyloi::DenseTensor input, ::tyloi::DenseTensor gate, InputGateMode mode);

::tyloi::DenseTensor quant_embedding_lookup(
    const QuantLinear& weight, ::tyloi::DenseTensor token_ids);

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
    const QuantLinear& linear);
