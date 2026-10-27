#pragma once

#include "ops/attention/full.h"
#include "ops/attention/linear.h"
#include "ops/ffn/dense.h"

#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace mfq::cuda {

struct Ffn;

namespace task {

using Slot = size_t;

struct Norm {
    Slot input, output;
    mfq_tensor_backend::Tensor weight;
    double eps, offset;
};

struct Concat {
    Slot first, second, output;
    int axis = -1;
};

struct Project {
    Slot input, output;
    const QuantLinear& weight;
};

struct Attention {
    Slot input, residual, output;
    const FullAttention& weights;
    size_t state;
};

struct Linear {
    Slot input, residual, output;
    const LinearAttention& weights;
    size_t state;
};

struct Dense {
    Slot residual, input, output;
    mfq::cuda::Dense weights;
};

struct FullFfn {
    Slot residual, input, output;
    const Ffn& weights;
    mfq_tensor_backend::Tensor norm;
    const FullAttention& attention;
};

struct LinearFfn {
    Slot residual, input, output;
    const Ffn& weights;
    mfq_tensor_backend::Tensor norm;
    const LinearAttention& attention;
};

} // namespace task

struct Task {
    std::variant<task::Norm, task::Concat, task::Project, task::Attention, task::Linear,
                 task::Dense, task::FullFfn, task::LinearFfn> op;
    int scope = -1;
    std::string_view trace;
};

// State and weights remain owned by the caller until the execution lane drains.
struct LinearStateView {
    LinearView state;
    LinearInputs* captured = nullptr;
};

struct Inputs {
    std::vector<mfq_tensor_backend::Tensor> values;
    size_t output = 0;
    std::span<const AttentionKvView> kv;
    AttentionStep attention;
    const RopeCache* rope = nullptr;
    Trace* trace = nullptr;
    int trace_scope = -1;
    std::span<const LinearStateView> linear;
    LinearAttention::Options linear_options;
    size_t scratch_offset = 0;
};

} // namespace mfq::cuda
