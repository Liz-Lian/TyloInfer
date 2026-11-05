#pragma once

#include "engine/models/common/attention.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/attention/linear.h"
#include "engine/backends/cuda/ops/ffn/dense.h"
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/ops/rope/rope.h"
#endif

#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace tyloi::cuda {

class Ffn;

namespace task {

using Slot = size_t;

struct Norm {
    Slot input, output;
    ::tyloi::DenseTensor weight;
    double eps, offset;
};

struct Embed {
    Slot input, output;
    const ::tyloi::Tensor& weight;
};

struct Softcap {
    Slot input, output;
    double cap;
};

struct Concat {
    Slot first, second, output;
    int axis = -1;
};

struct Project {
    Slot input, output;
    const ::tyloi::Tensor& weight;
};

struct Attention {
    Slot input, residual, output;
    const FullAttention& weights;
    size_t state;
    ::tyloi::DenseTensor* residual_out = nullptr;
};

struct Linear {
    Slot input, residual, output;
    const LinearAttention& weights;
    size_t state;
};

struct Replay {
    const LinearAttention& weights;
    size_t state;
    int64_t tokens;
};

struct Dense {
    Slot residual, input, output;
    const Ffn& weights;
    ::tyloi::DenseTensor norm;
    double eps, offset;
    bool bf16, trace;
};

struct FullFfn {
    Slot residual, input, output;
    const Ffn& weights;
    ::tyloi::DenseTensor norm;
    const FullAttention& attention;
};

struct LinearFfn {
    Slot residual, input, output;
    const Ffn& weights;
    ::tyloi::DenseTensor norm;
    const LinearAttention& attention;
};

} // namespace task

struct Task {
    std::variant<task::Norm, task::Embed, task::Softcap, task::Concat, task::Project, task::Attention,
                 task::Linear,
                 task::Replay, task::Dense, task::FullFfn, task::LinearFfn> op;
    int scope = -1;
    std::string_view trace;
    std::string_view profile = {};
};

// State and weights remain owned by the caller until the execution lane drains.
struct LinearStateView {
    LinearView state;
    LinearInputs* captured = nullptr;
};

struct Inputs {
    std::vector<::tyloi::DenseTensor> values;
    std::optional<size_t> output = 0;
    std::span<const AttentionKvView> kv;
    AttentionStep attention;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    std::optional<tyloi::ops::RopeTable> rope;
#else
    const RopeCache* rope = nullptr;
#endif
    Trace* trace = nullptr;
    int trace_scope = -1;
    std::span<const LinearStateView> linear;
    LinearAttention::Options linear_options;
    size_t scratch_offset = 0;
};

} // namespace tyloi::cuda
