#include "mmq.h"
#include "quant_linear.h"
#include "formats/format_features.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"

#include <stdexcept>

KlMmqScope::KlMmqScope(KlMmqMode mode, KlMmqState& current)
    : state(current), previous(current) {
    state.mode = mode;
    state.activation_quantize_calls = 0;
    state.dense_calls = 0;
    state.moe_calls = 0;
    state.fallback_calls = 0;
}

KlMmqScope::~KlMmqScope() {
    const auto kv_cache_capacity = state.kv_cache_capacity;
    state = previous;
    state.kv_cache_capacity = kv_cache_capacity;
}

KlKvCacheCapacityScope::KlKvCacheCapacityScope(
        int64_t capacity, KlMmqState& current)
    : state(current), previous_capacity(current.kv_cache_capacity) {
    state.kv_cache_capacity = capacity;
}

KlKvCacheCapacityScope::~KlKvCacheCapacityScope() {
    state.kv_cache_capacity = previous_capacity;
}

const char * kl_mmq_mode_name(KlMmqMode mode) {
    switch (mode) {
        case KlMmqMode::Default: return "default";
        case KlMmqMode::Nint8One: return "nint8_1";
        case KlMmqMode::Fp16: return "fp16";
    }
    return "invalid";
}

mfq_tensor_backend::Tensor KlMmqState::prepare_activation(
        mfq_tensor_backend::Tensor input) {
    if (mode != KlMmqMode::Nint8One) return input;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto original_shape = input.sizes().vec();
    auto flat = input.reshape({-1, input.size(-1)}).contiguous();
    auto quantized = nint8_one_quantize_reconstruct_cuda(flat);
    ++activation_quantize_calls;
    return quantized.at(3).reshape(original_shape).contiguous();
#else
    throw std::runtime_error("NINT activation diagnostics require a compatibility build");
#endif
}

void KlMmqState::reset() noexcept {
    *this = {};
}

mfq_tensor_backend::Tensor nint_kld(
        CudaProfiler& profiler, KlMmqState& mmq, const QuantLinear& weight,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        InputGateMode mode) {
    TYLOINFER_RUNTIME_CHECK(
        !weight.tensor_parallel() && (mfq::mfq_compat_enabled && weight.is_nint()),
        "FP32-output KLD down projection requires a local NINT tensor");
    TYLOINFER_RUNTIME_CHECK(
        mmq.mode == KlMmqMode::Fp16,
        "FP32-output NINT MMQ is restricted to the FP16 KLD path");
    auto shape = x.sizes().vec();
    auto y = nint_matmul_input_mul_f32(
        profiler, weight.nint, x.reshape({-1, x.size(-1)}),
        gate.reshape({-1, gate.size(-1)}), static_cast<int>(mode));
    if (weight.nint.q8_zero) ++mmq.dense_calls;
    shape.back() = y.size(-1);
    return y.reshape(shape);
}
