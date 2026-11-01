#pragma once

#include "backend.h"

#include <cstdint>

enum class KlMmqMode {
    Default,
    Nint8One,
    Fp16,
};

struct KlMmqState {
    KlMmqMode mode = KlMmqMode::Default;
    int64_t activation_quantize_calls = 0;
    int64_t dense_calls = 0;
    int64_t moe_calls = 0;
    int64_t fallback_calls = 0;
    int64_t kv_cache_capacity = 0;

    mfq_tensor_backend::Tensor prepare_activation(
        mfq_tensor_backend::Tensor input);
    void reset() noexcept;
};

const char* kl_mmq_mode_name(KlMmqMode mode);

struct KlMmqScope {
    KlMmqState& state;
    KlMmqState previous;

    KlMmqScope(KlMmqMode mode, KlMmqState& state);
    ~KlMmqScope();
};

struct KlKvCacheCapacityScope {
    KlMmqState& state;
    int64_t previous_capacity;

    KlKvCacheCapacityScope(
        int64_t capacity, KlMmqState& state);
    ~KlKvCacheCapacityScope();
};
