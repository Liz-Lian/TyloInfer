#pragma once
__global__ void nvq_quantize_x_gs24_kernel(
    const __half * x,
    int8_t * qx,
    float * xscale,
    int M,
    int K,
    int ng) {
    const int m = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (group >= ng) return;
    const int k = group * kGroupSize + lane;
    const bool valid = lane < kGroupSize && k < K;
    const float value = valid ? __half2float(x[static_cast<int64_t>(m) * K + k]) : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int q = 0;
    if (valid) {
        q = static_cast<int>(roundf(value / scale));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    if (lane == 0) xscale[static_cast<int64_t>(m) * ng + group] = scale;
    if (lane < kGroupSize) {
        qx[(static_cast<int64_t>(m) * ng + group) * kGroupSize + lane] =
            valid ? static_cast<int8_t>(q) : static_cast<int8_t>(0);
    }
}

template <int MODE>
__global__ void nvq_quantize_x_gate_gs24_kernel(
    const __half * x,
    const __half * gate,
    int8_t * qx,
    float * xscale,
    int M,
    int K,
    int ng) {
    const int m = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    const int k = group * kGroupSize + lane;
    const bool valid = lane < kGroupSize && k < K;
    float value = 0.0f;
    if (valid) {
        const int64_t offset = static_cast<int64_t>(m) * K + k;
        const float gate_value = __half2float(gate[offset]);
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        const float multiplier = MODE == 1 ? sigmoid : gate_value * sigmoid;
        value = __half2float(x[offset]) * multiplier;
    }
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int q = 0;
    if (valid) {
        q = static_cast<int>(roundf(value / scale));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    if (lane == 0) xscale[static_cast<int64_t>(m) * ng + group] = scale;
    if (lane < kGroupSize) {
        qx[(static_cast<int64_t>(m) * ng + group) * kGroupSize + lane] =
            valid ? static_cast<int8_t>(q) : static_cast<int8_t>(0);
    }
}

template <int FORMAT>
__device__ __forceinline__ float nvq_vec8_dot(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    const int8_t * activation,
    int row,
    int segment,
    int group,
    int ng,
    int nvec,
    int nsign,
    int sign_mode,
    uint32_t state) {
    return dot_nvq_vec8<FORMAT>(
        load_nvq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, segment, group, ng, nvec, nsign, sign_mode, state),
        activation);
}

template <int FORMAT>
__device__ __forceinline__ float nvq_vec8_scaled_fma(
    const NvqDeviceWeight & weight,
    const int8_t * activation,
    const float * activation_scale,
    int row,
    int segment,
    float acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    const int64_t sub_linear = static_cast<int64_t>(row) * weight.ng + group;
    const uint32_t sub = load_nvq_state<FORMAT>(weight.indices,weight.indices_nbytes,
        weight.sub_scale,weight.sub_scale_nbytes,sub_linear,weight.sub_bits,weight.ng);
    const float dot = nvq_vec8_dot<FORMAT>(
        weight.indices, weight.indices_nbytes,
        weight.aux, weight.aux_nbytes,
        weight.codebook, activation + k,
        row, segment, group, weight.ng, weight.nvec, weight.nsign,
        weight.sign_mode, sub);
    const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
        ? weight.neuron_scale[row] * static_cast<float>(sub)
        : format_scale<FORMAT>(weight.neuron_scale[row], sub, weight.codebook);
    return fmaf(weight_scale * activation_scale[group], dot, acc);
}

template <bool EXEC_LAYOUT = false, bool JSC = false>
__device__ __forceinline__ float nvq2_vec8_scaled_fma_s4(
    const NvqDeviceWeight & weight,
    const int8_t * activation,
    const float * activation_scale,
    int64_t index_row,
    int64_t sign_row,
    int64_t sub_row,
    float anchor,
    int segment,
    float acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    uint32_t index;
    uint32_t mask8;
    if constexpr (EXEC_LAYOUT) {
        const uint16_t metadata =
            reinterpret_cast<const uint16_t *>(weight.indices)[index_row + segment];
        index = metadata & 0xffu;
        mask8 = metadata >> 8;
    } else {
        index = weight.indices[index_row + segment];
        const int64_t sign_linear = sign_row + segment;
        const int64_t sign_bit = sign_linear * 7;
        const int64_t sign_byte = sign_bit >> 3;
        const int sign_shift = static_cast<int>(sign_bit & 7);
        uint32_t sign_word = weight.aux[sign_byte];
        if (sign_byte + 1 < weight.aux_nbytes) {
            sign_word |= static_cast<uint32_t>(weight.aux[sign_byte + 1]) << 8;
        }
        const uint32_t mask7 = (sign_word >> sign_shift) & 0x7fu;
        const int last = parity7(mask7) ^
            (weight.sign_mode ? ((index >> 7) & 1u) : 0u);
        mask8 = mask7 | (static_cast<uint32_t>(last) << 7);
    }
    const uint32_t sub = load_packed_4(weight.sub_scale, sub_row + group);
    const int8_t * bank = JSC
        ? active_codebook<kNvq2Jsc>(weight.codebook, sub)
        : weight.codebook;
    const int2 code = reinterpret_cast<const int2 *>(bank)[index];
    const int2 values = EXEC_LAYOUT ? apply_sign8_exec(code, mask8) : apply_sign8(code, mask8);
    const int2 x = *reinterpret_cast<const int2 *>(activation + k);
    const int dot = __dp4a(values.y, x.y, __dp4a(values.x, x.x, 0));
    const float weight_scale = JSC
        ? format_scale<kNvq2Jsc>(anchor, sub, weight.codebook)
        : anchor * static_cast<float>(sub);
    return fmaf(weight_scale * activation_scale[group], static_cast<float>(dot), acc);
}

__device__ __forceinline__ uint32_t load_nvq_sign7(
    const uint8_t * aux,
    int64_t aux_nbytes,
    int64_t sign_linear) {
    const int64_t sign_bit = sign_linear * 7;
    const int64_t sign_byte = sign_bit >> 3;
    const int sign_shift = static_cast<int>(sign_bit & 7);
    uint32_t word = aux[sign_byte];
    if (sign_byte + 1 < aux_nbytes) {
        word |= static_cast<uint32_t>(aux[sign_byte + 1]) << 8;
    }
    return (word >> sign_shift) & 0x7fu;
}

template <bool JSC = false, bool ANALYTIC2 = false>
__device__ __forceinline__ float nvq3_vec8_scaled_fma_s4(
    const NvqDeviceWeight & weight,
    const int8_t * activation,
    const float * activation_scale,
    int64_t index_row,
    int64_t sign_row,
    int64_t sub_row,
    float anchor,
    int segment,
    float acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    const int vector4 = segment * 2;
    const uint32_t index0 = weight.indices[index_row + vector4];
    const uint32_t index1 = vector4 + 1 < weight.nvec
        ? weight.indices[index_row + vector4 + 1]
        : 0;
    const uint32_t mask7 = load_nvq_sign7(weight.aux, weight.aux_nbytes, sign_row + segment);
    const uint32_t mask8 = mask7 | (static_cast<uint32_t>(parity7(mask7)) << 7);
    const uint32_t sub = load_packed_4(weight.sub_scale, sub_row + group);
    const uint8_t * metadata = reinterpret_cast<const uint8_t *>(weight.codebook);
    const int8_t * bank = weight.codebook;
    if constexpr (JSC) {
        const uint32_t bank_id = ANALYTIC2
            ? sub & 1u
            : metadata[kJscBankMapOffset + sub];
        bank = weight.codebook + kJscMetadataBytes + bank_id * kJscD4CodebookBytes;
    }
    const int2 values = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index0],
            reinterpret_cast<const int *>(bank)[index1]),
        mask8);
    const int2 x = *reinterpret_cast<const int2 *>(activation + k);
    const int dot = __dp4a(values.y, x.y, __dp4a(values.x, x.x, 0));
    float weight_scale;
    if constexpr (ANALYTIC2) {
        weight_scale = anchor * static_cast<float>((sub >> 1) + 1);
    } else if constexpr (JSC) {
        const __half * lut = reinterpret_cast<const __half *>(metadata + kJscLutOffset);
        weight_scale = anchor * __half2float(lut[sub]);
    } else {
        weight_scale = anchor * static_cast<float>(sub);
    }
    return fmaf(weight_scale * activation_scale[group], static_cast<float>(dot), acc);
}

template <bool EXEC_LAYOUT = false, bool JSC = false>
__device__ __forceinline__ void nvq2_vec8_pair_scaled_fma_s4(
    const NvqDeviceWeight & gate,
    const NvqDeviceWeight & up,
    const int8_t * activation,
    const float * activation_scale,
    int64_t index_row,
    int64_t sign_row,
    int64_t sub_row,
    float gate_anchor,
    float up_anchor,
    int segment,
    float & gate_acc,
    float & up_acc) {
    const int group = segment / 3;
    const int k = group * kGroupSize + (segment - group * 3) * 8;
    const int2 x = *reinterpret_cast<const int2 *>(activation + k);
    const float x_scale = activation_scale[group];
    uint32_t gate_index;
    uint32_t up_index;
    uint32_t gate_mask8;
    uint32_t up_mask8;
    if constexpr (EXEC_LAYOUT) {
        const uint16_t gate_metadata =
            reinterpret_cast<const uint16_t *>(gate.indices)[index_row + segment];
        const uint16_t up_metadata =
            reinterpret_cast<const uint16_t *>(up.indices)[index_row + segment];
        gate_index = gate_metadata & 0xffu;
        up_index = up_metadata & 0xffu;
        gate_mask8 = gate_metadata >> 8;
        up_mask8 = up_metadata >> 8;
    } else {
        const int64_t sign_linear = sign_row + segment;
        const int64_t sign_bit = sign_linear * 7;
        const int64_t sign_byte = sign_bit >> 3;
        const int sign_shift = static_cast<int>(sign_bit & 7);
        uint32_t gate_sign_word = gate.aux[sign_byte];
        uint32_t up_sign_word = up.aux[sign_byte];
        if (sign_byte + 1 < gate.aux_nbytes) {
            gate_sign_word |= static_cast<uint32_t>(gate.aux[sign_byte + 1]) << 8;
            up_sign_word |= static_cast<uint32_t>(up.aux[sign_byte + 1]) << 8;
        }
        const uint32_t gate_mask7 = (gate_sign_word >> sign_shift) & 0x7fu;
        const uint32_t up_mask7 = (up_sign_word >> sign_shift) & 0x7fu;
        gate_index = gate.indices[index_row + segment];
        up_index = up.indices[index_row + segment];
        const int gate_last = parity7(gate_mask7) ^
            (gate.sign_mode ? ((gate_index >> 7) & 1u) : 0u);
        const int up_last = parity7(up_mask7) ^
            (up.sign_mode ? ((up_index >> 7) & 1u) : 0u);
        gate_mask8 = gate_mask7 | (static_cast<uint32_t>(gate_last) << 7);
        up_mask8 = up_mask7 | (static_cast<uint32_t>(up_last) << 7);
    }
    const uint32_t gate_sub = load_packed_4(gate.sub_scale, sub_row + group);
    const uint32_t up_sub = load_packed_4(up.sub_scale, sub_row + group);
    const int8_t * gate_bank = JSC
        ? active_codebook<kNvq2Jsc>(gate.codebook, gate_sub)
        : gate.codebook;
    const int8_t * up_bank = JSC
        ? active_codebook<kNvq2Jsc>(up.codebook, up_sub)
        : up.codebook;
    const int2 gate_code = reinterpret_cast<const int2 *>(gate_bank)[gate_index];
    const int2 up_code = reinterpret_cast<const int2 *>(up_bank)[up_index];
    const int2 gate_values = EXEC_LAYOUT
        ? apply_sign8_exec(gate_code, gate_mask8)
        : apply_sign8(gate_code, gate_mask8);
    const int2 up_values = EXEC_LAYOUT
        ? apply_sign8_exec(up_code, up_mask8)
        : apply_sign8(up_code, up_mask8);
    const int gate_dot = __dp4a(
        gate_values.y, x.y, __dp4a(gate_values.x, x.x, 0));
    const int up_dot = __dp4a(
        up_values.y, x.y, __dp4a(up_values.x, x.x, 0));
    const float gate_scale = JSC
        ? format_scale<kNvq2Jsc>(gate_anchor, gate_sub, gate.codebook)
        : gate_anchor * static_cast<float>(gate_sub);
    const float up_scale = JSC
        ? format_scale<kNvq2Jsc>(up_anchor, up_sub, up.codebook)
        : up_anchor * static_cast<float>(up_sub);
    gate_acc = fmaf(gate_scale * x_scale, static_cast<float>(gate_dot), gate_acc);
    up_acc = fmaf(up_scale * x_scale, static_cast<float>(up_dot), up_acc);
}

// llama.cpp-style MMVQ schedule: one output row per block, multiple warps
// split K, and each lane consumes one complete 8-value VQ segment.
template <int FORMAT, int NWARPS, bool FAST_S4 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_m1_vec8_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    float acc = 0.0f;
    [[maybe_unused]] const NvqDeviceWeight fast_weight{
        indices, indices_nbytes, aux, aux_nbytes,
        sub_scale, sub_scale_nbytes, neuron_scale, codebook,
        0, N, ng, nvec, nsign, 4, sign_mode};
    [[maybe_unused]] const int64_t fast_index_row = static_cast<int64_t>(row) * nvec;
    [[maybe_unused]] const int64_t fast_sign_row = static_cast<int64_t>(row) * nsign;
    [[maybe_unused]] const int64_t fast_sub_row = static_cast<int64_t>(row) * ng;
    [[maybe_unused]] const float fast_anchor = neuron_scale[row];
    for (int segment = warp * 32 + lane;
         segment < nsign;
         segment += NWARPS * 32) {
        if constexpr (FAST_S4) {
            if constexpr (
                FORMAT == kNvq2 || FORMAT == kNvq2Exec ||
                FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec) {
                acc = nvq2_vec8_scaled_fma_s4<
                    FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec,
                    FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec>(
                    fast_weight, qx, xscale,
                    fast_index_row, fast_sign_row, fast_sub_row, fast_anchor,
                    segment, acc);
            } else {
                acc = nvq3_vec8_scaled_fma_s4<
                    FORMAT == kNvq3Jsc || FORMAT == kNvq3Jsc2,
                    FORMAT == kNvq3Jsc2>(
                    fast_weight, qx, xscale,
                    fast_index_row, fast_sign_row, fast_sub_row, fast_anchor,
                    segment, acc);
            }
        } else {
            const int group = segment / 3;
            const int segment_in_group = segment - group * 3;
            const int k = group * kGroupSize + segment_in_group * 8;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            const float dot = nvq_vec8_dot<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook, qx + k,
                row, segment, group, ng, nvec, nsign, sign_mode, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
            acc = fmaf(weight_scale * xscale[group], dot, acc);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) output[row] = __float2half(acc);
    }
}

template <int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq3j2_gemv_m1_group_kernel(
    const uint8_t * indices,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    const float * neuron_scale,
    const int8_t * metadata,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int N,
    int ng,
    int nvec,
    int nsign) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    const int64_t index_row = static_cast<int64_t>(row) * nvec;
    const int64_t sign_row = static_cast<int64_t>(row) * nsign;
    const int64_t sub_row = static_cast<int64_t>(row) * ng;
    const float anchor = neuron_scale[row];
    float acc = 0.0f;
    for (int group = warp * 32 + lane; group < ng; group += NWARPS * 32) {
        const uint32_t state = load_packed_4(sub_scale, sub_row + group);
        const int8_t * bank = metadata + kJscMetadataBytes
            + (state & 1u) * kJscD4CodebookBytes;
        int group_dot = 0;
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= nsign) continue;
            const int vector4 = segment * 2;
            const uint32_t index0 = indices[index_row + vector4];
            const uint32_t index1 = vector4 + 1 < nvec
                ? indices[index_row + vector4 + 1]
                : 0;
            const uint32_t mask7 = load_nvq_sign7(aux, aux_nbytes, sign_row + segment);
            const uint32_t mask8 = mask7
                | (static_cast<uint32_t>(parity7(mask7)) << 7);
            const int2 values = apply_sign8(
                make_int2(
                    reinterpret_cast<const int *>(bank)[index0],
                    reinterpret_cast<const int *>(bank)[index1]),
                mask8);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            group_dot = __dp4a(
                values.y, x.y, __dp4a(values.x, x.x, group_dot));
        }
        const float weight_scale = anchor * static_cast<float>((state >> 1) + 1);
        acc = fmaf(
            weight_scale * xscale[group], static_cast<float>(group_dot), acc);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) output[row] = __float2half(acc);
    }
}

__device__ __forceinline__ uint32_t extract_group_exec96_bits(
    const uint32_t (&words)[3], int bit, int bits) {
    return extract_group_exec96_bits_ptr(words, bit, bits);
}

__device__ __forceinline__ int aligned_group_d4_dot_words(
    const NvqDeviceWeight & weight,
    int group,
    const int8_t * qx,
    const uint32_t (&words)[3],
    uint32_t & state) {
    state = extract_group_exec96_bits(words, 84, 4);
    const int8_t * bank = active_codebook<kNvq3JscLGroupExec>(
        weight.codebook, state);
    const uint32_t index00 = extract_group_exec96_bits(words, 0, 10);
    const uint32_t index01 = extract_group_exec96_bits(words, 10, 10);
    const uint32_t index10 = extract_group_exec96_bits(words, 20, 10);
    const uint32_t index11 = extract_group_exec96_bits(words, 30, 10);
    const uint32_t index20 = extract_group_exec96_bits(words, 40, 10);
    const uint32_t index21 = extract_group_exec96_bits(words, 50, 10);
    const uint32_t mask0 = extract_group_exec96_bits(words, 60, 8);
    const uint32_t mask1 = extract_group_exec96_bits(words, 68, 8);
    const uint32_t mask2 = extract_group_exec96_bits(words, 76, 8);
    const int2 values0 = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index00],
            reinterpret_cast<const int *>(bank)[index01]),
        mask0);
    const int2 values1 = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index10],
            reinterpret_cast<const int *>(bank)[index11]),
        mask1);
    const int2 values2 = apply_sign8(
        make_int2(
            reinterpret_cast<const int *>(bank)[index20],
            reinterpret_cast<const int *>(bank)[index21]),
        mask2);
    const int8_t * activation = qx + group * kGroupSize;
    const int2 x0 = *reinterpret_cast<const int2 *>(activation);
    const int2 x1 = *reinterpret_cast<const int2 *>(activation + 8);
    const int2 x2 = *reinterpret_cast<const int2 *>(activation + 16);
    const int dot0 = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
    const int dot1 = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, 0));
    const int dot2 = __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, 0));
    return dot0 + dot1 + dot2;
}

template <int FORMAT>
__device__ __forceinline__ int aligned_group_dot(
    const NvqDeviceWeight & weight,
    int row,
    int group,
    const int8_t * qx,
    uint32_t & state) {
    static_assert(
        FORMAT == kNvq2JscXLGroupExec ||
        FORMAT == kNvq3JscLGroupExec);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        const uint64_t metadata = load_group_exec64(
            weight.indices, row, group, weight.ng);
        state = static_cast<uint32_t>(metadata >> 60);
        const int8_t * bank = active_codebook<FORMAT>(weight.codebook, state);
        int dot = 0;
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= weight.nsign) continue;
            const uint32_t segment_metadata =
                (metadata >> (local * 20)) & 0xfffffu;
            const uint32_t index = segment_metadata & 0xfffu;
            const uint32_t mask8 = segment_metadata >> 12;
            const int2 values = apply_sign8(
                reinterpret_cast<const int2 *>(bank)[index], mask8);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            dot = __dp4a(values.y, x.y, __dp4a(values.x, x.x, dot));
        }
        return dot;
    } else {
        uint32_t words[3];
        load_group_exec96_words(
            weight.indices, row, group, weight.ng, words);
        return aligned_group_d4_dot_words(weight, group, qx, words, state);
    }
}

template <int FORMAT>
__device__ __forceinline__ void aligned_group_pair_dot(
    const NvqDeviceWeight & gate,
    const NvqDeviceWeight & up,
    int row,
    int group,
    const int8_t * qx,
    uint32_t & gate_state,
    uint32_t & up_state,
    int & gate_dot,
    int & up_dot) {
    static_assert(
        FORMAT == kNvq2JscXLGroupExec ||
        FORMAT == kNvq3JscLGroupExec);
    gate_dot = 0;
    up_dot = 0;
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        const uint64_t gate_metadata = load_group_exec64(
            gate.indices, row, group, gate.ng);
        const uint64_t up_metadata = load_group_exec64(
            up.indices, row, group, up.ng);
        gate_state = static_cast<uint32_t>(gate_metadata >> 60);
        up_state = static_cast<uint32_t>(up_metadata >> 60);
        const int8_t * gate_bank = active_codebook<FORMAT>(
            gate.codebook, gate_state);
        const int8_t * up_bank = active_codebook<FORMAT>(
            up.codebook, up_state);
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= gate.nsign) continue;
            const uint32_t gate_segment =
                (gate_metadata >> (local * 20)) & 0xfffffu;
            const uint32_t up_segment =
                (up_metadata >> (local * 20)) & 0xfffffu;
            const uint32_t gate_index = gate_segment & 0xfffu;
            const uint32_t up_index = up_segment & 0xfffu;
            const uint32_t gate_mask = gate_segment >> 12;
            const uint32_t up_mask = up_segment >> 12;
            const int2 gate_values = apply_sign8(
                reinterpret_cast<const int2 *>(gate_bank)[gate_index],
                gate_mask);
            const int2 up_values = apply_sign8(
                reinterpret_cast<const int2 *>(up_bank)[up_index],
                up_mask);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            gate_dot = __dp4a(
                gate_values.y, x.y,
                __dp4a(gate_values.x, x.x, gate_dot));
            up_dot = __dp4a(
                up_values.y, x.y,
                __dp4a(up_values.x, x.x, up_dot));
        }
    } else {
        uint32_t gate_words[3];
        uint32_t up_words[3];
        load_group_exec96_words(
            gate.indices, row, group, gate.ng, gate_words);
        load_group_exec96_words(
            up.indices, row, group, up.ng, up_words);
        gate_state = extract_group_exec96_bits(gate_words, 84, 4);
        up_state = extract_group_exec96_bits(up_words, 84, 4);
        const int8_t * gate_bank = active_codebook<FORMAT>(
            gate.codebook, gate_state);
        const int8_t * up_bank = active_codebook<FORMAT>(
            up.codebook, up_state);
#pragma unroll
        for (int local = 0; local < 3; ++local) {
            const int segment = group * 3 + local;
            if (segment >= gate.nsign) continue;
            const int vector4 = segment * 2;
            const uint32_t gate_index0 = extract_group_exec96_bits(
                gate_words, local * 20, 10);
            const uint32_t gate_index1 = vector4 + 1 < gate.nvec
                ? extract_group_exec96_bits(
                    gate_words, local * 20 + 10, 10)
                : 0;
            const uint32_t up_index0 = extract_group_exec96_bits(
                up_words, local * 20, 10);
            const uint32_t up_index1 = vector4 + 1 < up.nvec
                ? extract_group_exec96_bits(
                    up_words, local * 20 + 10, 10)
                : 0;
            const uint32_t gate_mask = extract_group_exec96_bits(
                gate_words, 60 + local * 8, 8);
            const uint32_t up_mask = extract_group_exec96_bits(
                up_words, 60 + local * 8, 8);
            const int2 gate_values = apply_sign8(
                make_int2(
                    reinterpret_cast<const int *>(gate_bank)[gate_index0],
                    reinterpret_cast<const int *>(gate_bank)[gate_index1]),
                gate_mask);
            const int2 up_values = apply_sign8(
                make_int2(
                    reinterpret_cast<const int *>(up_bank)[up_index0],
                    reinterpret_cast<const int *>(up_bank)[up_index1]),
                up_mask);
            const int2 x = *reinterpret_cast<const int2 *>(
                qx + group * kGroupSize + local * 8);
            gate_dot = __dp4a(
                gate_values.y, x.y,
                __dp4a(gate_values.x, x.x, gate_dot));
            up_dot = __dp4a(
                up_values.y, x.y,
                __dp4a(up_values.x, x.x, up_dot));
        }
    }
}

template <
    int FORMAT, int NWARPS, int ROWS_PER_BLOCK = 1,
    bool ADD_RESIDUAL = false>
__global__ void __launch_bounds__(NWARPS * ROWS_PER_BLOCK * 32, 1)
aligned_group_gemv_m1_kernel(
    NvqDeviceWeight weight,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    const __half * residual) {
    const int row_slot = threadIdx.y / NWARPS;
    const int warp = threadIdx.y - row_slot * NWARPS;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_slot;
    const int lane = threadIdx.x;
    const bool valid = row < weight.N;

    const float anchor = valid ? weight.neuron_scale[row] : 0.0f;
    float acc = 0.0f;
    if (valid) {
        for (int group = warp * 32 + lane;
             group < weight.ng;
             group += NWARPS * 32) {
            uint32_t state = 0;
            const int dot = aligned_group_dot<FORMAT>(
                weight, row, group, qx, state);
            acc = fmaf(
                format_scale<FORMAT>(anchor, state, weight.codebook) *
                    xscale[group],
                static_cast<float>(dot), acc);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    if (lane == 0) partial[row_slot][warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[row_slot][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0 && valid) {
            const __half projected = __float2half(acc);
            if constexpr (ADD_RESIDUAL) {
                output[row] = __hadd(residual[row], projected);
            } else {
                output[row] = projected;
            }
        }
    }
}

template <int FORMAT, int NWARPS, int ROWS_PER_BLOCK = 1>
__global__ void __launch_bounds__(NWARPS * ROWS_PER_BLOCK * 32, 1)
aligned_group_multi2_m1_kernel(
    NvqDeviceWeight first,
    NvqDeviceWeight second,
    const int8_t * qx,
    const float * xscale,
    __half * output) {
    const int row_slot = threadIdx.y / NWARPS;
    const int warp = threadIdx.y - row_slot * NWARPS;
    const int combined_row = blockIdx.x * ROWS_PER_BLOCK + row_slot;
    const int total_rows = first.N + second.N;
    const bool valid = combined_row < total_rows;
    const bool use_second = combined_row >= first.N;
    const NvqDeviceWeight weight = use_second ? second : first;
    const int row = use_second ? combined_row - first.N : combined_row;
    const int lane = threadIdx.x;

    const float anchor = valid ? weight.neuron_scale[row] : 0.0f;
    float acc = 0.0f;
    if (valid) {
        for (int group = warp * 32 + lane;
             group < weight.ng;
             group += NWARPS * 32) {
            uint32_t state = 0;
            const int dot = aligned_group_dot<FORMAT>(weight, row, group, qx, state);
            acc = fmaf(
                format_scale<FORMAT>(anchor, state, weight.codebook) * xscale[group],
                static_cast<float>(dot), acc);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    if (lane == 0) partial[row_slot][warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = lane < NWARPS ? partial[row_slot][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0 && valid) output[combined_row] = __float2half(acc);
    }
}

template <int FORMAT>
__device__ __forceinline__ int aligned_group_dot_preloaded(
    const NvqDeviceWeight & weight,
    const int8_t * codebook,
    int row,
    int group,
    const int2 & x0,
    const int2 & x1,
    const int2 & x2,
    uint32_t & state) {
    static_assert(
        FORMAT == kNvq2JscXLGroupExec ||
        FORMAT == kNvq3JscLGroupExec);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        const uint64_t metadata = load_group_exec64(
            weight.indices, row, group, weight.ng);
        state = static_cast<uint32_t>(metadata >> 60);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        const uint32_t segment0 = static_cast<uint32_t>(metadata) & 0xfffffu;
        const uint32_t segment1 = static_cast<uint32_t>(metadata >> 20) & 0xfffffu;
        const uint32_t segment2 = static_cast<uint32_t>(metadata >> 40) & 0xfffffu;
        const int2 values0 = apply_sign8(
            reinterpret_cast<const int2 *>(bank)[segment0 & 0xfffu],
            segment0 >> 12);
        const int2 values1 = apply_sign8(
            reinterpret_cast<const int2 *>(bank)[segment1 & 0xfffu],
            segment1 >> 12);
        const int2 values2 = apply_sign8(
            reinterpret_cast<const int2 *>(bank)[segment2 & 0xfffu],
            segment2 >> 12);
        int dot = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
        dot = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, dot));
        return __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, dot));
    } else {
        uint32_t words[3];
        load_group_exec96_words(
            weight.indices, row, group, weight.ng, words);
        state = extract_group_exec96_bits(words, 84, 4);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        const uint32_t index00 = extract_group_exec96_bits(words, 0, 10);
        const uint32_t index01 = extract_group_exec96_bits(words, 10, 10);
        const uint32_t index10 = extract_group_exec96_bits(words, 20, 10);
        const uint32_t index11 = extract_group_exec96_bits(words, 30, 10);
        const uint32_t index20 = extract_group_exec96_bits(words, 40, 10);
        const uint32_t index21 = extract_group_exec96_bits(words, 50, 10);
        const int2 values0 = apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index00],
                reinterpret_cast<const int *>(bank)[index01]),
            extract_group_exec96_bits(words, 60, 8));
        const int2 values1 = apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index10],
                reinterpret_cast<const int *>(bank)[index11]),
            extract_group_exec96_bits(words, 68, 8));
        const int2 values2 = apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index20],
                reinterpret_cast<const int *>(bank)[index21]),
            extract_group_exec96_bits(words, 76, 8));
        const int dot0 = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
        const int dot1 = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, 0));
        const int dot2 = __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, 0));
        return dot0 + dot1 + dot2;
    }
}

__device__ __forceinline__ int aligned_group_dot_preloaded_e8_stage3_metadata(
    const int8_t * global_codebook,
    const int8_t * shared_codebook,
    bool table_is_staged,
    uint64_t metadata,
    const int2 & x0,
    const int2 & x1,
    const int2 & x2,
    uint32_t & state) {
    state = static_cast<uint32_t>(metadata >> 60);
    const uint32_t bank_id = state & 3u;
    const bool use_staged_bank = table_is_staged && bank_id < 3;
    const int8_t * bank = use_staged_bank
        ? shared_codebook + bank_id * kJscE8Codebook4096Bytes
        : global_codebook + kJscMetadataBytes +
            bank_id * kJscE8Codebook4096Bytes;
    const uint32_t segment0 = static_cast<uint32_t>(metadata) & 0xfffffu;
    const uint32_t segment1 = static_cast<uint32_t>(metadata >> 20) & 0xfffffu;
    const uint32_t segment2 = static_cast<uint32_t>(metadata >> 40) & 0xfffffu;
    const uint32_t index0 = segment0 & 0xfffu;
    const uint32_t index1 = segment1 & 0xfffu;
    const uint32_t index2 = segment2 & 0xfffu;
    int2 raw0;
    int2 raw1;
    int2 raw2;
    if (use_staged_bank) {
        const int * plane = reinterpret_cast<const int *>(bank);
        constexpr int plane_entries = 4096;
        raw0 = make_int2(plane[index0], plane[plane_entries + index0]);
        raw1 = make_int2(plane[index1], plane[plane_entries + index1]);
        raw2 = make_int2(plane[index2], plane[plane_entries + index2]);
    } else if (table_is_staged && bank_id == 3) {
        const int * partial = reinterpret_cast<const int *>(
            shared_codebook + 3 * kJscE8Codebook4096Bytes);
        const int2 * global = reinterpret_cast<const int2 *>(bank);
        raw0 = index0 < kJscE8PartialEntries
            ? make_int2(
                partial[index0], partial[kJscE8PartialEntries + index0])
            : global[index0];
        raw1 = index1 < kJscE8PartialEntries
            ? make_int2(
                partial[index1], partial[kJscE8PartialEntries + index1])
            : global[index1];
        raw2 = index2 < kJscE8PartialEntries
            ? make_int2(
                partial[index2], partial[kJscE8PartialEntries + index2])
            : global[index2];
    } else {
        raw0 = reinterpret_cast<const int2 *>(bank)[index0];
        raw1 = reinterpret_cast<const int2 *>(bank)[index1];
        raw2 = reinterpret_cast<const int2 *>(bank)[index2];
    }
    const int2 values0 = apply_sign8(raw0, segment0 >> 12);
    const int2 values1 = apply_sign8(raw1, segment1 >> 12);
    const int2 values2 = apply_sign8(raw2, segment2 >> 12);
    int dot = __dp4a(values0.y, x0.y, __dp4a(values0.x, x0.x, 0));
    dot = __dp4a(values1.y, x1.y, __dp4a(values1.x, x1.x, dot));
    return __dp4a(values2.y, x2.y, __dp4a(values2.x, x2.x, dot));
}

__device__ __forceinline__ int aligned_group_dot_preloaded_e8_stage3(
    const NvqDeviceWeight & weight,
    const int8_t * global_codebook,
    const int8_t * shared_codebook,
    bool table_is_staged,
    int row,
    int group,
    const int2 & x0,
    const int2 & x1,
    const int2 & x2,
    uint32_t & state) {
    const uint64_t metadata = load_group_exec64(
        weight.indices, row, group, weight.ng);
    return aligned_group_dot_preloaded_e8_stage3_metadata(
        global_codebook, shared_codebook, table_is_staged, metadata,
        x0, x1, x2, state);
}

// Decode geometry: every warp owns several output rows and reuses
// the same quantized activation values across those rows.  The compact D4
// tables fit in shared memory and are staged once per CTA; the 128 KiB E8
// tables remain in the read-only cache path.
template <
    int FORMAT, int NWARPS, int ROWS_PER_WARP, bool MULTI2,
    bool STAGE_E8_THREE_BANKS = false, bool ADD_RESIDUAL = false>
__global__ void __launch_bounds__(NWARPS * 32, 1)
aligned_group_row_tiled_m1_kernel(
    NvqDeviceWeight first,
    NvqDeviceWeight second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    const __half * residual) {
    static_assert(!ADD_RESIDUAL || !MULTI2);
    static_assert(NWARPS == 16 || NWARPS == 32);
    static_assert(
        ROWS_PER_WARP == 1 || ROWS_PER_WARP == 2 || ROWS_PER_WARP == 4);
    static_assert(
        !STAGE_E8_THREE_BANKS ||
        (FORMAT == kNvq2JscXLGroupExec && NWARPS == 32 &&
         (ROWS_PER_WARP == 1 || ROWS_PER_WARP == 4)));
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int linear_thread = warp * 32 + lane;
    const int block_threads = NWARPS * 32;
    const int k_pad = first.ng * kGroupSize;
    const int scale_offset = (k_pad + 3) & ~3;
    const int codebook_offset = scale_offset + first.ng * sizeof(float);
    extern __shared__ uint8_t shared_storage[];
    int8_t * shared_qx = reinterpret_cast<int8_t *>(shared_storage);
    float * shared_xscale = reinterpret_cast<float *>(
        shared_storage + scale_offset);

    if constexpr (!STAGE_E8_THREE_BANKS) {
        for (int item = linear_thread; item < k_pad; item += block_threads) {
            shared_qx[item] = qx[item];
        }
        for (int item = linear_thread; item < first.ng; item += block_threads) {
            shared_xscale[item] = xscale[item];
        }
    }

    const int8_t * first_codebook = first.codebook;
    const int8_t * second_codebook = second.codebook;
    const int rows_per_block = NWARPS * ROWS_PER_WARP;
    const int block_row_base = blockIdx.x * rows_per_block;
    const bool stage_second_table = MULTI2 && block_row_base >= first.N;
    const int8_t * staged_global_codebook = stage_second_table
        ? second.codebook
        : first.codebook;
    constexpr int stage_min_rows = ROWS_PER_WARP == 1 ? 4096 : 12288;
    const bool stage_active_table = STAGE_E8_THREE_BANKS &&
        (stage_second_table
            ? second.N >= stage_min_rows
            : first.N >= stage_min_rows);
    const int8_t * shared_e8_codebook = nullptr;
    if constexpr (STAGE_E8_THREE_BANKS) {
        int8_t * staged = reinterpret_cast<int8_t *>(shared_storage);
        constexpr int entries_per_bank = 4096;
        constexpr int staged_entries = 3 * entries_per_bank;
        if (stage_active_table) {
            for (int item = linear_thread;
                 item < staged_entries;
                 item += block_threads) {
                const int bank = item / entries_per_bank;
                const int entry = item - bank * entries_per_bank;
                const int2 value = reinterpret_cast<const int2 *>(
                    staged_global_codebook + kJscMetadataBytes +
                    bank * kJscE8Codebook4096Bytes)[entry];
                int * destination = reinterpret_cast<int *>(
                    staged + bank * kJscE8Codebook4096Bytes);
                destination[entry] = value.x;
                destination[entries_per_bank + entry] = value.y;
            }
            int * partial = reinterpret_cast<int *>(
                staged + 3 * kJscE8Codebook4096Bytes);
            for (int entry = linear_thread;
                 entry < kJscE8PartialEntries;
                 entry += block_threads) {
                const int2 value = reinterpret_cast<const int2 *>(
                    staged_global_codebook + kJscMetadataBytes +
                    3 * kJscE8Codebook4096Bytes)[entry];
                partial[entry] = value.x;
                partial[kJscE8PartialEntries + entry] = value.y;
            }
        }
        shared_e8_codebook = staged;
    }
    if constexpr (FORMAT == kNvq3JscLGroupExec) {
        int8_t * shared_first_codebook = reinterpret_cast<int8_t *>(
            shared_storage + codebook_offset);
        for (int item = linear_thread;
             item < first.codebook_nbytes;
             item += block_threads) {
            shared_first_codebook[item] = first.codebook[item];
        }
        first_codebook = shared_first_codebook;
        if constexpr (MULTI2) {
            int8_t * shared_second_codebook =
                shared_first_codebook + first.codebook_nbytes;
            for (int item = linear_thread;
                 item < second.codebook_nbytes;
                 item += block_threads) {
                shared_second_codebook[item] = second.codebook[item];
            }
            second_codebook = shared_second_codebook;
        }
    }
    __syncthreads();

    const int total_rows = MULTI2 ? first.N + second.N : first.N;
    const int row_base = blockIdx.x * rows_per_block + warp;
    float accumulators[ROWS_PER_WARP] = {};
    float anchors[ROWS_PER_WARP] = {};
#pragma unroll
    for (int item = 0; item < ROWS_PER_WARP; ++item) {
        const int combined_row = row_base + item * NWARPS;
        if (combined_row < total_rows) {
            if constexpr (MULTI2) {
                anchors[item] = combined_row < first.N
                    ? first.neuron_scale[combined_row]
                    : second.neuron_scale[combined_row - first.N];
            } else {
                anchors[item] = first.neuron_scale[combined_row];
            }
        }
    }

    for (int group = lane; group < first.ng; group += 32) {
        const int8_t * activation =
            (STAGE_E8_THREE_BANKS ? qx : shared_qx) + group * kGroupSize;
        const int2 x0 = *reinterpret_cast<const int2 *>(activation);
        const int2 x1 = *reinterpret_cast<const int2 *>(activation + 8);
        const int2 x2 = *reinterpret_cast<const int2 *>(activation + 16);
        const float activation_scale = STAGE_E8_THREE_BANKS
            ? xscale[group]
            : shared_xscale[group];
        if constexpr (STAGE_E8_THREE_BANKS) {
            {
                uint64_t metadata[ROWS_PER_WARP] = {};
                int rows[ROWS_PER_WARP] = {};
                bool use_second[ROWS_PER_WARP] = {};
                bool valid[ROWS_PER_WARP] = {};
#pragma unroll
                for (int item = 0; item < ROWS_PER_WARP; ++item) {
                    const int combined_row = row_base + item * NWARPS;
                    valid[item] = combined_row < total_rows;
                    use_second[item] = combined_row >= first.N;
                    rows[item] = use_second[item]
                        ? combined_row - first.N
                        : combined_row;
                    if (valid[item]) {
                        const uint8_t * indices = use_second[item]
                            ? second.indices
                            : first.indices;
                        metadata[item] = load_group_exec64(
                            indices, rows[item], group, first.ng);
                    }
                }
#pragma unroll
                for (int item = 0; item < ROWS_PER_WARP; ++item) {
                    if (!valid[item]) continue;
                    const int8_t * active_table = use_second[item]
                        ? second_codebook
                        : first_codebook;
                    uint32_t state = 0;
                    const int dot =
                        aligned_group_dot_preloaded_e8_stage3_metadata(
                            active_table, shared_e8_codebook,
                            stage_active_table &&
                                active_table == staged_global_codebook,
                            metadata[item], x0, x1, x2, state);
                    accumulators[item] = fmaf(
                        format_scale<FORMAT>(
                            anchors[item], state, active_table) *
                            activation_scale,
                        static_cast<float>(dot), accumulators[item]);
                }
            }
        } else {
#pragma unroll
            for (int item = 0; item < ROWS_PER_WARP; ++item) {
                const int combined_row = row_base + item * NWARPS;
                if (combined_row >= total_rows) continue;
                uint32_t state = 0;
                int dot = 0;
                const int8_t * active_table = first_codebook;
                if constexpr (MULTI2) {
                    if (combined_row < first.N) {
                        if constexpr (STAGE_E8_THREE_BANKS) {
                            dot = aligned_group_dot_preloaded_e8_stage3(
                                first, first_codebook, shared_e8_codebook,
                                stage_active_table &&
                                    first_codebook == staged_global_codebook,
                                combined_row, group, x0, x1, x2, state);
                        } else {
                            dot = aligned_group_dot_preloaded<FORMAT>(
                                first, first_codebook, combined_row, group,
                                x0, x1, x2, state);
                        }
                    } else {
                        const int row = combined_row - first.N;
                        active_table = second_codebook;
                        if constexpr (STAGE_E8_THREE_BANKS) {
                            dot = aligned_group_dot_preloaded_e8_stage3(
                                second, second_codebook, shared_e8_codebook,
                                stage_active_table &&
                                    second_codebook == staged_global_codebook,
                                row, group, x0, x1, x2, state);
                        } else {
                            dot = aligned_group_dot_preloaded<FORMAT>(
                                second, second_codebook, row, group,
                                x0, x1, x2, state);
                        }
                    }
                } else {
                    if constexpr (STAGE_E8_THREE_BANKS) {
                        dot = aligned_group_dot_preloaded_e8_stage3(
                            first, first_codebook, shared_e8_codebook,
                            stage_active_table,
                            combined_row, group, x0, x1, x2, state);
                    } else {
                        dot = aligned_group_dot_preloaded<FORMAT>(
                            first, first_codebook, combined_row, group,
                            x0, x1, x2, state);
                    }
                }
                accumulators[item] = fmaf(
                    format_scale<FORMAT>(anchors[item], state, active_table) *
                        activation_scale,
                    static_cast<float>(dot), accumulators[item]);
            }
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            accumulators[item] += __shfl_xor_sync(
                0xffffffff, accumulators[item], offset);
        }
    }
    if constexpr (ADD_RESIDUAL) {
        auto * projected_rows = reinterpret_cast<__half *>(shared_storage);
        // Every warp must finish its codebook reads before this storage is
        // reused for the coalesced residual epilogue.
        __syncthreads();
        if (lane == 0) {
#pragma unroll
            for (int item = 0; item < ROWS_PER_WARP; ++item) {
                const int local_row = warp + item * NWARPS;
                const int combined_row = block_row_base + local_row;
                if (combined_row < total_rows) {
                    projected_rows[local_row] =
                        __float2half(accumulators[item]);
                }
            }
        }
        __syncthreads();
        if (linear_thread < rows_per_block) {
            const int combined_row = block_row_base + linear_thread;
            if (combined_row < total_rows) {
                output[combined_row] = __hadd(
                    residual[combined_row], projected_rows[linear_thread]);
            }
        }
    } else if (lane == 0) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            const int combined_row = row_base + item * NWARPS;
            if (combined_row < total_rows) {
                output[combined_row] = __float2half(accumulators[item]);
            }
        }
    }
}

template <int FORMAT, int NWARPS, bool OUTPUT_F32 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) aligned_group_swiglu_pair_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * qx,
    const float * xscale,
    __half * output_h,
    float * output_f) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= gate.N) return;

    const float gate_anchor = gate.neuron_scale[row];
    const float up_anchor = up.neuron_scale[row];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int group = warp * 32 + lane;
         group < gate.ng;
         group += NWARPS * 32) {
        uint32_t gate_state = 0;
        uint32_t up_state = 0;
        int gate_dot = 0;
        int up_dot = 0;
        aligned_group_pair_dot<FORMAT>(
            gate, up, row, group, qx,
            gate_state, up_state, gate_dot, up_dot);
        const float activation_scale = xscale[group];
        gate_acc = fmaf(
            format_scale<FORMAT>(gate_anchor, gate_state, gate.codebook) *
                activation_scale,
            static_cast<float>(gate_dot), gate_acc);
        up_acc = fmaf(
            format_scale<FORMAT>(up_anchor, up_state, up.codebook) *
                activation_scale,
            static_cast<float>(up_dot), up_acc);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
    }
    __shared__ float partial[2][NWARPS];
    if (lane == 0) {
        partial[0][warp] = gate_acc;
        partial[1][warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < NWARPS ? partial[0][lane] : 0.0f;
        up_acc = lane < NWARPS ? partial[1][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float value = up_value * (gate_value * sigmoid);
            if constexpr (OUTPUT_F32) output_f[row] = value;
            else output_h[row] = __float2half(value);
        }
    }
}

// Decode a group once; its three integer segment dots share the scale/state.
// Four warps cover a long projection, while short rows use independent warps.
template<int FORMAT, int Rows, int Warps>
__global__ void __launch_bounds__(128) nvq_gemv_group_kernel(
    NvqDeviceWeight w, const int8_t* qx, const float* xscale,
    half* output, int M) {
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int row = blockIdx.x * (4 / Warps) + warp / Warps;
    const bool valid_row = row < w.N;
    float acc[Rows] = {};
    const float anchor = valid_row ? w.neuron_scale[row] : 0.0f;
    for (int group = valid_row ? (warp % Warps) * 32 + lane : w.ng;
         group < w.ng; group += Warps * 32) {
        const int64_t state_index = int64_t(row) * w.ng + group;
        const uint32_t state = load_nvq_state<FORMAT>(w.indices,w.indices_nbytes,w.sub_scale,
            w.sub_scale_nbytes,state_index,w.sub_bits,w.ng);
        const float scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
            ? anchor * float(state) : format_scale<FORMAT>(anchor, state, w.codebook);
        uint64_t group_indices = 0;
        uint32_t group_signs = 0;
        bool group_words = false;
        if constexpr (FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL) {
            constexpr int bits = format_index_bits(FORMAT);
            const int64_t bit = (int64_t(row) * w.nvec + group * 3) * bits;
            const int64_t byte = (bit >> 3) & ~int64_t(3);
            // Three 10/12-bit indices always fit these two aligned words.
            // The final unpadded bytes retain the checked scalar decoder.
            group_words = (reinterpret_cast<uintptr_t>(w.indices) & 3u) == 0 && byte + 8 <= w.indices_nbytes;
            if (group_words) {
                const auto* words = reinterpret_cast<const uint32_t*>(w.indices + byte);
                group_indices = (uint64_t(words[0]) | (uint64_t(words[1]) << 32)) >> (bit & 31);
                group_signs = load_packed_sign_group3(w.aux,
                    (int64_t(row) * w.nsign + group * 3) * 7, w.aux_nbytes);
            }
        }
        float dot[Rows] = {};
#pragma unroll
        for (int segment = 0; segment < 3; ++segment) {
            const int vector = group * 3 + segment;
            NvqVec8Values<FORMAT> weight;
            if ((FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL) && group_words) {
                constexpr int bits = format_index_bits(FORMAT);
                const uint32_t index = (group_indices >> (segment * bits)) & ((1u << bits) - 1u);
                const uint32_t mask7 = (group_signs >> (segment * 7)) & 127u;
                const uint32_t last = parity7(mask7) ^ (w.sign_mode ? ((index >> 7) & 1u) : 0u);
                const uint32_t mask8 = mask7 | (last << 7);
                const auto* bank = active_codebook<FORMAT>(w.codebook, state);
                weight = {apply_sign8(reinterpret_cast<const int2*>(bank)[index], mask8), 0, vector < w.nvec};
            } else {
                weight = load_nvq_vec8<FORMAT>(
                    w.indices, w.indices_nbytes, w.aux, w.aux_nbytes, w.codebook,
                    row, vector, group, w.ng, w.nvec, w.nsign, w.sign_mode, state);
            }
#pragma unroll
            for (int m = 0; m < Rows; ++m)
                if (m < M) dot[m] += dot_nvq_vec8<FORMAT>(weight,
                    qx + (int64_t(m) * w.ng + group) * kGroupSize + segment * 8);
        }
#pragma unroll
        for (int m = 0; m < Rows; ++m)
            if (m < M) acc[m] = fmaf(scale * xscale[int64_t(m) * w.ng + group], dot[m], acc[m]);
    }
#pragma unroll
    for (int m = 0; m < Rows; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], offset);
    }
    if constexpr (Warps == 1) {
#pragma unroll
        for (int m = 0; m < Rows; ++m)
            if (lane == 0 && valid_row && m < M)
                output[int64_t(m) * w.N + row] = __float2half(acc[m]);
    } else {
        __shared__ float partial[Rows][4];
#pragma unroll
        for (int m = 0; m < Rows; ++m)
            if (lane == 0) partial[m][warp] = acc[m];
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int m = 0; m < Rows; ++m) {
                float value = lane < Warps ? partial[m][lane] : 0.0f;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    value += __shfl_xor_sync(0xffffffffu, value, offset);
                if (lane == 0 && valid_row && m < M)
                    output[int64_t(m) * w.N + row] = __float2half(value);
            }
        }
    }
}

template <int FORMAT, int NWARPS, int MAX_M>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_batch_vec8_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
    for (int segment = warp * 32 + lane; segment < nsign; segment += NWARPS * 32) {
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
        const auto weight = load_nvq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, segment, group, ng, nvec, nsign, sign_mode, sub);
        const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
            ? neuron_scale[row] * static_cast<float>(sub)
            : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float dot = dot_nvq_vec8<FORMAT>(
                    weight, qx + (static_cast<int64_t>(m) * ng * kGroupSize + k));
                acc[m] = fmaf(
                    weight_scale * xscale[static_cast<int64_t>(m) * ng + group], dot, acc[m]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        }
    }
    __shared__ float partial[MAX_M][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) partial[m][warp] = acc[m];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            acc[m] = lane < NWARPS ? partial[m][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
            }
            if (lane == 0 && m < M) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[m]);
            }
        }
    }
}

// One grid covers two projections with independent packed streams/codebooks.
// The activation is quantized once before this kernel.
template <int FORMAT, int NWARPS, int MAX_M, bool FAST_NVQ2_S4 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_multi2_vec8_kernel(
    NvqDeviceWeight first,
    NvqDeviceWeight second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M) {
    const int combined_row = blockIdx.x;
    const bool use_second = combined_row >= first.N;
    const NvqDeviceWeight weight = use_second ? second : first;
    const int row = use_second ? combined_row - first.N : combined_row;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= weight.N) return;

    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
    [[maybe_unused]] const int64_t fast_index_row = static_cast<int64_t>(row) * weight.nvec;
    [[maybe_unused]] const int64_t fast_sign_row = static_cast<int64_t>(row) * weight.nsign;
    [[maybe_unused]] const int64_t fast_sub_row = static_cast<int64_t>(row) * weight.ng;
    [[maybe_unused]] const float fast_anchor = weight.neuron_scale[row];
    for (int segment = warp * 32 + lane;
         segment < weight.nsign;
         segment += NWARPS * 32) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const int8_t * qrow = qx + static_cast<int64_t>(m) * weight.ng * kGroupSize;
                const float * srow = xscale + static_cast<int64_t>(m) * weight.ng;
                if constexpr (FAST_NVQ2_S4) {
                    acc[m] = nvq2_vec8_scaled_fma_s4<
                        FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec,
                        FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec>(
                        weight, qrow, srow,
                        fast_index_row, fast_sign_row, fast_sub_row, fast_anchor,
                        segment, acc[m]);
                } else {
                    acc[m] = nvq_vec8_scaled_fma<FORMAT>(
                        weight, qrow, srow, row, segment, acc[m]);
                }
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        }
    }
    __shared__ float partial[MAX_M][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) partial[m][warp] = acc[m];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            acc[m] = lane < NWARPS ? partial[m][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
            }
            if (lane == 0 && m < M) {
                output[static_cast<int64_t>(m) * (first.N + second.N) + combined_row] =
                    __float2half(acc[m]);
            }
        }
    }
}

// NINT-style paired projection: each block computes matching gate/up rows and
// applies SwiGLU after the same fp16 rounding as the materialized path.
template <int FORMAT, int NWARPS, bool OUTPUT_F32 = false, bool FAST_NVQ2_S4 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq_gemv_swiglu_pair_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * qx,
    const float * xscale,
    __half * output_h,
    float * output_f) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= gate.N) return;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    [[maybe_unused]] const int64_t fast_index_row = static_cast<int64_t>(row) * gate.nvec;
    [[maybe_unused]] const int64_t fast_sign_row = static_cast<int64_t>(row) * gate.nsign;
    [[maybe_unused]] const int64_t fast_sub_row = static_cast<int64_t>(row) * gate.ng;
    const float fast_gate_anchor = gate.neuron_scale[row];
    const float fast_up_anchor = up.neuron_scale[row];
    for (int segment = warp * 32 + lane;
        segment < gate.nsign;
         segment += NWARPS * 32) {
        if constexpr (FAST_NVQ2_S4) {
            nvq2_vec8_pair_scaled_fma_s4<
                FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec,
                FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec>(
                gate, up, qx, xscale,
                fast_index_row, fast_sign_row, fast_sub_row,
                fast_gate_anchor, fast_up_anchor, segment, gate_acc, up_acc);
        } else {
            gate_acc = nvq_vec8_scaled_fma<FORMAT>(
                gate, qx, xscale, row, segment, gate_acc);
            up_acc = nvq_vec8_scaled_fma<FORMAT>(
                up, qx, xscale, row, segment, up_acc);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
    }
    __shared__ float partial[2][NWARPS];
    if (lane == 0) {
        partial[0][warp] = gate_acc;
        partial[1][warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < NWARPS ? partial[0][lane] : 0.0f;
        up_acc = lane < NWARPS ? partial[1][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float value = up_value * (gate_value * sigmoid);
            if constexpr (OUTPUT_F32) {
                output_f[row] = value;
            } else {
                output_h[row] = __float2half(value);
            }
        }
    }
}

__device__ __forceinline__ uint32_t load_u8x4(const uint8_t * data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

// Decode four adjacent vectors per thread, then replay the original MMVQ FMA
// order from shared memory. This keeps the packed-load savings without changing
// the quantized activation produced for Wdown.
template <int NWARPS, bool OUTPUT_F32 = false>
__global__ void __launch_bounds__(NWARPS * 32, 1) nvq2_gemv_swiglu_vec4_ordered_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * qx,
    const float * xscale,
    __half * output_h,
    float * output_f) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int thread = warp * 32 + lane;
    if (row >= gate.N) return;

    extern __shared__ uint32_t metadata[];
    const int64_t index_row = static_cast<int64_t>(row) * gate.nvec;
    const int64_t sign_row = static_cast<int64_t>(row) * gate.nsign;
    const int units = gate.nsign / 4;
    for (int unit = thread; unit < units; unit += NWARPS * 32) {
        const int segment_base = unit * 4;
        const uint32_t gate_indices = *reinterpret_cast<const uint32_t *>(
            gate.indices + index_row + segment_base);
        const uint32_t up_indices = *reinterpret_cast<const uint32_t *>(
            up.indices + index_row + segment_base);
        const int64_t sign_bit = (sign_row + segment_base) * 7;
        const int64_t sign_byte = sign_bit >> 3;
        const int sign_shift = static_cast<int>(sign_bit & 7);
        const uint32_t gate_signs = load_u8x4(gate.aux + sign_byte) >> sign_shift;
        const uint32_t up_signs = load_u8x4(up.aux + sign_byte) >> sign_shift;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t gate_index = (gate_indices >> (8 * i)) & 0xffu;
            const uint32_t up_index = (up_indices >> (8 * i)) & 0xffu;
            const uint32_t gate_mask7 = (gate_signs >> (7 * i)) & 0x7fu;
            const uint32_t up_mask7 = (up_signs >> (7 * i)) & 0x7fu;
            metadata[segment_base + i] = gate_index | (up_index << 8) |
                (gate_mask7 << 16) | (up_mask7 << 23);
        }
    }
    __syncthreads();

    const int64_t sub_row = static_cast<int64_t>(row) * gate.ng;
    const float gate_anchor = gate.neuron_scale[row];
    const float up_anchor = up.neuron_scale[row];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int segment = thread; segment < gate.nsign; segment += NWARPS * 32) {
        const uint32_t meta = metadata[segment];
        const uint32_t gate_index = meta & 0xffu;
        const uint32_t up_index = (meta >> 8) & 0xffu;
        const uint32_t gate_mask7 = (meta >> 16) & 0x7fu;
        const uint32_t up_mask7 = (meta >> 23) & 0x7fu;
        const int gate_last = parity7(gate_mask7) ^
            (gate.sign_mode ? ((gate_index >> 7) & 1u) : 0u);
        const int up_last = parity7(up_mask7) ^
            (up.sign_mode ? ((up_index >> 7) & 1u) : 0u);
        const int2 gate_values = apply_sign8(
            reinterpret_cast<const int2 *>(gate.codebook)[gate_index],
            gate_mask7 | (static_cast<uint32_t>(gate_last) << 7));
        const int2 up_values = apply_sign8(
            reinterpret_cast<const int2 *>(up.codebook)[up_index],
            up_mask7 | (static_cast<uint32_t>(up_last) << 7));
        const int2 x = *reinterpret_cast<const int2 *>(qx + segment * 8);
        const int gate_dot = __dp4a(
            gate_values.y, x.y, __dp4a(gate_values.x, x.x, 0));
        const int up_dot = __dp4a(
            up_values.y, x.y, __dp4a(up_values.x, x.x, 0));
        const int group = segment / 3;
        const float activation_scale = xscale[group];
        const uint32_t gate_sub = load_packed_4(gate.sub_scale, sub_row + group);
        const uint32_t up_sub = load_packed_4(up.sub_scale, sub_row + group);
        const float gate_scale = gate_anchor * static_cast<float>(gate_sub);
        const float up_scale = up_anchor * static_cast<float>(up_sub);
        gate_acc = fmaf(
            gate_scale * activation_scale, static_cast<float>(gate_dot), gate_acc);
        up_acc = fmaf(
            up_scale * activation_scale, static_cast<float>(up_dot), up_acc);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
    }
    __shared__ float partial[2][NWARPS];
    if (lane == 0) {
        partial[0][warp] = gate_acc;
        partial[1][warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < NWARPS ? partial[0][lane] : 0.0f;
        up_acc = lane < NWARPS ? partial[1][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            const float value = up_value * (gate_value * sigmoid);
            if constexpr (OUTPUT_F32) {
                output_f[row] = value;
            } else {
                output_h[row] = __float2half(value);
            }
        }
    }
}

template <int GS>
__global__ void nvq_quantize_f32_kernel(
    const float * input,
    int8_t * output_qx,
    float * output_xscale,
    int K,
    int K_pad) {
    const int group = blockIdx.x;
    const int lane = threadIdx.x;
    const int k = group * GS + lane;
    const bool valid = lane < GS && k < K;
    const float value = valid ? input[k] : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int quant = 0;
    if (valid) {
        quant = static_cast<int>(roundf(value / scale));
        quant = quant < -127 ? -127 : (quant > 127 ? 127 : quant);
    }
    if (lane == 0) output_xscale[group] = scale;
    if (lane < GS && k < K_pad) output_qx[k] = static_cast<int8_t>(quant);
}

// The aligned group layout is faster when gate/up rows remain independent:
// 2N blocks expose both projections to the scheduler and keep the GEMV at the
// lower single-projection register count.  The existing N-float workspace is
// exactly large enough for the two fp16 projections, so this adds no workspace.
template <int GS>
__global__ void nvq_swiglu_quantize_f16_pair_kernel(
    const __half * gate_up,
    int8_t * output_qx,
    float * output_xscale,
    int K,
    int K_pad) {
    const int group = blockIdx.x;
    const int lane = threadIdx.x;
    if (group * GS >= K_pad) return;
    const int k = group * GS + lane;
    const bool valid = lane < GS && k < K;
    float value = 0.0f;
    if (valid) {
        const float gate_value = __half2float(gate_up[k]);
        const float up_value = __half2float(gate_up[K + k]);
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        value = up_value * (gate_value * sigmoid);
    }
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    int quant = 0;
    if (valid) {
        quant = static_cast<int>(roundf(value / scale));
        quant = quant < -127 ? -127 : (quant > 127 ? 127 : quant);
    }
    if (lane == 0) output_xscale[group] = scale;
    if (lane < GS && k < K_pad) output_qx[k] = static_cast<int8_t>(quant);
}

// Computes gate/up, applies SwiGLU, and writes the q8 activation layout used
// directly by the following NINT/NVQ down projection.
template <int FORMAT, int DOWN_GS>
__global__ void __launch_bounds__(1024, 1) nvq_ffn_swiglu_quant_kernel(
    NvqDeviceWeight gate,
    NvqDeviceWeight up,
    const int8_t * input_qx,
    const float * input_xscale,
    int8_t * output_qx,
    float * output_xscale,
    int down_k_pad) {
    const int output_group = blockIdx.x;
    const int element = threadIdx.y;
    const int lane = threadIdx.x;
    const int row = output_group * DOWN_GS + element;

    __shared__ float values[DOWN_GS];
    __shared__ float scale;

    float value = 0.0f;
    if (row < gate.N) {
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (int segment = lane; segment < gate.nsign; segment += 32) {
            gate_acc = nvq_vec8_scaled_fma<FORMAT>(
                gate, input_qx, input_xscale, row, segment, gate_acc);
            up_acc = nvq_vec8_scaled_fma<FORMAT>(
                up, input_qx, input_xscale, row, segment, up_acc);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xffffffff, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xffffffff, up_acc, offset);
        }
        if (lane == 0) {
            const float gate_value = __half2float(__float2half(gate_acc));
            const float up_value = __half2float(__float2half(up_acc));
            const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
            value = up_value * (gate_value * sigmoid);
        }
    }
    if (lane == 0) values[element] = value;
    __syncthreads();

    if (element == 0) {
        float amax = lane < DOWN_GS ? fabsf(values[lane]) : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
        }
        if (lane == 0) {
            scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            output_xscale[output_group] = scale;
        }
    }
    __syncthreads();

    if (lane == 0) {
        int quant = 0;
        if (row < gate.N) {
            quant = static_cast<int>(roundf(values[element] / scale));
            quant = max(-127, min(127, quant));
        }
        const int offset = output_group * DOWN_GS + element;
        if (offset < down_k_pad) output_qx[offset] = static_cast<int8_t>(quant);
    }
}


__device__ __forceinline__ int mma168_i(int fragment) {
    return ((fragment / 2) * 8) + (threadIdx.x / 4);
}

__device__ __forceinline__ int mma168_j(int fragment) {
    return ((threadIdx.x % 4) * 2) + (fragment % 2);
}

__device__ __forceinline__ void load_mma_a_m16n8k32(int (&a)[4], const int * ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "l"(ptr));
}

__device__ __forceinline__ void load_mma_b_m16n8k32(int (&b)[2], const int * ptr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(b[0]), "=r"(b[1])
        : "l"(ptr));
}

__device__ __forceinline__ void mma_m16n8k32_s8(
    int (&d)[4], const int (&a)[4], const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nvq_mmq_mma24_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    constexpr int kChunkGroups = 8;
    constexpr int kGroupInts = 8;
    constexpr int kStride = kChunkGroups * kGroupInts + 4;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kRowsPerWarp = 8;
    constexpr int kWarps = 8;
    constexpr int kTileN = kRowsPerWarp * kWarps;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;

    __shared__ int activation[kTileM][kStride];
    __shared__ int weight[kWarps][kRowsPerWarp][kStride];
    __shared__ __half weight_scale[kChunkGroups][kTileN];
    __shared__ float activation_scale[kChunkGroups][kTileM];

    float acc[MTILES][4];
#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) acc[mt][fragment] = 0.0f;
    }

    for (int group_base = 0; group_base < ng; group_base += kChunkGroups) {
        const int active_groups = min(kChunkGroups, ng - group_base);

        for (int index = tid; index < kTileM * kChunkGroups * kGroupInts;
             index += kWarps * 32) {
            const int m_local = index / (kChunkGroups * kGroupInts);
            const int rest = index - m_local * kChunkGroups * kGroupInts;
            const int group_local = rest / kGroupInts;
            const int quartet = rest - group_local * kGroupInts;
            const int m = m0 + m_local;
            int value = 0;
            if (group_local < active_groups && quartet < kChunksPerGroup &&
                (!NEED_CHECK || m < M)) {
                const int group = group_base + group_local;
                value = load_i8x4(
                    qx + (static_cast<int64_t>(m) * ng + group) * kGroupSize + quartet * 4);
            }
            activation[m_local][group_local * kGroupInts + quartet] = value;
        }

        constexpr int kWeightTasks = kRowsPerWarp * kChunkGroups;
        for (int task = lane; task < kWeightTasks; task += 32) {
            const int row_local = task / kChunkGroups;
            const int group_local = task - row_local * kChunkGroups;
            const int row = n0 + warp * kRowsPerWarp + row_local;
            const int group = group_base + group_local;
            const bool valid = group_local < active_groups && (!NEED_CHECK || row < N);
            uint32_t sub = 0;
            if (valid) {
                const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
                sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            }
#pragma unroll
            for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
                weight[warp][row_local][group_local * kGroupInts + quartet] = valid
                    ? decode_chunk4<FORMAT>(
                          indices, indices_nbytes, aux, aux_nbytes, codebook,
                          row, group, quartet, nvec, nsign, ng, sign_mode, sub)
                    : 0;
            }
            weight[warp][row_local][group_local * kGroupInts + 6] = 0;
            weight[warp][row_local][group_local * kGroupInts + 7] = 0;
            float scale = 0.0f;
            if (valid) {
                scale = format_scale<FORMAT>(neuron_scale[row], sub, codebook);
            }
            weight_scale[group_local][warp * kRowsPerWarp + row_local] = __float2half(scale);
        }

        for (int index = tid; index < kTileM * kChunkGroups; index += kWarps * 32) {
            const int m_local = index / kChunkGroups;
            const int group_local = index - m_local * kChunkGroups;
            const int m = m0 + m_local;
            activation_scale[group_local][m_local] =
                group_local < active_groups && (!NEED_CHECK || m < M)
                ? xscale[static_cast<int64_t>(m) * ng + group_base + group_local]
                : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int group_local = 0; group_local < kChunkGroups; ++group_local) {
            int a[MTILES][4];
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                load_mma_a_m16n8k32(
                    a[mt],
                    &activation[mt * 16 + lane % 16]
                               [group_local * kGroupInts + (lane / 16) * 4]);
            }
            int b[2];
            load_mma_b_m16n8k32(
                b,
                &weight[warp][lane % 8]
                       [group_local * kGroupInts + (((lane / 8) * 4) & 7)]);
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                int dot[4] = {0, 0, 0, 0};
                mma_m16n8k32_s8(dot, a[mt], b);
#pragma unroll
                for (int fragment = 0; fragment < 4; ++fragment) {
                    const int m_local = mt * 16 + mma168_i(fragment);
                    const int row_local = warp * kRowsPerWarp + mma168_j(fragment);
                    const int m = m0 + m_local;
                    const int row = n0 + row_local;
                    if (!NEED_CHECK || (m < M && row < N)) {
                        const float scale = activation_scale[group_local][m_local]
                            * __half2float(weight_scale[group_local][row_local]);
                        acc[mt][fragment] = fmaf(scale, static_cast<float>(dot[fragment]),
                                                acc[mt][fragment]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) {
            const int m = m0 + mt * 16 + mma168_i(fragment);
            const int row = n0 + warp * kRowsPerWarp + mma168_j(fragment);
            if (!NEED_CHECK || (m < M && row < N)) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[mt][fragment]);
            }
        }
    }
}

template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nvq_gemm_f16_gs24_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const __half * x,
    __half * output,
    int M,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    constexpr int kGroupsPerChunk = 4;
    constexpr int kTileK = kGroupSize * kGroupsPerChunk;
    constexpr int kStrideK = kTileK + 8;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kTileN = 64;
    constexpr int kWarps = 8;
    constexpr int kWmmaM = 16;
    constexpr int kWmmaN = 16;
    constexpr int kWmmaK = 16;
    constexpr int kNFragments = kTileN / kWmmaN;
    constexpr int kAccumulatorsPerWarp = (MTILES + 1) / 2;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;
    const int warp_m0 = warp / kNFragments;
    const int warp_n = warp % kNFragments;

    __shared__ __half weight_tile[kTileN][kStrideK];
    __shared__ __half activation_tile[kTileM][kStrideK];
    __shared__ float output_tile[kNFragments][kWmmaM][kWmmaN];

    using FragmentA = wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>;
    FragmentC accumulators[kAccumulatorsPerWarp];
#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulatorsPerWarp; ++accumulator) {
        wmma::fill_fragment(accumulators[accumulator], 0.0f);
    }

    const int chunks = (ng + kGroupsPerChunk - 1) / kGroupsPerChunk;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int group_base = chunk * kGroupsPerChunk;
        const int k_base = group_base * kGroupSize;

        // There are exactly 64 * 4 row/group tasks, one per block thread.
        const int row_local = tid / kGroupsPerChunk;
        const int group_local = tid - row_local * kGroupsPerChunk;
        const int row = n0 + row_local;
        const int group = group_base + group_local;
        const bool valid_weight = group < ng && (!NEED_CHECK || row < N);
        uint32_t state = 0;
        float scale = 0.0f;
        if (valid_weight) {
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            scale = format_scale<FORMAT>(neuron_scale[row], state, codebook);
        }
#pragma unroll
        for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
            const int packed = valid_weight
                ? decode_chunk4<FORMAT>(
                      indices, indices_nbytes, aux, aux_nbytes, codebook,
                      row, group, quartet, nvec, nsign, ng, sign_mode, state)
                : 0;
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int shift = pair * 16;
                const int value0 = static_cast<int>(
                    static_cast<int8_t>((packed >> shift) & 0xff));
                const int value1 = static_cast<int>(
                    static_cast<int8_t>((packed >> (shift + 8)) & 0xff));
                const __half2 values = __halves2half2(
                    __float2half(scale * static_cast<float>(value0)),
                    __float2half(scale * static_cast<float>(value1)));
                *reinterpret_cast<__half2 *>(
                    &weight_tile[row_local]
                                [group_local * kGroupSize + quartet * 4 + pair * 2]) = values;
            }
        }

        constexpr int kActivationPairs = kTileM * (kTileK / 2);
        for (int index = tid; index < kActivationPairs; index += kWarps * 32) {
            const int m_local = index / (kTileK / 2);
            const int pair = index - m_local * (kTileK / 2);
            const int m = m0 + m_local;
            const int k = k_base + pair * 2;
            __half2 values = __float2half2_rn(0.0f);
            if ((!NEED_CHECK || m < M) && k + 1 < K && (K & 1) == 0) {
                values = *reinterpret_cast<const __half2 *>(
                    x + static_cast<int64_t>(m) * K + k);
            } else if ((!NEED_CHECK || m < M) && k < K) {
                const __half value0 = x[static_cast<int64_t>(m) * K + k];
                const __half value1 = k + 1 < K
                    ? x[static_cast<int64_t>(m) * K + k + 1]
                    : __float2half(0.0f);
                values = __halves2half2(value0, value1);
            }
            *reinterpret_cast<__half2 *>(
                &activation_tile[m_local][pair * 2]) = values;
        }
        __syncthreads();

        const bool warp_active = warp_m0 < MTILES;
#pragma unroll
        for (int k_local = 0; k_local < kTileK; k_local += kWmmaK) {
            if (warp_active) {
                FragmentB weight_fragment;
                wmma::load_matrix_sync(
                    weight_fragment,
                    &weight_tile[warp_n * kWmmaN][k_local],
                    kStrideK);
#pragma unroll
                for (int accumulator = 0;
                     accumulator < kAccumulatorsPerWarp;
                     ++accumulator) {
                    const int m_fragment = warp_m0 + accumulator * 2;
                    if (m_fragment < MTILES) {
                        FragmentA activation_fragment;
                        wmma::load_matrix_sync(
                            activation_fragment,
                            &activation_tile[m_fragment * kWmmaM][k_local],
                            kStrideK);
                        wmma::mma_sync(
                            accumulators[accumulator], activation_fragment,
                            weight_fragment, accumulators[accumulator]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int accumulator = 0;
         accumulator < kAccumulatorsPerWarp;
         ++accumulator) {
#pragma unroll
        for (int warp_row = 0; warp_row < 2; ++warp_row) {
            const int m_fragment = warp_m0 + accumulator * 2;
            const bool owns_fragment = warp_m0 == warp_row && m_fragment < MTILES;
            if (owns_fragment) {
                wmma::store_matrix_sync(
                    &output_tile[warp_n][0][0], accumulators[accumulator],
                    kWmmaN, wmma::mem_row_major);
            }
            __syncthreads();
            if (owns_fragment) {
                const int output_m0 = m0 + m_fragment * kWmmaM;
                const int output_n0 = n0 + warp_n * kWmmaN;
#pragma unroll
                for (int element = lane; element < kWmmaM * kWmmaN; element += 32) {
                    const int local_m = element / kWmmaN;
                    const int local_n = element - local_m * kWmmaN;
                    const int m = output_m0 + local_m;
                    const int n = output_n0 + local_n;
                    if (!NEED_CHECK || (m < M && n < N)) {
                        output[static_cast<int64_t>(m) * N + n] =
                            __float2half(output_tile[warp_n][local_m][local_n]);
                    }
                }
            }
            __syncthreads();
        }
    }
}
