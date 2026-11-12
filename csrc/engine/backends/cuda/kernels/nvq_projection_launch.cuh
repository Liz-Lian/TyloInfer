#pragma once
int aligned_group_rows_per_block(int format) {
    const char * value = tyloi::cuda::env_lookup(
        format == kNvq2JscXLGroupExec
            ? "TYLOINFER_NVQ_GROUP_ROWS_E8"
            : "TYLOINFER_NVQ_GROUP_ROWS_D4");
    if (value == nullptr) value = tyloi::cuda::env_lookup("TYLOINFER_NVQ_GROUP_ROWS");
    if (value != nullptr && value[0] == '4') return 4;
    if (value != nullptr && value[0] == '2') return 2;
    return 1;
}

int aligned_group_row_tile(int format) {
    const char * value = tyloi::cuda::env_lookup(
        format == kNvq2JscXLGroupExec
            ? "TYLOINFER_NVQ_ROW_TILE_E8"
            : "TYLOINFER_NVQ_ROW_TILE_D4");
    if (value == nullptr) value = tyloi::cuda::env_lookup("TYLOINFER_NVQ_ROW_TILE");
    return value == nullptr ? 0 : std::atoi(value);
}

bool aligned_group_e8_down_w32r1_enabled() {
    static const bool value = [] {
        const char * text = tyloi::cuda::env_lookup("TYLOINFER_NVQ_E8_STAGE_DOWN_W32R1");
        return text == nullptr || text[0] != '0';
    }();
    return value;
}

template <
    int FORMAT, int NWARPS, int ROWS_PER_WARP, bool MULTI2,
    bool STAGE_E8_THREE_BANKS = false, bool ADD_RESIDUAL = false>
void launch_aligned_group_row_tiled_m1(
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    cudaStream_t stream,
    const __half * residual = nullptr) {
    static_assert(!ADD_RESIDUAL || !MULTI2);
    const int total_rows = MULTI2 ? first.N + second.N : first.N;
    const int rows_per_block = NWARPS * ROWS_PER_WARP;
    size_t shared_bytes = 0;
    if constexpr (STAGE_E8_THREE_BANKS) {
        shared_bytes =
            3 * kJscE8Codebook4096Bytes + kJscE8PartialBytes;
        static const cudaError_t attribute_status = cudaFuncSetAttribute(
            aligned_group_row_tiled_m1_kernel<
                FORMAT, NWARPS, ROWS_PER_WARP, MULTI2, true, ADD_RESIDUAL>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        if (attribute_status != cudaSuccess)
            throw std::runtime_error("failed to opt in to the E8 row-tiled shared-memory size");
    } else {
        const int k_pad = first.ng * kGroupSize;
        const int scale_offset = (k_pad + 3) & ~3;
        shared_bytes = static_cast<size_t>(scale_offset) +
            static_cast<size_t>(first.ng) * sizeof(float);
        if constexpr (FORMAT == kNvq3JscLGroupExec) {
            shared_bytes += static_cast<size_t>(first.codebook_nbytes);
            if constexpr (MULTI2) {
                shared_bytes += static_cast<size_t>(second.codebook_nbytes);
            }
        }
    }
    aligned_group_row_tiled_m1_kernel<
        FORMAT, NWARPS, ROWS_PER_WARP, MULTI2,
        STAGE_E8_THREE_BANKS, ADD_RESIDUAL>
        <<<(total_rows + rows_per_block - 1) / rows_per_block,
           dim3(32, NWARPS), shared_bytes, stream>>>(
            first, second, qx, xscale, output, residual);
}

template <int FORMAT, bool ADD_RESIDUAL = false>
void launch_aligned_group_m1(
    const NvqDeviceWeight & weight,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    cudaStream_t stream,
    const __half * residual = nullptr) {
    const int tile = aligned_group_row_tile(FORMAT);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        if (tile == 3243 && weight.codebook_nbytes >=
                kJscMetadataBytes + 4 * kJscE8Codebook4096Bytes) {
            if (weight.N >= 12288) {
                launch_aligned_group_row_tiled_m1<
                    FORMAT, 32, 4, false, true, ADD_RESIDUAL>(
                    weight, weight, qx, xscale, output, stream, residual);
                return;
            }
            if (weight.N >= 4096 && aligned_group_e8_down_w32r1_enabled()) {
                launch_aligned_group_row_tiled_m1<
                    FORMAT, 32, 1, false, true, ADD_RESIDUAL>(
                    weight, weight, qx, xscale, output, stream, residual);
                return;
            }
        }
        if (tile == 164) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 16, 4, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
        if (tile == 324) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 32, 4, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
    } else {
        if (tile == 162) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 16, 2, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
        if (tile == 322) {
            launch_aligned_group_row_tiled_m1<
                FORMAT, 32, 2, false, false, ADD_RESIDUAL>(
                weight, weight, qx, xscale, output, stream, residual);
            return;
        }
    }
    const int rows = aligned_group_rows_per_block(FORMAT);
    if (rows == 4) {
        aligned_group_gemv_m1_kernel<FORMAT, 4, 4, ADD_RESIDUAL>
            <<<(weight.N + 3) / 4, dim3(32, 16), 0, stream>>>(
                weight, qx, xscale, output, residual);
    } else if (rows == 2) {
        aligned_group_gemv_m1_kernel<FORMAT, 4, 2, ADD_RESIDUAL>
            <<<(weight.N + 1) / 2, dim3(32, 8), 0, stream>>>(
                weight, qx, xscale, output, residual);
    } else {
        aligned_group_gemv_m1_kernel<FORMAT, 4, 1, ADD_RESIDUAL>
            <<<weight.N, dim3(32, 4), 0, stream>>>(
                weight, qx, xscale, output, residual);
    }
}

template <int FORMAT>
void launch_aligned_group_multi2_m1(
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    cudaStream_t stream) {
    const int blocks = first.N + second.N;
    const int tile = aligned_group_row_tile(FORMAT);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        if (tile == 3243 &&
            (first.N >= 12288 || second.N >= 12288) &&
            first.codebook_nbytes >=
                kJscMetadataBytes + 4 * kJscE8Codebook4096Bytes &&
            second.codebook_nbytes >=
                kJscMetadataBytes + 4 * kJscE8Codebook4096Bytes) {
            launch_aligned_group_row_tiled_m1<FORMAT, 32, 4, true, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
        if (tile == 164) {
            launch_aligned_group_row_tiled_m1<FORMAT, 16, 4, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
        if (tile == 324) {
            launch_aligned_group_row_tiled_m1<FORMAT, 32, 4, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
    } else {
        if (tile == 162) {
            launch_aligned_group_row_tiled_m1<FORMAT, 16, 2, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
        if (tile == 322) {
            launch_aligned_group_row_tiled_m1<FORMAT, 32, 2, true>(
                first, second, qx, xscale, output, stream);
            return;
        }
    }
    const int rows = aligned_group_rows_per_block(FORMAT);
    if (rows == 4) {
        aligned_group_multi2_m1_kernel<FORMAT, 4, 4>
            <<<(blocks + 3) / 4, dim3(32, 16), 0, stream>>>(
                first, second, qx, xscale, output);
    } else if (rows == 2) {
        aligned_group_multi2_m1_kernel<FORMAT, 4, 2>
            <<<(blocks + 1) / 2, dim3(32, 8), 0, stream>>>(
                first, second, qx, xscale, output);
    } else {
        aligned_group_multi2_m1_kernel<FORMAT, 4, 1>
            <<<blocks, dim3(32, 4), 0, stream>>>(
                first, second, qx, xscale, output);
    }
}
