#pragma once
template <int NWARPS>
void launch_m1_vec8_by_format(
    int format, const Projection& a,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
    if (format == kNvq2 && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            a.weight.indices, a.weight.indices_bytes, a.weight.aux, a.weight.aux_bytes,
            a.weight.state, a.weight.state_bytes, a.weight.scale,
            a.weight.table, a.quantized, a.scales,
            static_cast<__half *>(a.output),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq2Exec && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2Exec, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            a.weight.indices, a.weight.indices_bytes, a.weight.aux, a.weight.aux_bytes,
            a.weight.state, a.weight.state_bytes, a.weight.scale,
            a.weight.table, a.quantized, a.scales,
            static_cast<__half *>(a.output),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq2Jsc && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2Jsc, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            a.weight.indices, a.weight.indices_bytes, a.weight.aux, a.weight.aux_bytes,
            a.weight.state, a.weight.state_bytes, a.weight.scale,
            a.weight.table, a.quantized, a.scales,
            static_cast<__half *>(a.output),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq2JscExec && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq2JscExec, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            a.weight.indices, a.weight.indices_bytes, a.weight.aux, a.weight.aux_bytes,
            a.weight.state, a.weight.state_bytes, a.weight.scale,
            a.weight.table, a.quantized, a.scales,
            static_cast<__half *>(a.output),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq3 && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq3, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            a.weight.indices, a.weight.indices_bytes, a.weight.aux, a.weight.aux_bytes,
            a.weight.state, a.weight.state_bytes, a.weight.scale,
            a.weight.table, a.quantized, a.scales,
            static_cast<__half *>(a.output),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq3Jsc && sub_bits == 4) {
        nvq_gemv_m1_vec8_kernel<kNvq3Jsc, NWARPS, true><<<N, dim3(32, NWARPS), 0, stream>>>(
            a.weight.indices, a.weight.indices_bytes, a.weight.aux, a.weight.aux_bytes,
            a.weight.state, a.weight.state_bytes, a.weight.scale,
            a.weight.table, a.quantized, a.scales,
            static_cast<__half *>(a.output),
            N, ng, nvec, nsign, sub_bits, sign_mode);
        return;
    }
    if (format == kNvq3Jsc2 && sub_bits == 4) {
        nvq3j2_gemv_m1_group_kernel<NWARPS><<<N, dim3(32, NWARPS), 0, stream>>>(
            a.weight.indices, a.weight.aux, a.weight.aux_bytes,
            a.weight.state, a.weight.scale,
            a.weight.table, a.quantized, a.scales,
            static_cast<__half *>(a.output),
            N, ng, nvec, nsign);
        return;
    }
    if (format == kNvq2JscXLGroupExec || format == kNvq3JscLGroupExec) {
        const NvqDeviceWeight weight{
            a.weight.indices, a.weight.indices_bytes,
            a.weight.aux, a.weight.aux_bytes,
            a.weight.state, a.weight.state_bytes,
            a.weight.scale, a.weight.table,
            static_cast<int>(a.weight.table_bytes),
            N, ng, nvec, nsign, sub_bits, sign_mode};
        if (format == kNvq2JscXLGroupExec) {
            launch_aligned_group_m1<kNvq2JscXLGroupExec>(
                weight, a.quantized, a.scales,
                static_cast<__half *>(a.output), stream);
        } else {
            launch_aligned_group_m1<kNvq3JscLGroupExec>(
                weight, a.quantized, a.scales,
                static_cast<__half *>(a.output), stream);
        }
        return;
    }
    launch_by_format(format, [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_gemv_m1_vec8_kernel<F, NWARPS><<<N, dim3(32, NWARPS), 0, stream>>>(
            a.weight.indices, a.weight.indices_bytes, a.weight.aux, a.weight.aux_bytes,
            a.weight.state, a.weight.state_bytes, a.weight.scale,
            a.weight.table, a.quantized, a.scales,
            static_cast<__half *>(a.output),
            N, ng, nvec, nsign, sub_bits, sign_mode);
    });
}

template <int NWARPS, int MAX_M>
void launch_batch_vec8_by_format(
    int format, const Projection& a,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
    launch_by_format(format, [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_gemv_batch_vec8_kernel<F, NWARPS, MAX_M><<<N, dim3(32, NWARPS), 0, stream>>>(
            a.weight.indices, a.weight.indices_bytes, a.weight.aux, a.weight.aux_bytes,
            a.weight.state, a.weight.state_bytes, a.weight.scale,
            a.weight.table, a.quantized, a.scales,
            static_cast<__half *>(a.output),
            M, N, ng, nvec, nsign, sub_bits, sign_mode);
    });
}

bool launch_group_vec8(
    int format, const Projection& a,
    int M, int N, int ng, int nvec, int nsign, int sub_bits, int sign_mode, cudaStream_t stream) {
    if (M > 4 || ng > 128 || N < 128 || N > 4096 ||
        (sub_bits != 4 && format != kNvq1L) || tyloi::cuda::env_lookup("TYLOINFER_NVQ_DENSE_REFERENCE")) return false;
    if (M == 1 && ng > 32 &&
        (format == kNvq2Jsc || format == kNvq3Jsc || format == kNvq3JscL)) return false;
    const NvqDeviceWeight weight{a.weight.indices, a.weight.indices_bytes,
        a.weight.aux, a.weight.aux_bytes, a.weight.state, a.weight.state_bytes,
        a.weight.scale, a.weight.table, int(a.weight.table_bytes),
        N, ng, nvec, nsign, sub_bits, sign_mode};
    launch_by_format(format, [&](auto tag) {
        const auto launch = [&](auto rows, auto warps) {
            constexpr int rows_per_block = 4 / decltype(warps)::value;
            nvq_gemv_group_kernel<decltype(tag)::value, decltype(rows)::value, decltype(warps)::value><<<
                (N + rows_per_block - 1) / rows_per_block, dim3(32, 4), 0, stream>>>(
                weight, a.quantized, a.scales,
                static_cast<half*>(a.output), M);
        };
        const auto select_warps = [&](auto rows) {
            if (ng <= 32) launch(rows, std::integral_constant<int, 1>{});
            else launch(rows, std::integral_constant<int, 4>{});
        };
        if (M == 1) select_warps(std::integral_constant<int, 1>{});
        else if (M <= 2) select_warps(std::integral_constant<int, 2>{});
        else select_warps(std::integral_constant<int, 4>{});
    });
    return true;
}

void launch_selected_batch_vec8(
    int format, const Projection& a,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
#define NVQ_SELECTED_BATCH(NW, MAX_M)                                                   \
    launch_batch_vec8_by_format<NW, MAX_M>(                                            \
        format, a,  \
        M, N, ng, nvec, nsign, sub_bits, sign_mode, stream)
    if (launch_group_vec8(format, a, M, N, ng, nvec, nsign, sub_bits, sign_mode, stream)) return;
    const bool compact_s = format == kNvq1S || format == kNpq0S;
    const bool compact_s_small = compact_s && N <= 1024;
    if (M <= 2) {
        if (compact_s_small) {
            NVQ_SELECTED_BATCH(8, 2);
        } else if (compact_s) {
            NVQ_SELECTED_BATCH(4, 2);
        } else if (is_e8_format(format)) {
            if (N <= 1024) {
                NVQ_SELECTED_BATCH(8, 2);
            } else {
                NVQ_SELECTED_BATCH(4, 2);
            }
        } else if (is_d4_format(format)) {
            NVQ_SELECTED_BATCH(2, 2);
        } else {
            NVQ_SELECTED_BATCH(8, 2);
        }
    } else if (M <= 4) {
        if (compact_s_small) {
            NVQ_SELECTED_BATCH(8, 4);
        } else if (compact_s) {
            NVQ_SELECTED_BATCH(4, 4);
        } else if (is_e8_format(format)) {
            if (N <= 1024) {
                NVQ_SELECTED_BATCH(8, 4);
            } else {
                NVQ_SELECTED_BATCH(4, 4);
            }
        } else if (is_d4_format(format)) {
            NVQ_SELECTED_BATCH(4, 4);
        } else {
            NVQ_SELECTED_BATCH(8, 4);
        }
    } else if (M <= 8 && compact_s && (compact_s_small || ng >= 384)) {
        NVQ_SELECTED_BATCH(8, 8);
    } else if (M <= 8 && compact_s) {
        NVQ_SELECTED_BATCH(4, 8);
    } else if (M <= 8 && is_e8_format(format)) {
        if (N <= 1024) {
            NVQ_SELECTED_BATCH(8, 8);
        } else {
            NVQ_SELECTED_BATCH(4, 8);
        }
    } else if (M <= 8) {
        NVQ_SELECTED_BATCH(8, 8);
    } else if (M <= 12 && compact_s_small) {
        NVQ_SELECTED_BATCH(8, 12);
    } else if (M <= 12 && compact_s) {
        NVQ_SELECTED_BATCH(4, 12);
    } else if (M <= 12 && is_e8_format(format)) {
        if (M == 12 && N <= 1024) {
            NVQ_SELECTED_BATCH(8, 12);
        } else {
            NVQ_SELECTED_BATCH(4, 12);
        }
    } else if (M <= 12) {
        NVQ_SELECTED_BATCH(8, 12);
    } else if (compact_s_small) {
        NVQ_SELECTED_BATCH(8, 16);
    } else if (compact_s) {
        NVQ_SELECTED_BATCH(4, 16);
    } else if (is_e8_format(format)) {
        if (N <= 1024) {
            NVQ_SELECTED_BATCH(8, 16);
        } else {
            NVQ_SELECTED_BATCH(4, 16);
        }
    } else {
        NVQ_SELECTED_BATCH(8, 16);
    }
#undef NVQ_SELECTED_BATCH
}

void launch_selected_m1_vec8(
    int format, const Projection& a,
    int N,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    cudaStream_t stream) {
    if (launch_group_vec8(format, a, 1, N, ng, nvec, nsign, sub_bits, sign_mode, stream)) return;
    if (format == kNvq1S && N <= 2048) {
        launch_m1_vec8_by_format<8>(
            format, a, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
    } else if (format == kNpq0S && N <= 1024) {
        launch_m1_vec8_by_format<2>(
            format, a, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
    } else {
        launch_m1_vec8_by_format<4>(
            format, a, N, ng, nvec, nsign, sub_bits, sign_mode, stream);
    }
}
