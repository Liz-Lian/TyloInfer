#pragma once
template <int FORMAT, int NWARPS, int MAX_M, bool FAST_NVQ2_S4 = false>
void launch_multi2_vec8(
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const int8_t* qx,
    const float* xscale,
    __half* output,
    int M,
    cudaStream_t stream) {
    nvq_gemv_multi2_vec8_kernel<FORMAT, NWARPS, MAX_M, FAST_NVQ2_S4>
        <<<first.N + second.N, dim3(32, NWARPS), 0, stream>>>(
            first, second, qx, xscale,
            output, M);
}

void launch_selected_multi2_vec8(
    int format,
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const int8_t* qx,
    const float* xscale,
    __half* output,
    int M,
    cudaStream_t stream) {
#define NVQ_MULTI2(F, NW, MAX_M) \
    launch_multi2_vec8<F, NW, MAX_M>(first, second, qx, xscale, output, M, stream)
#define NVQ_MULTI2_FAST(F, NW, MAX_M) \
    launch_multi2_vec8<F, NW, MAX_M, true>(first, second, qx, xscale, output, M, stream)
    const bool fast_nvq2_s4 =
        (format == kNvq2 || format == kNvq2Exec ||
         format == kNvq2Jsc || format == kNvq2JscExec) &&
        first.sub_bits == 4 && second.sub_bits == 4;
    if (M == 1) {
        switch (format) {
            case kNvq1L: NVQ_MULTI2(kNvq1L, 4, 1); break;
            case kNvq2:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2, 4, 1);
                else NVQ_MULTI2(kNvq2, 4, 1);
                break;
            case kNvq3: NVQ_MULTI2(kNvq3, 4, 1); break;
            case kNvq2Exec:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2Exec, 4, 1);
                else NVQ_MULTI2(kNvq2Exec, 4, 1);
                break;
            case kNvq2Jsc:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2Jsc, 4, 1);
                else NVQ_MULTI2(kNvq2Jsc, 4, 1);
                break;
            case kNvq2JscExec:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2JscExec, 4, 1);
                else NVQ_MULTI2(kNvq2JscExec, 4, 1);
                break;
            case kNpq0L: NVQ_MULTI2(kNpq0L, 4, 1); break;
            case kNvq1S: NVQ_MULTI2(kNvq1S, 4, 1); break;
            case kNpq0S: NVQ_MULTI2(kNpq0S, 4, 1); break;
            case kNvq3Jsc: NVQ_MULTI2(kNvq3Jsc, 4, 1); break;
            case kNvq3Jsc2: NVQ_MULTI2(kNvq3Jsc2, 4, 1); break;
            case kNvq3Jsc512: NVQ_MULTI2(kNvq3Jsc512, 4, 1); break;
            case kNvq2JscL: NVQ_MULTI2(kNvq2JscL, 4, 1); break;
            case kNvq2JscXL: NVQ_MULTI2(kNvq2JscXL, 4, 1); break;
            case kNvq3JscL: NVQ_MULTI2(kNvq3JscL, 4, 1); break;
            case kNvq2JscXLGroupExec:
                launch_aligned_group_multi2_m1<kNvq2JscXLGroupExec>(
                    first, second, qx, xscale,
                    output, stream);
                break;
            case kNvq3JscLGroupExec:
                launch_aligned_group_multi2_m1<kNvq3JscLGroupExec>(
                    first, second, qx, xscale,
                    output, stream);
                break;
        }
    } else if (M <= 4) {
        if (format == kNvq1L || format == kNvq1S) {
            if (format == kNvq1S) NVQ_MULTI2(kNvq1S, 8, 4);
            else NVQ_MULTI2(kNvq1L, 8, 4);
        } else if (is_e8_format(format)) {
            if (format == kNvq2Exec) NVQ_MULTI2(kNvq2Exec, 4, 4);
            else if (format == kNvq2Jsc) NVQ_MULTI2(kNvq2Jsc, 4, 4);
            else if (format == kNvq2JscExec) NVQ_MULTI2(kNvq2JscExec, 4, 4);
            else if (format == kNvq2JscL) NVQ_MULTI2(kNvq2JscL, 4, 4);
            else if (format == kNvq2JscXL) NVQ_MULTI2(kNvq2JscXL, 4, 4);
            else if (format == kNvq2JscXLGroupExec) {
                NVQ_MULTI2(kNvq2JscXLGroupExec, 4, 4);
            }
            else NVQ_MULTI2(kNvq2, 4, 4);
        } else if (format == kNpq0L || format == kNpq0S) {
            if (format == kNpq0S) NVQ_MULTI2(kNpq0S, 4, 4);
            else NVQ_MULTI2(kNpq0L, 4, 4);
        } else if (M <= 2) {
            if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 2, 4);
            else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 2, 4);
            else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 2, 4);
            else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 2, 4);
            else if (format == kNvq3JscLGroupExec) {
                NVQ_MULTI2(kNvq3JscLGroupExec, 2, 4);
            }
            else NVQ_MULTI2(kNvq3, 2, 4);
        } else {
            if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 4, 4);
            else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 4, 4);
            else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 4, 4);
            else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 4, 4);
            else if (format == kNvq3JscLGroupExec) {
                NVQ_MULTI2(kNvq3JscLGroupExec, 4, 4);
            }
            else NVQ_MULTI2(kNvq3, 4, 4);
        }
    } else if (format == kNvq1L || format == kNvq1S) {
        if (format == kNvq1S) NVQ_MULTI2(kNvq1S, 8, 8);
        else NVQ_MULTI2(kNvq1L, 8, 8);
    } else if (is_e8_format(format)) {
        if (format == kNvq2Exec) NVQ_MULTI2(kNvq2Exec, 4, 8);
        else if (format == kNvq2Jsc) NVQ_MULTI2(kNvq2Jsc, 4, 8);
        else if (format == kNvq2JscExec) NVQ_MULTI2(kNvq2JscExec, 4, 8);
        else if (format == kNvq2JscL) NVQ_MULTI2(kNvq2JscL, 4, 8);
        else if (format == kNvq2JscXL) NVQ_MULTI2(kNvq2JscXL, 4, 8);
        else if (format == kNvq2JscXLGroupExec) {
            NVQ_MULTI2(kNvq2JscXLGroupExec, 4, 8);
        }
        else NVQ_MULTI2(kNvq2, 4, 8);
    } else if (format == kNpq0L || format == kNpq0S) {
        if (format == kNpq0S) NVQ_MULTI2(kNpq0S, 4, 8);
        else NVQ_MULTI2(kNpq0L, 4, 8);
    } else {
        if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 8, 8);
        else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 8, 8);
        else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 8, 8);
        else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 8, 8);
        else if (format == kNvq3JscLGroupExec) {
            NVQ_MULTI2(kNvq3JscLGroupExec, 8, 8);
        }
        else NVQ_MULTI2(kNvq3, 8, 8);
    }
#undef NVQ_MULTI2
#undef NVQ_MULTI2_FAST
}
