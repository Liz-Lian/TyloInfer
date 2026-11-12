/** Dispatch heterogeneous F16 routes with original tile and cohort policy. */
bool hetero_f16(const Hetero& a, cudaStream_t stream) {
    const int route_tile_m=a.tile_m,pairs=a.tokens*a.routes,n_experts=a.experts;
    const int out_per_expert=a.output_rows,neuron_len=a.width,pools=a.pools,routes=a.routes;
    const int format_group=a.format_group;
    const bool masked_experts=a.masked_experts,routed_input=a.routed_input;
    const int tile_m = static_cast<int>(route_tile_m);
    const int rows_per_expert = std::max(
        1, (pairs + static_cast<int>(n_experts) - 1) /
               static_cast<int>(n_experts));
    const int fine_bm = rows_per_expert <= 16
        ? 16
        : (rows_per_expert <= 32 ? 32 : 64);
    const int max_tiles =
        (pairs + tile_m - 1) / tile_m + static_cast<int>(n_experts);
    const int base_block_cap = pairs >= 32768 ? 8192 : 4096;
    const int64_t route_tiles =
        (pairs + tile_m - 1) / tile_m;
    const int narrow_ntiles_n =
        (static_cast<int>(out_per_expert) + 63) / 64;
    // Use two smaller resident CTAs only while each useful tile gets its own
    // CTA; longer routes retain the wider tile's activation reuse.
    const bool use_narrow_tile = tile_m == 128 &&
        route_tiles * narrow_ntiles_n <= base_block_cap;
    const int tile_n = use_narrow_tile
        ? 64 : kNvqMoeF16HeteroTileN;
    const int ntiles_n =
        (static_cast<int>(out_per_expert) + tile_n - 1) / tile_n;
    const int64_t useful_tiles =
        (pairs + tile_m - 1) / tile_m;
    const int64_t useful_tasks = useful_tiles * ntiles_n;
    int block_cap = std::max(
        base_block_cap,
        static_cast<int>(std::min<int64_t>(
            INT_MAX, (useful_tasks + 3) / 4)));
    const int64_t max_tasks =
        static_cast<int64_t>(max_tiles) * ntiles_n;
    if (max_tasks > block_cap) {
        const int average_route_tiles =
            (rows_per_expert + tile_m - 1) / tile_m;
        const int task_period = average_route_tiles * ntiles_n;
        // Rotate CTAs between expert phases when the grid stride would
        // otherwise repeat the same valid/empty task pattern.  For a fully
        // covered multi-pool tensor, one extra CTA breaks a repeated pool
        // phase without changing single-pool launches.
        if (task_period > 1 && block_cap % task_period == 0) {
            if (masked_experts && tile_m != 8 && task_period >= 64) {
                block_cap += task_period;
            } else if (tile_m == 8 && base_block_cap == 4096 &&
                       rows_per_expert > 16) {
                block_cap += std::min(16, std::max(1, task_period / 4));
            } else if (pools > 1 && task_period >= 256) {
                ++block_cap;
            }
        }
    }
    const int blocks = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(max_tasks, block_cap)));
#define NVQ_MOE_HETERO_F16_LAUNCH(                                             \
    BM_VALUE, BN_VALUE, ROUTE_VALUE, GROUP_VALUE, FORMAT_GROUP_VALUE)          \
    do {                                                                         \
        constexpr int kSharedBytes = sizeof(NvqMoeF16SharedStorage<             \
            BM_VALUE, BN_VALUE, GROUP_VALUE>);                                  \
        if constexpr (kSharedBytes > 48 * 1024) {                               \
            static const cudaError_t attribute_status = cudaFuncSetAttribute(   \
                nvq_moe_grouped_hetero_f16_kernel<                              \
                    BM_VALUE, BN_VALUE, ROUTE_VALUE, GROUP_VALUE,               \
                    FORMAT_GROUP_VALUE, true>,                                  \
                cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);     \
            if (attribute_status != cudaSuccess) return false;          \
        }                                                                        \
        nvq_moe_grouped_hetero_f16_kernel<                                      \
            BM_VALUE, BN_VALUE, ROUTE_VALUE, GROUP_VALUE,                       \
            FORMAT_GROUP_VALUE, true><<<                                        \
            blocks, dim3(32, (BM_VALUE >= 128 ? BM_VALUE / 64 : 1) *            \
                (BN_VALUE / 16)), kSharedBytes,                                 \
            stream>>>(                                       \
        a.weight_ptrs, a.weight_sizes,      \
        a.pool_params, a.expert_pool,       \
        a.expert_local,                                      \
        static_cast<const __half *>(a.input),              \
        a.ids_dst, a.expert_bounds,         \
        a.tile_bounds, a.tile_experts,      \
        static_cast<__half *>(a.output), routes,          \
        static_cast<int>(n_experts), pools,                                    \
        static_cast<int>(out_per_expert), static_cast<int>(neuron_len),        \
        max_tiles, routed_input);                                                \
    } while (false)
    // A bounded format switch fits the wide G2 task in 64 registers, so two
    // BM128 CTAs can reside together.  Cross-group mixtures keep the full
    // switch, G4 path, and its wider K reuse.
    const char* wide_rows_option = tyloi::cuda::env_lookup("TYLOINFER_NVQ_PREFILL_M256");
    const bool wide_rows = wide_rows_option && std::atoi(wide_rows_option) != 0;
    const char* narrow_g2_option = tyloi::cuda::env_lookup("TYLOINFER_NVQ_PREFILL_NARROW_G2");
    const bool narrow_g2 = narrow_g2_option && std::atoi(narrow_g2_option) != 0;
    const char* cohorts_option = tyloi::cuda::env_lookup("TYLOINFER_NVQ_PREFILL_COHORTS");
    const bool cohorts = !cohorts_option || std::atoi(cohorts_option) != 0;
    // Keep the M/N geometry and Tensor Core sum order. A shorter K tile
    // lowers decode liveness and shared storage, allowing three narrow CTAs.
    if (cohorts && tile_m == 128 && format_group == kNvqMoeF16AllFormats) {
        // Disjoint format switches write disjoint expert rows. Every task
        // retains the canonical Tensor Core K sequence, with no new maps
        // or weight buffers. Keep G4 for narrow tiles and the legacy decoder.
        if (use_narrow_tile) {
            NVQ_MOE_HETERO_F16_LAUNCH(128,64,128,4,kNvqMoeF16StandardFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,64,128,4,kNvqMoeF16ExtendedFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,64,128,4,kNvqMoeF16LegacyFormats);
        } else {
            NVQ_MOE_HETERO_F16_LAUNCH(128,128,128,2,kNvqMoeF16StandardFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,128,128,2,kNvqMoeF16ExtendedFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,128,128,4,kNvqMoeF16LegacyFormats);
        }
    } else if (narrow_g2 && tile_m == 128 && use_narrow_tile &&
            format_group == kNvqMoeF16StandardFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 2, kNvqMoeF16StandardFormats);
    } else if (narrow_g2 && tile_m == 128 && use_narrow_tile &&
            format_group == kNvqMoeF16ExtendedFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 2, kNvqMoeF16ExtendedFormats);
    } else if (wide_rows && tile_m == 128 && use_narrow_tile && rows_per_expert >= 128 &&
            out_per_expert <= neuron_len) {
        if (format_group == kNvqMoeF16StandardFormats) {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16StandardFormats);
        } else if (format_group == kNvqMoeF16ExtendedFormats) {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16ExtendedFormats);
        } else if (format_group == kNvqMoeF16LegacyFormats) {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16LegacyFormats);
        } else {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16AllFormats);
        }
    } else if (tile_m == 128 && use_narrow_tile &&
            format_group == kNvqMoeF16StandardFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 4, kNvqMoeF16StandardFormats);
    } else if (tile_m == 128 && !use_narrow_tile &&
            format_group == kNvqMoeF16StandardFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 2, kNvqMoeF16StandardFormats);
    } else if (tile_m == 128 && !use_narrow_tile &&
            format_group == kNvqMoeF16ExtendedFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 2, kNvqMoeF16ExtendedFormats);
    } else if (tile_m == 128 && !use_narrow_tile &&
            format_group == kNvqMoeF16LegacyFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 2, kNvqMoeF16LegacyFormats);
    } else if (tile_m == 128 && use_narrow_tile) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 4, kNvqMoeF16AllFormats);
    } else if (tile_m == 128) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 4, kNvqMoeF16AllFormats);
    } else if (tile_m == 64) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 64, 4, kNvqMoeF16AllFormats);
    } else if (fine_bm == 16) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            16, 128, 8, 4, kNvqMoeF16AllFormats);
    } else if (fine_bm == 32) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            32, 128, 8, 4, kNvqMoeF16AllFormats);
    } else {
        NVQ_MOE_HETERO_F16_LAUNCH(
            64, 128, 8, 4, kNvqMoeF16AllFormats);
    }
#undef NVQ_MOE_HETERO_F16_LAUNCH
    return true;
}
