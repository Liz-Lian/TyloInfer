/** Dispatch the original NVQ pooled route kernels after quantization. */
bool nvq_grouped(const Grouped& a, cudaStream_t stream) {
    const auto& p=a.projection; const auto& wgt=p.weight;
    const int tokens=a.tokens,routes=a.routes,pairs=tokens*routes,n_experts=a.experts;
    const int pool_experts=a.pool_experts,out_per_expert=a.output_rows,K=wgt.width;
    const int gs=wgt.group_size,sub_bits=wgt.state_bits,format=wgt.format,sign_mode=wgt.sign_mode;
    const bool routed_input=a.routed_input;
    const int ng=(K+gs-1)/gs,nvec=(K+(is_d4_format(format)?3:7))/(is_d4_format(format)?4:8),nsign=(K+7)/8;
    if(dense_nvq_format(int(format)) && wgt.aux_bytes==0) {
        const NvqDeviceWeight w{wgt.indices,wgt.indices_bytes,nullptr,0,nullptr,0,
            wgt.scale,wgt.table,int(wgt.table_bytes),
            int(pool_experts*out_per_expert),ng,nvec,nsign,int(sub_bits),int(sign_mode)};
        launch_by_format(int(format),[&](auto tag) {
            constexpr int F=decltype(tag)::value;
            if constexpr(dense_nvq_format(F))
                nvq_moe_groups_kernel<F><<<dim3((out_per_expert+3)/4,pairs),dim3(32,4),0,stream>>>(
                    w,p.quantized,p.scales,a.ids,
                    a.expert_local,static_cast<half*>(p.output),
                    int(n_experts),int(pool_experts),int(out_per_expert),routes,routed_input);
        });
        return true;
    }

    if (tokens <= 8) {
        const char * rows_env = tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_ROWS_PER_BLOCK");
        int rows_per_block = rows_env == nullptr ? 2 : std::atoi(rows_env);
        if (rows_per_block != 1 && rows_per_block != 2 &&
                rows_per_block != 4 && rows_per_block != 8) {
            rows_per_block = 2;
        }
        const char * warps_env = tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_WARPS");
        int warps_override = warps_env == nullptr ? 0 : std::atoi(warps_env);
        if (warps_override != 1 && warps_override != 2 &&
                warps_override != 4 && warps_override != 8) {
            warps_override = 0;
        }
        const char * exact_reduction_env =
            tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_EXACT_REDUCTION");
        const bool exact_reduction =
            exact_reduction_env != nullptr &&
            std::atoi(exact_reduction_env) != 0;
        const char * exact_physical_env =
            tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_EXACT_PHYSICAL_WARPS");
        const int exact_physical_warps =
            exact_physical_env != nullptr && std::atoi(exact_physical_env) == 4
            ? 4
            : 2;
        const char * share_group_env =
            tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_SHARE_GROUP_STATE");
        const bool share_group_state =
            share_group_env != nullptr && std::atoi(share_group_env) != 0;
#define NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, SHARE_VALUE)              \
        nvq_moe_mmvq_kernel<F, NWARPS_VALUE, ROWS_VALUE, SHARE_VALUE><<<         \
            dim3((static_cast<int>(out_per_expert) + ROWS_VALUE - 1) /          \
                     ROWS_VALUE,                                                 \
                 pairs),                                                         \
            dim3(32, NWARPS_VALUE), 0, stream>>>(                                \
            wgt.indices, wgt.indices_bytes,                        \
            wgt.aux, wgt.aux_bytes,                                \
            wgt.state, wgt.state_bytes,                    \
            wgt.scale, wgt.table,         \
            p.quantized, p.scales,                     \
            a.ids, a.expert_local,           \
            static_cast<__half *>(p.output),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), routed_input)
#define NVQ_MOE_SELECTED_LAUNCH(NWARPS_VALUE, ROWS_VALUE)                        \
        do {                                                                      \
            if constexpr (F == kNvq2Exec || F == kNvq2JscExec) {                 \
                if (share_group_state) {                                          \
                    NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, true);         \
                } else {                                                          \
                    NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, false);        \
                }                                                                 \
            } else {                                                              \
                NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, false);            \
            }                                                                     \
        } while (0)
#define NVQ_MOE_EXACT_LAUNCH(PHYSICAL_WARPS_VALUE, LOGICAL_WARPS_VALUE)          \
        nvq_moe_mmvq_exact_reduction_kernel<                                    \
            F, PHYSICAL_WARPS_VALUE, LOGICAL_WARPS_VALUE, 2><<<                 \
            dim3((static_cast<int>(out_per_expert) + 1) / 2, pairs),            \
            dim3(32, PHYSICAL_WARPS_VALUE), 0, stream>>>(                        \
            wgt.indices, wgt.indices_bytes,                        \
            wgt.aux, wgt.aux_bytes,                                \
            wgt.state, wgt.state_bytes,                    \
            wgt.scale, wgt.table,         \
            p.quantized, p.scales,                     \
            a.ids, a.expert_local,           \
            static_cast<__half *>(p.output),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), routed_input)
        launch_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            const char* grouped=tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_GROUP_DOT");
            if(!exact_reduction && ng<=128 && out_per_expert<=4096 && (!grouped || grouped[0]!='0')) {
                const NvqDeviceWeight w{wgt.indices,wgt.indices_bytes,wgt.aux,wgt.aux_bytes,
                    wgt.state,wgt.state_bytes,wgt.scale,wgt.table,
                    int(wgt.table_bytes),int(pool_experts*out_per_expert),ng,nvec,nsign,int(sub_bits),int(sign_mode)};
                tyloi::cuda::NvqRoutedReuseParams reuse;
                reuse.weight=w;reuse.format=F;reuse.expert_local=a.expert_local;
                reuse.ids=a.ids;reuse.input=p.quantized;reuse.scales=p.scales;
                reuse.output=static_cast<half*>(p.output);
                reuse.experts=int(n_experts);reuse.local_experts=int(pool_experts);reuse.rows=int(out_per_expert);
                reuse.routes=routes;reuse.pairs=pairs;reuse.down=routed_input;
                if(!tyloi::cuda::nvq_launch_routed_reuse(reuse,stream))
                    nvq_moe_groups_kernel<F><<<dim3((out_per_expert+3)/4,pairs),dim3(32,4),0,stream>>>(
                    w,p.quantized,p.scales,a.ids,a.expert_local,
                    static_cast<half*>(p.output),int(n_experts),int(pool_experts),int(out_per_expert),routes,routed_input);
                return;
            }
            if (exact_reduction) {
                if (K >= 4096) {
                    if (exact_physical_warps == 4) {
                        NVQ_MOE_EXACT_LAUNCH(4, 8);
                    } else {
                        NVQ_MOE_EXACT_LAUNCH(2, 8);
                    }
                } else {
                    NVQ_MOE_EXACT_LAUNCH(2, 4);
                }
                return;
            }
            if (warps_override == 1) {
                NVQ_MOE_SELECTED_LAUNCH(1, 2);
                return;
            }
            if (warps_override == 2) {
                if (rows_per_block == 4) {
                    NVQ_MOE_SELECTED_LAUNCH(2, 4);
                } else {
                    NVQ_MOE_SELECTED_LAUNCH(2, 2);
                }
                return;
            }
            if (warps_override == 4) {
                NVQ_MOE_SELECTED_LAUNCH(4, 2);
                return;
            }
            if (warps_override == 8) {
                NVQ_MOE_SELECTED_LAUNCH(8, 2);
                return;
            }
            if (K >= 4096) {
                if (rows_per_block == 1) NVQ_MOE_SELECTED_LAUNCH(8, 1);
                else if (rows_per_block == 4) NVQ_MOE_SELECTED_LAUNCH(8, 4);
                else if (rows_per_block == 8) NVQ_MOE_SELECTED_LAUNCH(8, 8);
                else NVQ_MOE_SELECTED_LAUNCH(8, 2);
            } else {
                if (rows_per_block == 1) NVQ_MOE_SELECTED_LAUNCH(4, 1);
                else if (rows_per_block == 4) NVQ_MOE_SELECTED_LAUNCH(4, 4);
                else if (rows_per_block == 8) NVQ_MOE_SELECTED_LAUNCH(4, 8);
                else NVQ_MOE_SELECTED_LAUNCH(4, 2);
            }
        });
#undef NVQ_MOE_EXACT_LAUNCH
#undef NVQ_MOE_SELECTED_LAUNCH
#undef NVQ_MOE_SMALL_LAUNCH
    } else {
        constexpr int route_tile = 8;
        const int row_tiles = (static_cast<int>(out_per_expert) + 3) / 4;
        const int max_tiles = (pairs + route_tile - 1) / route_tile +
            static_cast<int>(n_experts);
        const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
        const int blocks = static_cast<int>(
            std::min<int64_t>(max_tasks, 4096));
#define NVQ_MOE_GROUPED_LAUNCH()                                                 \
        nvq_moe_grouped_tile_kernel<F, route_tile><<<                            \
            blocks, dim3(32, 4), 0, stream>>>(                                   \
            wgt.indices, wgt.indices_bytes,                        \
            wgt.aux, wgt.aux_bytes,                                \
            wgt.state, wgt.state_bytes,                    \
            wgt.scale, wgt.table,         \
            p.quantized, p.scales,                     \
            a.expert_local, a.ids_dst,       \
            a.expert_bounds, a.tile_bounds,  \
            a.tile_experts,                                    \
            static_cast<__half *>(p.output),                \
            routes, static_cast<int>(n_experts),                                 \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), max_tiles, row_tiles, routed_input)
        launch_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            NVQ_MOE_GROUPED_LAUNCH();
        });
#undef NVQ_MOE_GROUPED_LAUNCH
    }
    return false;
}
