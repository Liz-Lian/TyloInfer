/** Dispatch heterogeneous quantized routes with the original occupancy policy. */
void hetero_quant(const HeteroQuant& q, cudaStream_t stream) {
    const auto& a=q.route;
    const int K=a.width,ng=(K+23)/24,tokens=a.tokens,routes=a.routes,pairs=tokens*routes;
    const int n_experts=a.experts,pools=a.pools,out_per_expert=a.output_rows;
    const bool routed_input=a.routed_input;
#define NVQ_MOE_HETERO_MMVQ_LAUNCH(NWARPS_VALUE, ROWS_VALUE)                   \
    nvq_moe_mmvq_hetero_kernel<                                                \
        NWARPS_VALUE, ROWS_VALUE, false><<<                                    \
        dim3((static_cast<int>(out_per_expert) + ROWS_VALUE - 1) / ROWS_VALUE, pairs), \
        dim3(32, NWARPS_VALUE), 0, stream>>>(                                  \
        a.weight_ptrs, a.weight_sizes,     \
        a.pool_params, a.expert_pool,      \
        a.expert_local, q.codes,               \
        q.scales, q.ids,                     \
        static_cast<__half *>(a.output), pairs, routes,   \
        static_cast<int>(n_experts), pools, static_cast<int>(out_per_expert),  \
        routed_input)
    const char* grouped=tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_GROUP_DOT");
    if(ng<=128 && out_per_expert<=4096 && (!grouped || grouped[0]!='0')) {
        tyloi::cuda::NvqRoutedReuseParams reuse;
        reuse.pointers=a.weight_ptrs;reuse.sizes=a.weight_sizes;
        reuse.params=a.pool_params;reuse.expert_pool=a.expert_pool;
        reuse.expert_local=a.expert_local;reuse.ids=q.ids;
        reuse.active=q.active;
        reuse.input=q.codes;reuse.scales=q.scales;
        reuse.output=static_cast<half*>(a.output);
        reuse.experts=int(n_experts);reuse.pools=pools;reuse.rows=int(out_per_expert);
        reuse.routes=routes;reuse.pairs=pairs;reuse.down=routed_input;
        if(!tyloi::cuda::nvq_launch_routed_reuse(reuse,stream))
            nvq_moe_groups_hetero_kernel<<<dim3((out_per_expert+3)/4,pairs),dim3(32,4),0,stream>>>(
            a.weight_ptrs,a.weight_sizes,a.pool_params,
            a.expert_pool,a.expert_local,q.ids,
            q.codes,q.scales,static_cast<half*>(a.output),
            q.active,
            int(n_experts),pools,int(out_per_expert),routes,pairs,routed_input);
    }else if(q.active != nullptr) {
        const auto launch=[&](auto warp_tag) {
            constexpr int warps=decltype(warp_tag)::value;
            const int blocks=int(std::min<int64_t>(nvq_active_resident_blocks<warps>(),
                ((out_per_expert-1)/2+1)*pairs));
            nvq_moe_mmvq_hetero_kernel<warps,2,false,true><<<blocks,dim3(32,warps),0,stream>>>(
                a.weight_ptrs,a.weight_sizes,a.pool_params,
                a.expert_pool,a.expert_local,q.codes,
                q.scales,q.ids,static_cast<__half*>(a.output),
                pairs,routes,static_cast<int>(n_experts),pools,static_cast<int>(out_per_expert),routed_input,
                q.active);
        };
        if(K>=4096)launch(std::integral_constant<int,8>{});else launch(std::integral_constant<int,4>{});
    } else if (K >= 4096) NVQ_MOE_HETERO_MMVQ_LAUNCH(8,2);
    else {
        // Keep each row's four-warp accumulation and reduction unchanged.
        // The opt-in sweep varies only the number of rows sharing a CTA.
        const auto* requested=tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_ROW_TILE");
        const int rows=tokens==1 && requested ? std::atoi(requested) : 2;
        switch(rows) {
            case 1: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,1);break;
            case 4: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,4);break;
            case 8: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,8);break;
            case 16: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,16);break;
            case 32: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,32);break;
            default: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,2);break;
        }
    }
#undef NVQ_MOE_HETERO_MMVQ_LAUNCH
}
