#include "nvq_raw.h"
#include "env.h"
#include <stdexcept>
#include <cstdlib>
#include "packed_nvq.cuh"
#include "nvq_dense_group.cuh"
#include "nvq_format.cuh"
#include "async_copy.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>
using namespace nvcuda;
namespace {
using namespace tyloi::cuda::packed_nvq;
#include "nvq_projection_device.cuh"
#include "nvq_projection_launch.cuh"
using tyloi::cuda::nvq_raw::Projection;
using tyloi::cuda::nvq_format::launch_by_format;
#include "nvq_vec_launch.cuh"
}
namespace tyloi::cuda::nvq_raw {
using namespace tyloi::cuda::packed_nvq;
using tyloi::cuda::nvq_format::launch_by_format;
void gemm(const Weight& w, const void* input, void* output, int rows, cudaStream_t stream) {
    const int M=rows,N=w.rows,K=w.width,ng=(K+w.group_size-1)/w.group_size;
    const int nvec=(K+(is_d4_format(w.format)?3:7))/(is_d4_format(w.format)?4:8);
    const int nsign=(K+7)/8;
    const dim3 block(32,8);
#define NVQ_F16_GEMM_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE) \
    nvq_gemm_f16_gs24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<< \
        dim3((N+63)/64,(M+16*MTILES_VALUE-1)/(16*MTILES_VALUE)),block,0,stream>>>( \
        w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,w.table, \
        static_cast<const __half*>(input),static_cast<__half*>(output), \
        M,N,K,ng,nvec,nsign,w.state_bits,w.sign_mode)
    launch_by_format(w.format, [&](auto tag) {
        constexpr int F=decltype(tag)::value;
        if (M<=16) { if(M==16 && N%64==0) NVQ_F16_GEMM_LAUNCH(1,false); else NVQ_F16_GEMM_LAUNCH(1,true); }
        else if (M<=32) { if(M==32 && N%64==0) NVQ_F16_GEMM_LAUNCH(2,false); else NVQ_F16_GEMM_LAUNCH(2,true); }
        else if (M<=48) { if(M==48 && N%64==0) NVQ_F16_GEMM_LAUNCH(3,false); else NVQ_F16_GEMM_LAUNCH(3,true); }
        else if (M<=64) { if(M==64 && N%64==0) NVQ_F16_GEMM_LAUNCH(4,false); else NVQ_F16_GEMM_LAUNCH(4,true); }
        else if (M<=80) { if(M==80 && N%64==0) NVQ_F16_GEMM_LAUNCH(5,false); else NVQ_F16_GEMM_LAUNCH(5,true); }
        else if (M<=96) { if(M==96 && N%64==0) NVQ_F16_GEMM_LAUNCH(6,false); else NVQ_F16_GEMM_LAUNCH(6,true); }
        else if (M<=112) { if(M==112 && N%64==0) NVQ_F16_GEMM_LAUNCH(7,false); else NVQ_F16_GEMM_LAUNCH(7,true); }
        else { if(M%128==0 && N%64==0) NVQ_F16_GEMM_LAUNCH(8,false); else NVQ_F16_GEMM_LAUNCH(8,true); }
    });
#undef NVQ_F16_GEMM_LAUNCH
}

static void quantize_input(const Projection& a, bool gated, cudaStream_t stream) {
    const int M=a.rows,K=a.weight.width,ng=(K+a.weight.group_size-1)/a.weight.group_size;
    if(gated && a.gate_mode==1) nvq_quantize_x_gate_gs24_kernel<1><<<dim3(M,ng),32,0,stream>>>(
        static_cast<const __half*>(a.input),static_cast<const __half*>(a.gate),a.quantized,a.scales,M,K,ng);
    else if(gated) nvq_quantize_x_gate_gs24_kernel<2><<<dim3(M,ng),32,0,stream>>>(
        static_cast<const __half*>(a.input),static_cast<const __half*>(a.gate),a.quantized,a.scales,M,K,ng);
    else nvq_quantize_x_gs24_kernel<<<dim3(M,ng),32,0,stream>>>(
        static_cast<const __half*>(a.input),a.quantized,a.scales,M,K,ng);
}
void gemv(const Projection& a, bool quantize, bool gate, int method, cudaStream_t stream) {
    const auto& w=a.weight;
    const int M=a.rows,N=w.rows,K=w.width,ng=(K+w.group_size-1)/w.group_size;
    const int nvec=(K+(is_d4_format(w.format)?3:7))/(is_d4_format(w.format)?4:8);
    const int nsign=(K+7)/8;
    if(quantize) quantize_input(a,gate,stream);
    if(method==1) {
        switch(a.warps) {
        case 1:launch_m1_vec8_by_format<1>(w.format,a,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);break;
        case 2:launch_m1_vec8_by_format<2>(w.format,a,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);break;
        case 4:launch_m1_vec8_by_format<4>(w.format,a,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);break;
        case 8:launch_m1_vec8_by_format<8>(w.format,a,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);break;
        }
    } else if(method==2) {
#define NVQ_RAW_BATCH(NW) \
    if(M<=2) launch_batch_vec8_by_format<NW,2>(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream); \
    else if(M<=4) launch_batch_vec8_by_format<NW,4>(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream); \
    else if(M<=8) launch_batch_vec8_by_format<NW,8>(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream); \
    else if(M<=12) launch_batch_vec8_by_format<NW,12>(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream); \
    else launch_batch_vec8_by_format<NW,16>(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream)
        switch(a.warps) { case 2: NVQ_RAW_BATCH(2); break; case 4: NVQ_RAW_BATCH(4); break; case 8: NVQ_RAW_BATCH(8); break; }
#undef NVQ_RAW_BATCH
    } else if(M==1) launch_selected_m1_vec8(w.format,a,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);
    else launch_selected_batch_vec8(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);
}
bool group_exec(int format) { return format==kNvq2JscXLGroupExec || format==kNvq3JscLGroupExec; }
void gemv_residual(const Projection& a, cudaStream_t stream) {
    const auto& w=a.weight;
    const int ng=(w.width+w.group_size-1)/w.group_size;
    const int nvec=(w.width+(is_d4_format(w.format)?3:7))/(is_d4_format(w.format)?4:8);
    const int nsign=(w.width+7)/8;
    const NvqDeviceWeight weight{w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,
        w.scale,w.table,int(w.table_bytes),w.rows,ng,nvec,nsign,w.state_bits,w.sign_mode};
    if(w.format==kNvq2JscXLGroupExec) launch_aligned_group_m1<kNvq2JscXLGroupExec,true>(
        weight,a.quantized,a.scales,static_cast<__half*>(a.output),stream,static_cast<const __half*>(a.residual));
    else launch_aligned_group_m1<kNvq3JscLGroupExec,true>(
        weight,a.quantized,a.scales,static_cast<__half*>(a.output),stream,static_cast<const __half*>(a.residual));
}
void mmq(const Projection& a, bool gated, cudaStream_t stream) {
    const auto& w=a.weight;
    const int M=a.rows,N=w.rows,K=w.width,ng=(K+w.group_size-1)/w.group_size;
    const int nvec=(K+(is_d4_format(w.format)?3:7))/(is_d4_format(w.format)?4:8);
    const int nsign=(K+7)/8;
    quantize_input(a,gated,stream);
#define NVQ_RAW_MMQ(TILES,CHECK) \
    nvq_mmq_mma24_kernel<F,TILES,CHECK><<<dim3((N+63)/64,1),dim3(32,8),0,stream>>>( \
        w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,w.table, \
        a.quantized,a.scales,static_cast<__half*>(a.output),M,N,ng,nvec,nsign,w.state_bits,w.sign_mode)
    launch_by_format(w.format,[&](auto tag) {
        constexpr int F=decltype(tag)::value;
        if(M<=16) { if(M==16 && N%64==0) NVQ_RAW_MMQ(1,false); else NVQ_RAW_MMQ(1,true); }
        else if(M<=32) { if(M==32 && N%64==0) NVQ_RAW_MMQ(2,false); else NVQ_RAW_MMQ(2,true); }
        else if(M<=48) { if(M==48 && N%64==0) NVQ_RAW_MMQ(3,false); else NVQ_RAW_MMQ(3,true); }
        else { if(M==64 && N%64==0) NVQ_RAW_MMQ(4,false); else NVQ_RAW_MMQ(4,true); }
    });
#undef NVQ_RAW_MMQ
}
}
