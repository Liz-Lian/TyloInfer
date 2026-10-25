#include "backends/cuda/engine/executor/options.h"
#include "mfq_cuda_embedding_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "mfq_cuda_quant_ops.h"
#include "resident_allocations.h"
#include "vq.h"
#include "backends/cuda/engine/executor/execution.h"
#include "format.h"
#include "quant_dot.h"
#include "nvq_group.h"
#include "cpu_projection_rows.h"
#include "nvq_dense_pack.h"
#include <atomic>
#include <cstdio>

namespace {
NvqWorkspace& resident_nvq_workspace(const NvqWeight& weight,int rows) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mfq::cuda::ResidentAllocationScope resident;
#endif
    return weight.workspace(rows);
}
}

bool prepare_nvq_dense_groups(NvqWeight& w) {
    if(w.dense_groups)return true;
    if(!w.indices_packed.is_cpu() || w.gs!=24 || w.sub_bits!=4 ||
            w.out<=0 || w.out>std::numeric_limits<int>::max() ||
            w.neuron_len<=0 || w.neuron_len>std::numeric_limits<int>::max() ||
            !mfq::cpu::nvq_dense_rows_dot_kernel(int(w.kernel_format)))return false;
    const int rows=int(w.out),pairs=int((w.neuron_len+7)/8),groups=int(w.ng);
    const bool d4=w.kernel_format==10 || w.kernel_format==11 || w.kernel_format==12 || w.kernel_format==15;
    if(d4 && (w.neuron_len+3)/4!=2*pairs)return false;
    // Every expert in an arena starts at a byte boundary. No per-expert
    // padding is inserted; unsupported odd shapes keep canonical storage.
    const int bits=w.kernel_format==12?9:(w.kernel_format==13 || w.kernel_format==15)?10:w.kernel_format==14?12:8;
    if((int64_t(rows)*(int64_t(pairs)*((d4?2:1)*bits+7)+int64_t(groups)*4))&7)return false;
    std::vector<uint8_t> compact;
#define PACK(B,P) mfq::cpu::dense::pack<B,P>(w.indices_packed.data_ptr<uint8_t>(),w.indices_packed.numel(), \
        w.aux_packed.data_ptr<uint8_t>(),w.aux_packed.numel(),w.sub_scale_packed.data_ptr<uint8_t>(), \
        w.sub_scale_packed.numel(),rows,pairs,groups)
    switch(w.kernel_format) {
        case 5:compact=PACK(8,1);break;case 13:compact=PACK(10,1);break;case 14:compact=PACK(12,1);break;
        case 10:case 11:compact=PACK(8,2);break;case 12:compact=PACK(9,2);break;case 15:compact=PACK(10,2);break;
        default:return false;
    }
#undef PACK
    if(compact.size()>size_t(w.indices_packed.numel()+w.aux_packed.numel()+w.sub_scale_packed.numel()))
        throw std::runtime_error("compact NVQ unexpectedly expanded weights");
    w.indices_packed=mfq_tensor_backend::tensor(compact);
    w.aux_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
    w.sub_scale_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
    w.dense_groups=true;
    return true;
}

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

void prepare_nvq_integer_codebook(NvqWeight& w) {
    const auto enabled=mfq::cuda::runtime_options::nvq1_integer_delta();
    if(!enabled || !*enabled || w.integer_codebook.defined() ||
       (w.kernel_format!=1 && w.kernel_format!=8) || !w.codebook.is_cuda())return;
    const auto canonical=w.codebook.to(mfq_tensor_backend::kCPU).contiguous();
    const bool small=w.kernel_format==8;
    const int bank_bytes=small?512*8:2048*8;
    if(canonical.numel()!=int64_t(small?2:1)*bank_bytes)
        throw std::invalid_argument("NVQ1 integer decode book has invalid canonical size");
    std::vector<int8_t> digits(2*bank_bytes);
    const auto* source=canonical.data_ptr<int8_t>();
    for(int negative=0;negative<2;++negative)for(int i=0;i<bank_bytes;++i) {
        const int v=source[(small?negative*bank_bytes:0)+i];
        if(v<-1 || v>1)throw std::invalid_argument("NVQ1 integer decode digits require -1/0/1");
        digits[negative*bank_bytes+i]=int8_t((small?32:8)*v+(negative?-1:1)*(small?5:1));
    }
    w.integer_codebook=mfq_tensor_backend::tensor(digits).to(w.codebook.device()).contiguous();
}
void prepare_nvq_decode_records(NvqWeight& w,bool /*immutable*/) {
    const auto enabled=mfq::cuda::runtime_options::nvq1_group_records();
    if(!enabled || !*enabled || w.decode_records.defined() ||
       (w.kernel_format!=1 && w.kernel_format!=8) || !w.indices_packed.is_cuda() || w.gs!=24 ||
       w.sub_bits!=(w.kernel_format==8?4:3))return;
    if(w.aux_packed.numel()==0){w.decode_records=w.indices_packed;return;}
    const auto indices=w.indices_packed.to(mfq_tensor_backend::kCPU).contiguous();
    const auto auxiliary=w.aux_packed.to(mfq_tensor_backend::kCPU).contiguous();
    const auto states=w.sub_scale_packed.to(mfq_tensor_backend::kCPU).contiguous();
    const auto bits=[](const mfq_tensor_backend::Tensor& data,int64_t bit,int width) {
        uint32_t value=0;
        for(int i=0;i<width;++i)if((bit+i)/8<data.numel())
            value|=uint32_t((data.data_ptr<uint8_t>()[(bit+i)/8]>>((bit+i)&7))&1u)<<i;
        return value;
    };
    const bool small=w.kernel_format==8;
    const int index_bits=small?9:11,record_bytes=small?4:5,nvec=int((w.neuron_len+7)/8);
    std::vector<uint8_t> records(size_t(w.out)*w.ng*record_bytes);
    for(int64_t row=0;row<w.out;++row)for(int group=0;group<w.ng;++group) {
        const int64_t linear=row*w.ng+group;
        uint64_t record=0;
        for(int segment=0;segment<3;++segment)if(group*3+segment<nvec)
            record|=uint64_t(bits(indices,(row*nvec+group*3+segment)*index_bits,index_bits))<<(segment*index_bits);
        record|=uint64_t(bits(states,linear*w.sub_bits,int(w.sub_bits)))<<(3*index_bits);
        record|=uint64_t(bits(auxiliary,linear,1))<<(3*index_bits+w.sub_bits);
        std::memcpy(records.data()+linear*record_bytes,&record,record_bytes);
    }
    w.decode_records=mfq_tensor_backend::tensor(records).to(w.indices_packed.device()).contiguous();
    w.indices_packed=w.decode_records;
    w.aux_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
    w.sub_scale_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
}

enum class NvqMatmulPath {
    Gemv,
    Mmq,
    OnlineF16,
    DequantGemm,
};

static NvqMatmulPath select_nvq_matmul_path(const NvqWeight & w, int M) {
    // Short batches of small expert matrices use vector decode followed by
    // FP32-accumulating GEMM. The decision depends on matrix geometry.
    if (M >= 9 && M <= 16 && w.out <= 4096 && w.neuron_len <= 4096 &&
            !mfq::cuda::runtime_options::nvq_dense_reference()) {
        return NvqMatmulPath::DequantGemm;
    }
    const bool e8_family =
        w.format == 2 || w.format == 5 || w.format == 7 ||
        w.format == 8 || w.format == 9 ||
        w.format == 13 || w.format == 14;

    if (w.format == 8 && M >= 14 && M <= 16 && w.neuron_len >= 2 * w.out) {
        return NvqMatmulPath::DequantGemm;
    }

    if (w.format == 9 && M >= 13) {
        const bool wide_output = w.out * 8 >= w.neuron_len * 21;
        const bool wide_mmq =
            w.out >= 4096 && (w.out >= 2 * w.neuron_len || w.neuron_len >= 2 * w.out);
        if (M <= 15) {
            if (wide_mmq) return NvqMatmulPath::Mmq;
            return w.out >= 1024 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
        }
        if (M == 16) {
            if (w.out >= 4096) return NvqMatmulPath::Mmq;
            return w.out >= 1024 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
        }
        if (M <= 31) {
            if (wide_output) return NvqMatmulPath::OnlineF16;
            if (w.out >= 4096 && w.out >= w.neuron_len) return NvqMatmulPath::Mmq;
            return NvqMatmulPath::DequantGemm;
        }
        if (M == 32) {
            return w.out >= 4096 ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        if (M <= 47) {
            return wide_output ? NvqMatmulPath::OnlineF16 : NvqMatmulPath::DequantGemm;
        }
        if (M == 48) {
            return wide_output ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        if (M <= 63) {
            return wide_output ? NvqMatmulPath::OnlineF16 : NvqMatmulPath::DequantGemm;
        }
        if (M == 64) {
            if (wide_output) return NvqMatmulPath::OnlineF16;
            return w.out >= 8192 ? NvqMatmulPath::Mmq : NvqMatmulPath::DequantGemm;
        }
        return NvqMatmulPath::DequantGemm;
    }

    if (M <= 13) return NvqMatmulPath::Gemv;
    if (M == 14) {
        return w.out >= 2048 ? NvqMatmulPath::Gemv : NvqMatmulPath::DequantGemm;
    }
    if (M == 15) {
        if (e8_family && w.out >= 8192) return NvqMatmulPath::Mmq;
        if (e8_family && w.out >= 2048) return NvqMatmulPath::Gemv;
        return NvqMatmulPath::DequantGemm;
    }
    if (M == 16) {
        if (e8_family && w.out >= 6144) return NvqMatmulPath::Mmq;
        if (e8_family && w.out >= 2048) return NvqMatmulPath::Gemv;
        if ((w.format == 3 || w.format == 10 ||
             w.format == 12 || w.format == 15) &&
            w.out >= 4096 && w.neuron_len >= 8192) {
            return NvqMatmulPath::Mmq;
        }
        return NvqMatmulPath::DequantGemm;
    }

    const bool wide_expansion = e8_family && w.out >= 3 * w.neuron_len;
    if (wide_expansion && ((M >= 17 && M <= 31) || (M >= 33 && M <= 47))) {
        return NvqMatmulPath::OnlineF16;
    }
    if (M == 32 && e8_family && w.out >= 8192) return NvqMatmulPath::Mmq;
    if (M == 48 && e8_family && w.out >= 12288) return NvqMatmulPath::Mmq;
    if (M == 64 && (w.format == 7 || w.format == 9) && w.out >= 8192) {
        return NvqMatmulPath::Mmq;
    }
    return NvqMatmulPath::DequantGemm;
}
static bool cpu_nvq_d4(int format) {
    return format == 3 || format == 10 || format == 11 ||
        format == 12 || format == 15;
}

static const int8_t * cpu_nvq_codebook(
        const int8_t * metadata,
        int format,
        uint32_t state) {
    constexpr int64_t header = 64;
    if (format == 1 || format == 2 || format == 3 || format == 8) {
        return metadata;
    }
    const uint32_t bank = format == 11
        ? state & 1u
        : static_cast<uint8_t>(metadata[36 + state]);
    if (format == 5) return metadata + header + bank * 256 * 8;
    if (format == 10 || format == 11) {
        return metadata + header + bank * 256 * 4;
    }
    if (format == 12) return metadata + header + bank * 512 * 4;
    if (format == 13) return metadata + header + bank * 1024 * 8;
    if (format == 14) return metadata + header + bank * 4096 * 8;
    if (format == 15) return metadata + header + bank * 1024 * 4;
    throw std::runtime_error("unsupported CPU NVQ codebook format");
}

static float cpu_nvq_scale(
        const int8_t * metadata,
        int format,
        float anchor,
        uint32_t state) {
    if (format == 1) {
        return anchor * static_cast<float>(state) * 0.125f;
    }
    if (format == 8) {
        return anchor * static_cast<float>(state) * 0.03125f;
    }
    if (format == 2 || format == 3) {
        return anchor * static_cast<float>(state);
    }
    if (format == 11) {
        return anchor * static_cast<float>((state >> 1) + 1);
    }
    const int64_t lut_offset = (format == 7 || format == 9) ? 8 : 4;
    return anchor * cpu_half_from_bytes(
        metadata, lut_offset + static_cast<int64_t>(state) * 2);
}

static int cpu_nvq_parity7(uint32_t value) {
    value &= 0x7fu;
    value ^= value >> 4;
    value ^= value >> 2;
    value ^= value >> 1;
    return static_cast<int>(value & 1u);
}

using CpuNvqDecodeView=mfq::cpu::NvqDecodeView;

static inline uint32_t cpu_nvq_bits(const uint8_t* data, int64_t bytes,
        int64_t bit, int bits) {
    const int64_t offset=bit>>3;
    uint32_t word=0;
    // Native canonical payloads have no mandatory tail padding.
    if (bytes-offset>=4) std::memcpy(&word,data+offset,4);
    else std::memcpy(&word,data+offset,static_cast<size_t>(bytes-offset));
    return (word>>(bit&7))&((1u<<bits)-1u);
}

static inline uint32_t cpu_nvq_state(const uint8_t* indices,const uint8_t* states,
        int64_t state_bytes,int64_t linear,int bits,int format) {
    if(state_bytes==0 && (format==1 || format==8)) {
        uint64_t record=0;const int bytes=format==8?4:5;
        std::memcpy(&record,indices+linear*bytes,bytes);
        return uint32_t(record>>(format==8?27:33))&((1u<<bits)-1u);
    }
    return cpu_nvq_bits(states,state_bytes,linear*bits,bits);
}

// Geometry, storage pointers and state tables are prepared once per GEMV.
// E8 indices/signs are consumed once per eight values, D4 once per four.
template<int Format>
static void cpu_decode_nvq_group(const CpuNvqDecodeView& w,
        int64_t row, int group, uint32_t state, int8_t values[24]) {
    constexpr bool d4 = Format==3 || Format==10 || Format==11 || Format==12 || Format==15;
    constexpr bool delta_format = Format==1 || Format==8;
    constexpr int bits = Format==1 ? 11 : Format==7 ? 7 :
        (Format==8 || Format==12) ? 9 : Format==9 ? 6 :
        (Format==13 || Format==15) ? 10 : Format==14 ? 12 : 8;
    const int8_t* bank=w.banks[state];
    int delta=1;uint64_t record=0;
    if constexpr(delta_format)if(w.aux_bytes==0)
        std::memcpy(&record,w.indices+(row*w.groups+group)*(Format==8?4:5),Format==8?4:5);
    if constexpr (delta_format) {
        const bool negative=w.aux_bytes==0?((record>>(Format==8?31:36))&1u)!=0:
            cpu_nvq_bits(w.aux,w.aux_bytes,row*w.groups+group,1)!=0;
        delta=negative ? -1 : 1;
        if constexpr (Format==8) bank+=int(negative)*512*8;
    }
    for (int chunk=0; chunk<3; ++chunk) {
        const int vector8=group*3+chunk;
        const int vector=d4 ? vector8*2 : vector8;
        if (vector>=w.nvec) {
            std::fill(values+chunk*8,values+chunk*8+8,int8_t(0));
            continue;
        }
        const int64_t linear=row*w.nvec+vector;
        const uint32_t code=delta_format && w.aux_bytes==0?
            uint32_t(record>>(chunk*bits))&((1u<<bits)-1u):bits==8 ? w.indices[linear] :
            cpu_nvq_bits(w.indices,w.index_bytes,linear*bits,bits);
        int8_t decoded[8];
        if constexpr (Format==7) {
            std::memcpy(decoded,bank+(code&7u)*4,4);
            // The second codebook begins after all eight first-half banks.
            std::memcpy(decoded+4,bank+256+state*32+(code>>3)*4,4);
        } else if constexpr (d4) {
            std::memcpy(decoded,bank+int64_t(code)*4,4);
            if (vector+1<w.nvec) {
                const uint32_t second=bits==8 ? w.indices[linear+1] :
                    cpu_nvq_bits(w.indices,w.index_bytes,(linear+1)*bits,bits);
                std::memcpy(decoded+4,bank+int64_t(second)*4,4);
            } else std::fill(decoded+4,decoded+8,int8_t(0));
        } else std::memcpy(decoded,bank+int64_t(code)*8,8);
        uint32_t mask8=0;
        if constexpr (!delta_format && Format!=7 && Format!=9) {
            const uint32_t mask7=cpu_nvq_bits(w.aux,w.aux_bytes,
                (row*w.nsign+vector8)*7,7);
            const uint32_t parity=cpu_nvq_parity7(mask7) ^
                ((Format==2 && w.sign_mode!=0) ? ((code>>7)&1u) : 0u);
            mask8=mask7|(parity<<7);
        }
        for (int i=0; i<8; ++i) {
            int value=decoded[i];
            if constexpr (Format==1) value=8*value+delta;
            else if constexpr (Format==8) value=32*value+5*delta;
            else value*=1-2*int((mask8>>i)&1u);
            values[chunk*8+i]=static_cast<int8_t>(value);
        }
    }
}

using CpuNvqDecoder=void(*)(const CpuNvqDecodeView&,int64_t,int,uint32_t,int8_t*);
static CpuNvqDecoder cpu_nvq_decoder(int format) {
    switch (format) {
#define TYLOINFER_NVQ_CPU_CASE(F) case F: return cpu_decode_nvq_group<F>;
        TYLOINFER_NVQ_CPU_CASE(1) TYLOINFER_NVQ_CPU_CASE(2) TYLOINFER_NVQ_CPU_CASE(3)
        TYLOINFER_NVQ_CPU_CASE(5) TYLOINFER_NVQ_CPU_CASE(7) TYLOINFER_NVQ_CPU_CASE(8)
        TYLOINFER_NVQ_CPU_CASE(9) TYLOINFER_NVQ_CPU_CASE(10) TYLOINFER_NVQ_CPU_CASE(11)
        TYLOINFER_NVQ_CPU_CASE(12) TYLOINFER_NVQ_CPU_CASE(13) TYLOINFER_NVQ_CPU_CASE(14)
        TYLOINFER_NVQ_CPU_CASE(15)
#undef TYLOINFER_NVQ_CPU_CASE
        default: throw std::runtime_error("unsupported CPU NVQ format");
    }
}

CpuProjectionRows make_cpu_projection_rows(const NvqWeight& w) {
    TYLOINFER_RUNTIME_CHECK(w.indices_packed.is_cpu() && w.gs==24 && w.sub_bits>0 && w.sub_bits<=4,
        "original-FP32 NVQ rows require canonical CPU storage");
    CpuProjectionRows p; p.width=static_cast<int>(w.neuron_len); p.outputs=static_cast<int>(w.out);
    const int format=static_cast<int>(w.kernel_format);
    const int vector=cpu_nvq_d4(format) ? 4 : 8;
    auto& v=p.nvq;
    v.indices=w.indices_packed.data_ptr<uint8_t>(); v.aux=w.aux_packed.data_ptr<uint8_t>();
    v.index_bytes=w.indices_packed.numel(); v.aux_bytes=w.aux_packed.numel();
    v.nvec=static_cast<int>((w.neuron_len+vector-1)/vector);v.nsign=static_cast<int>((w.neuron_len+7)/8);
    v.groups=static_cast<int>(w.ng);v.sign_mode=static_cast<int>(w.sign_mode);
    v.states=w.sub_scale_packed.data_ptr<uint8_t>();v.state_bytes=w.sub_scale_packed.numel();
    v.anchors=w.neuron_scale.data_ptr<float>();v.width=w.neuron_len;v.state_bits=static_cast<int>(w.sub_bits);
    const auto* metadata=w.codebook.data_ptr<int8_t>();
    for(uint32_t state=0;state<(1u<<w.sub_bits);++state) {
        v.multipliers[state]=cpu_nvq_scale(metadata,format,1.0f,state);
        v.banks[state]=format==7 ? metadata+64+state*32 : format==9 ? metadata+64+state*64*8 : cpu_nvq_codebook(metadata,format,state);
    }
    p.nvq_rows=w.dense_groups ? mfq::cpu::nvq_dense_rows_dot_kernel(format) :
        (format==1 || format==8) && v.aux_bytes==0?nullptr:mfq::cpu::nvq_rows_dot_kernel(format);
    TYLOINFER_RUNTIME_CHECK(!w.dense_groups || p.nvq_rows,"compact NVQ CPU decoder unavailable");
    if(!p.nvq_rows) {
        const auto decoder=cpu_nvq_decoder(format);
        const auto dot=mfq::cpu::scaled_i8_dot_kernel();
        p.scalar_rows=[v,decoder,dot,format](const float* x,int batch,int stride,float* y,int out,int begin,int end) {
            for(int row=begin;row<end;++row) {
                for(int t=0;t<batch;++t) y[t*out+row]=0;
                for(int group=0;group<v.groups;++group) {
                    const auto state=cpu_nvq_state(v.indices,v.states,v.state_bytes,
                        int64_t(row)*v.groups+group,v.state_bits,format);
                    const float scale=v.anchors[row]*v.multipliers[state];
                    alignas(32) int8_t codes[24];decoder(v,row,group,state,codes);
                    const int valid=static_cast<int>(std::min<int64_t>(24,v.width-int64_t(group)*24));
                    for(int t=0;t<batch;++t) y[t*out+row]+=dot(codes,x+int64_t(t)*stride+group*24,valid,scale);
                }
            }
        };
    }
    return p;
}
static mfq_tensor_backend::Tensor nvq_matmul_cpu(
        const NvqWeight & w,
        mfq_tensor_backend::Tensor x) {
    TYLOINFER_RUNTIME_CHECK(!x.is_cuda(), "CPU NVQ GEMV requires CPU activations");
    TYLOINFER_RUNTIME_CHECK(
        !w.indices_packed.is_cuda() && !w.sub_scale_packed.is_cuda() &&
        !w.neuron_scale.is_cuda() && !w.codebook.is_cuda() && !w.aux_packed.is_cuda(),
        "CPU NVQ GEMV requires CPU-resident weights");
    TYLOINFER_RUNTIME_CHECK(w.gs == 24,
        "CPU NVQ GEMV currently requires 24-value groups");
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat32), w.neuron_len).contiguous();
    const int64_t rows = x.size(0);
    const float* input = x.data_ptr<float>();
    const auto dot = mfq::cpu::scaled_i8_dot_kernel();
    const int64_t outputs = w.out;
    const uint8_t * scales = w.sub_scale_packed.data_ptr<uint8_t>();
    const float * anchors = w.neuron_scale.data_ptr<float>();
    const int8_t * metadata = w.codebook.data_ptr<int8_t>();
    const int64_t scale_bytes = w.sub_scale_packed.numel();
    const int format=static_cast<int>(w.kernel_format);
    const auto decoder=cpu_nvq_decoder(format);
    const auto fused_rows=w.dense_groups ? mfq::cpu::nvq_dense_rows_dot_kernel(format) :
        (format==1 || format==8) && w.aux_packed.numel()==0?nullptr:mfq::cpu::nvq_rows_dot_kernel(format);
    TYLOINFER_RUNTIME_CHECK(!w.dense_groups || fused_rows,"compact NVQ CPU decoder unavailable");
    const int vector_size=cpu_nvq_d4(format) ? 4 : 8;
    TYLOINFER_RUNTIME_CHECK(w.sub_bits>0 && w.sub_bits<=4,"CPU NVQ state width must be 1..4");
    CpuNvqDecodeView view{w.indices_packed.data_ptr<uint8_t>(),
        w.aux_packed.data_ptr<uint8_t>(), w.indices_packed.numel(),w.aux_packed.numel(),
        static_cast<int>((w.neuron_len+vector_size-1)/vector_size),
        static_cast<int>((w.neuron_len+7)/8),static_cast<int>(w.ng),static_cast<int>(w.sign_mode)};
    for (uint32_t state=0; state<(1u<<w.sub_bits); ++state) {
        view.multipliers[state]=cpu_nvq_scale(metadata,format,1.0f,state);
        view.banks[state]=format==7 ? metadata+64+state*32 :
            format==9 ? metadata+64+state*64*8 : cpu_nvq_codebook(metadata,format,state);
    }
    if (fused_rows) {
        view.states=scales; view.state_bytes=scale_bytes;
        view.anchors=anchors; view.width=w.neuron_len;
        view.state_bits=static_cast<int>(w.sub_bits);
        auto result=mfq_tensor_backend::empty({rows,outputs},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32));
        auto* output=result.data_ptr<float>();
        mfq_parallel_for(0,outputs,1,[&](int64_t begin,int64_t end) {
            fused_rows(view,input,rows,w.neuron_len,output,outputs,begin,end);
        });
        return result.to(mfq_tensor_backend::kFloat16);
    }
    auto result=mfq_tensor_backend::empty({rows,outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    auto* output=result.data_ptr<mfq_half>();
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            for (int group = 0; group < w.ng; ++group) {
                const int64_t scale_linear = neuron * w.ng + group;
                const uint32_t state = cpu_nvq_state(view.indices,scales,scale_bytes,
                    scale_linear,static_cast<int>(w.sub_bits),format);
                const float scale = anchors[neuron]*view.multipliers[state];
                const int valid = static_cast<int>(std::min<int64_t>(
                    w.gs, w.neuron_len - static_cast<int64_t>(group) * w.gs));
                alignas(32) int8_t weights[24];
                decoder(view,neuron,group,state,weights);
                for (int64_t row = 0; row < rows; ++row) {
                    accumulators[static_cast<size_t>(row)] += dot(weights,
                        input + row*w.neuron_len + static_cast<int64_t>(group)*w.gs,
                        valid, scale);
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] =
                    mfq_half(accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}
mfq_tensor_backend::Tensor nvq_matmul(
        CudaProfiler& profiler,
        const NvqWeight& w,
        mfq_tensor_backend::Tensor x) {
    if (!x.is_cuda()) return nvq_matmul_cpu(w, std::move(x));
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    int M = (int)x.size(0);
    const NvqMatmulPath path = select_nvq_matmul_path(w, M);
    if (path == NvqMatmulPath::Gemv) {
        NvqWorkspace & ws = resident_nvq_workspace(w,M);
        return profiler.measure("nvq.gemv", [&]() {
            return nvq_gemv_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::Mmq) {
        NvqWorkspace & ws = resident_nvq_workspace(w,M);
        return profiler.measure("nvq.mma24", [&]() {
            return nvq_mmq_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::OnlineF16) {
        return profiler.measure("nvq.gemm_online_f16", [&]() {
            return nvq_gemm_f16_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode);
        });
    }
    auto weight = profiler.measure("nvq.dequant", [&]() { return nvq_dequant(w); });
    return profiler.measure("nvq.gemm", [&]() {
        return nint_cublas_gemm_nt_f32acc_cuda(x, weight);
    });
}

mfq_tensor_backend::Tensor nvq_matmul_input_mul(
    CudaProfiler& profiler,
    const NvqWeight & w,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor gate,
    int mode) {
    TYLOINFER_RUNTIME_CHECK(mode == 1 || mode == 2, "NVQ input gate mode must be 1(sigmoid) or 2(silu)");
    if (!x.is_cuda()) {
        x = x.contiguous().to(mfq_tensor_backend::kFloat16);
        gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
        TYLOINFER_RUNTIME_CHECK(x.sizes() == gate.sizes(), "NVQ x and gate shapes must match");
        auto value = mode == 1
            ? x * mfq_tensor_backend::sigmoid(gate)
            : x * mfq_tensor_backend::silu(gate);
        return nvq_matmul_cpu(w, value);
    }
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
    TYLOINFER_RUNTIME_CHECK(x.sizes() == gate.sizes(), "NVQ x and gate shapes must match");
    x = pad_last(x, w.neuron_len);
    gate = pad_last(gate, w.neuron_len);
    int M = (int)x.size(0);
    const NvqMatmulPath path = select_nvq_matmul_path(w, M);
    if (path == NvqMatmulPath::Gemv) {
        NvqWorkspace & ws = resident_nvq_workspace(w,M);
        return profiler.measure("nvq.gemv_gate", [&]() {
            return nvq_gemv_gate_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, gate, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, mode, ws.qx, ws.xscale);
        });
    }
    if (path == NvqMatmulPath::Mmq) {
        NvqWorkspace & ws = resident_nvq_workspace(w,M);
        return profiler.measure("nvq.mma24_gate", [&]() {
            return nvq_mmq_gate_ws_cuda(
                w.indices_packed, w.aux_packed, w.sub_scale_packed,
                w.neuron_scale, w.codebook, x, gate, w.neuron_len, w.gs,
                w.sub_bits, w.kernel_format, w.sign_mode, mode, ws.qx, ws.xscale);
        });
    }
    mfq_tensor_backend::Tensor value = mode == 1 ? x * mfq_tensor_backend::sigmoid(gate) : x * mfq_tensor_backend::silu(gate);
    return nvq_matmul(profiler, w, value);
}

bool nvq_pair_compatible(const NvqWeight & first, const NvqWeight & second) {
    return first.format == second.format && first.kernel_format == second.kernel_format &&
           first.gs == second.gs &&
           first.neuron_len == second.neuron_len && first.ng == second.ng;
}

bool nvq_fusion_enabled(const CudaExecutionConfig& config) {
    return config.nvq_fusion;
}

mfq_tensor_backend::Tensor nvq_matmul_multi2(
    CudaProfiler& profiler,
    const NvqWeight & first,
    const NvqWeight & second,
    mfq_tensor_backend::Tensor x) {
    if (!nvq_pair_compatible(first, second)) {
        throw std::runtime_error("NVQ multi-projection requires compatible formats and input layouts");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), first.neuron_len);
    const int M = (int)x.size(0);
    if (M > 8) {
        return mfq_tensor_backend::cat(
            {nvq_matmul(profiler, first, x),
             nvq_matmul(profiler, second, x)}, -1);
    }
    NvqWorkspace & ws = resident_nvq_workspace(first,M);
    return profiler.measure("nvq.gemv_multi2", [&]() {
        return nvq_gemv_multi2_ws_cuda(
            first.indices_packed, first.aux_packed, first.sub_scale_packed,
            first.neuron_scale, first.codebook,
            second.indices_packed, second.aux_packed, second.sub_scale_packed,
            second.neuron_scale, second.codebook,
            x, first.neuron_len, first.gs,
            first.sub_bits, first.kernel_format, first.sign_mode,
            second.sub_bits, second.kernel_format, second.sign_mode,
            ws.qx, ws.xscale);
    });
}

mfq_tensor_backend::Tensor nvq_matmul_swiglu(
    CudaProfiler& profiler,
    const NvqWeight & gate,
    const NvqWeight & up,
    mfq_tensor_backend::Tensor x) {
    if (!nvq_pair_compatible(gate, up) || gate.out != up.out) {
        throw std::runtime_error("NVQ SwiGLU requires compatible equal-width gate/up weights");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), gate.neuron_len);
    if (x.size(0) != 1) {
        auto pair = nvq_matmul_multi2(profiler, gate, up, x);
        auto parts = pair.split_with_sizes({gate.out, up.out}, -1);
        return mfq_tensor_backend::silu(parts[0]) * parts[1];
    }
    NvqWorkspace & ws = resident_nvq_workspace(gate,1);
    return profiler.measure("nvq.gemv_swiglu", [&]() {
        return nvq_gemv_swiglu_ws_cuda(
            gate.indices_packed, gate.aux_packed, gate.sub_scale_packed,
            gate.neuron_scale, gate.codebook,
            up.indices_packed, up.aux_packed, up.sub_scale_packed,
            up.neuron_scale, up.codebook,
            x, gate.neuron_len, gate.gs,
            gate.sub_bits, gate.kernel_format, gate.sign_mode,
            up.sub_bits, up.kernel_format, up.sign_mode,
            ws.qx, ws.xscale);
    });
}

mfq_tensor_backend::Tensor nvq_ffn_swiglu_down(
    CudaProfiler& profiler,
    const NvqWeight & gate,
    const NvqWeight & up,
    const NvqWeight & down,
    mfq_tensor_backend::Tensor x,
    MfqOptional<mfq_tensor_backend::Tensor> residual) {
    if (!nvq_pair_compatible(gate, up) || gate.out != up.out || gate.out != down.neuron_len) {
        throw std::runtime_error("NVQ fused FFN weight layouts are incompatible");
    }
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), gate.neuron_len);
    if (x.size(0) != 1 || (down.gs != 24 && down.gs != 28 && down.gs != 32)) {
        auto output = nvq_matmul(
            profiler, down,
            nvq_matmul_swiglu(profiler, gate, up, x));
        return residual.has_value()
            ? acc_cuda(residual.value(), output)
            : output;
    }
    NvqWorkspace & input_ws = resident_nvq_workspace(gate,1);
    NvqWorkspace & output_ws = resident_nvq_workspace(down,1);
    if (!input_ws.swiglu_scratch.defined() || input_ws.swiglu_scratch.numel() < gate.out) {
        input_ws.swiglu_scratch = mfq_tensor_backend::empty(
            {gate.out}, gate.indices_packed.options().dtype(
                mfq_tensor_backend::kFloat32));
    }
    profiler.measure("nvq.ffn_swiglu_quant", [&]() {
        nvq_ffn_swiglu_quant_ws_cuda(
            gate.indices_packed, gate.aux_packed, gate.sub_scale_packed,
            gate.neuron_scale, gate.codebook,
            up.indices_packed, up.aux_packed, up.sub_scale_packed,
            up.neuron_scale, up.codebook,
            x, gate.neuron_len, gate.gs,
            gate.sub_bits, gate.kernel_format, gate.sign_mode,
            up.sub_bits, up.kernel_format, up.sign_mode, down.gs,
            input_ws.qx, input_ws.xscale,
            output_ws.qx, output_ws.xscale, input_ws.swiglu_scratch);
        return 0;
    });
    return profiler.measure(
        residual.has_value() ? "nvq.gemv_qx_residual" : "nvq.gemv_qx",
        [&]() {
        if (residual.has_value()) {
            return nvq_gemv_qx_residual_ws_cuda(
                down.indices_packed, down.aux_packed, down.sub_scale_packed,
                down.neuron_scale, down.codebook,
                down.neuron_len, down.gs, down.sub_bits,
                down.kernel_format, down.sign_mode,
                output_ws.qx, output_ws.xscale, residual.value());
        }
        return nvq_gemv_qx_ws_cuda(
            down.indices_packed, down.aux_packed, down.sub_scale_packed,
            down.neuron_scale, down.codebook,
            down.neuron_len, down.gs, down.sub_bits, down.kernel_format, down.sign_mode,
            output_ws.qx, output_ws.xscale);
    });
}

mfq_tensor_backend::Tensor nvq_embedding(const NvqWeight & w, mfq_tensor_backend::Tensor token_ids) {
    return nvq_embedding_lookup_cuda(
        w.indices_packed, w.aux_packed, w.sub_scale_packed,
        w.neuron_scale, w.codebook, token_ids, w.neuron_len, w.gs,
        w.sub_bits, w.kernel_format, w.sign_mode);
}
static bool nvq_delta_format(int format) {
    return format == 1 || format == 8;
}

static bool nvq_no_aux_format(int format) {
    return format == 7 || format == 9;
}

static void copy_packed_bits(
        const std::vector<uint8_t> & source,
        size_t source_bit,
        std::vector<uint8_t> & destination,
        size_t destination_bit,
        size_t bit_count) {
    if (bit_count == 0) return;
    if (source_bit + bit_count >
            source.size() * 8 ||
        destination_bit + bit_count >
            destination.size() * 8) {
        throw std::runtime_error(
            "packed tensor-parallel bit copy is out of bounds");
    }
    if ((source_bit & 7u) == 0 &&
        (destination_bit & 7u) == 0) {
        const size_t full_bytes =
            bit_count / 8;
        if (full_bytes != 0) {
            std::memcpy(
                destination.data() +
                    destination_bit / 8,
                source.data() +
                    source_bit / 8,
                full_bytes);
            source_bit += full_bytes * 8;
            destination_bit += full_bytes * 8;
            bit_count -= full_bytes * 8;
        }
    }
    while (bit_count >= 8) {
        const size_t source_byte =
            source_bit >> 3;
        const int source_shift =
            static_cast<int>(source_bit & 7u);
        uint16_t source_word =
            source[source_byte];
        if (source_shift != 0 &&
            source_byte + 1 < source.size()) {
            source_word |=
                static_cast<uint16_t>(
                    source[source_byte + 1])
                << 8;
        }
        const uint8_t value =
            static_cast<uint8_t>(
                source_word >> source_shift);
        const size_t destination_byte =
            destination_bit >> 3;
        const int destination_shift =
            static_cast<int>(
                destination_bit & 7u);
        destination[destination_byte] |=
            static_cast<uint8_t>(
                value << destination_shift);
        if (destination_shift != 0 &&
            destination_byte + 1 <
                destination.size()) {
            destination[destination_byte + 1] |=
                static_cast<uint8_t>(
                    value >>
                    (8 - destination_shift));
        }
        source_bit += 8;
        destination_bit += 8;
        bit_count -= 8;
    }
    for (size_t bit = 0; bit < bit_count; ++bit) {
        if ((source[source_bit >> 3] >>
             (source_bit & 7u)) & 1u) {
            destination[destination_bit >> 3] |=
                static_cast<uint8_t>(
                    1u << (destination_bit & 7u));
        }
        ++source_bit;
        ++destination_bit;
    }
}

static std::vector<uint8_t> slice_packed_rows(
        const std::vector<uint8_t> & source,
        int64_t source_rows,
        int64_t source_items_per_row,
        int64_t row_begin,
        int64_t row_end,
        int64_t item_begin,
        int64_t item_count,
        int bits) {
    if (bits == 0) return {};
    if (source_rows <= 0 || source_items_per_row <= 0 ||
        row_begin < 0 || row_begin >= row_end ||
        row_end > source_rows || item_begin < 0 ||
        item_count <= 0 ||
        item_begin + item_count > source_items_per_row) {
        throw std::runtime_error("invalid packed tensor-parallel slice");
    }
    const int64_t destination_rows = row_end - row_begin;
    const size_t destination_items =
        static_cast<size_t>(destination_rows) * item_count;
    std::vector<uint8_t> destination(
        (destination_items * static_cast<size_t>(bits) + 7) / 8,
        0);
    for (int64_t row = 0; row < destination_rows; ++row) {
        const size_t source_item =
            static_cast<size_t>(row + row_begin) *
                source_items_per_row +
            static_cast<size_t>(item_begin);
        const size_t destination_item =
            static_cast<size_t>(row) * item_count;
        copy_packed_bits(
            source,
            source_item * static_cast<size_t>(bits),
            destination,
            destination_item * static_cast<size_t>(bits),
            static_cast<size_t>(item_count) *
                static_cast<size_t>(bits));
    }
    return destination;
}

mfq::nvq::Data slice_nvq_cpu(
        const mfq::nvq::Data & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NVQ");
    if (axis != TensorParallelAxis::Output &&
        axis != TensorParallelAxis::Input) {
        throw std::runtime_error("NVQ shard requires an output or input axis");
    }
    mfq::nvq::Data result = source;
    int64_t row_begin = 0;
    int64_t row_end = source.out;
    int64_t group_begin = 0;
    int64_t group_count = source.ng;
    if (axis == TensorParallelAxis::Output) {
        if (begin < 0 || begin >= end || end > source.out) {
            throw std::runtime_error("invalid NVQ output shard");
        }
        row_begin = begin;
        row_end = end;
        result.out = static_cast<int>(end - begin);
        result.shape[0] = result.out;
    } else {
        if (begin < 0 || begin >= end || end > source.ng) {
            throw std::runtime_error("invalid NVQ input shard");
        }
        group_begin = begin;
        group_count = end - begin;
        result.ng = static_cast<int>(group_count);
        const int64_t element_begin = begin * source.gs;
        const int64_t element_end =
            std::min<int64_t>(end * source.gs, source.neuron_len);
        result.neuron_len =
            static_cast<int>(element_end - element_begin);
        result.shape[1] = result.neuron_len;
        const int vector_size = mfq::nvq::vector_size(source.format);
        if (element_begin % vector_size != 0 ||
            element_begin % 8 != 0) {
            throw std::runtime_error(
                "NVQ input shard does not preserve vector boundaries");
        }
        result.nvec =
            (result.neuron_len + vector_size - 1) / vector_size;
        result.nsign = (result.neuron_len + 7) / 8;
    }

    result.neuron_scale_h.assign(
        source.neuron_scale_h.begin() +
            static_cast<ptrdiff_t>(row_begin),
        source.neuron_scale_h.begin() +
            static_cast<ptrdiff_t>(row_end));
    result.sub_scale_packed = slice_packed_rows(
        source.sub_scale_packed,
        source.out, source.ng,
        row_begin, row_end,
        group_begin, group_count,
        source.sub_bits);

    const int vector_size = mfq::nvq::vector_size(source.format);
    const int64_t element_begin = group_begin * source.gs;
    const int64_t vector_begin = element_begin / vector_size;
    const int64_t vector_count =
        axis == TensorParallelAxis::Output
        ? source.nvec
        : result.nvec;
    result.indices_packed = slice_packed_rows(
        source.indices_packed,
        source.out, source.nvec,
        row_begin, row_end,
        vector_begin, vector_count,
        mfq::nvq::index_bits(source.format));

    if (nvq_delta_format(source.format)) {
        result.aux_packed = slice_packed_rows(
            source.aux_packed,
            source.out, source.ng,
            row_begin, row_end,
            group_begin, group_count,
            1);
    } else if (nvq_no_aux_format(source.format)) {
        result.aux_packed.clear();
    } else {
        const int64_t sign_begin = element_begin / 8;
        const int64_t sign_count =
            axis == TensorParallelAxis::Output
            ? source.nsign
            : result.nsign;
        result.aux_packed = slice_packed_rows(
            source.aux_packed,
            source.out, source.nsign,
            row_begin, row_end,
            sign_begin, sign_count,
            7);
    }
    return result;
}

constexpr int64_t kNvq2JscXlGroupExecKernelFormat = 16;
constexpr int64_t kNvq3JscLGroupExecKernelFormat = 17;

static std::vector<uint8_t> repack_nvq2_exec_metadata(const mfq::nvq::Data & c) {
    if ((c.format != 2 && c.format != 5) || c.nvec != c.nsign) {
        throw std::runtime_error("NVQ2 execution metadata requires one sign record per vector");
    }
    const size_t count = (size_t)c.out * c.nvec;
    if (c.indices_packed.size() != count) {
        throw std::runtime_error("NVQ2 index stream length mismatch during execution repack");
    }
    std::vector<uint8_t> metadata(count * 2);
    for (size_t linear = 0; linear < count; ++linear) {
        const size_t bit = linear * 7;
        const size_t byte = bit >> 3;
        const int shift = (int)(bit & 7);
        uint16_t word = c.aux_packed[byte];
        if (byte + 1 < c.aux_packed.size()) {
            word |= (uint16_t)c.aux_packed[byte + 1] << 8;
        }
        const uint8_t index = c.indices_packed[linear];
        const uint8_t mask7 = (uint8_t)((word >> shift) & 0x7f);
        uint8_t parity = mask7;
        parity ^= parity >> 4;
        parity ^= parity >> 2;
        parity ^= parity >> 1;
        const uint8_t last = (parity & 1u) ^
            (c.sign_mode ? ((index >> 7) & 1u) : 0u);
        metadata[2 * linear] = c.indices_packed[linear];
        metadata[2 * linear + 1] = mask7 | (last << 7);
    }
    return metadata;
}

static uint32_t load_compact_bits_cpu(
        const std::vector<uint8_t> & data, size_t bit, int bits) {
    const size_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t word = byte < data.size() ? data[byte] : 0;
    if (byte + 1 < data.size()) word |= static_cast<uint32_t>(data[byte + 1]) << 8;
    if (byte + 2 < data.size()) word |= static_cast<uint32_t>(data[byte + 2]) << 16;
    return (word >> shift) & ((1u << bits) - 1u);
}

static void put_compact_bits96(
        uint32_t (&words)[3], int bit, int bits, uint32_t value) {
    const int word = bit >> 5;
    const int shift = bit & 31;
    words[word] |= value << shift;
    if (shift + bits > 32) words[word + 1] |= value >> (32 - shift);
}

static uint32_t compact_parity7(uint32_t value) {
    value &= 0x7fu;
    value ^= value >> 4;
    value ^= value >> 2;
    value ^= value >> 1;
    return value & 1u;
}

// NVQ2J-XL needs exactly 64 bits per gs24 group (three 12-bit indices, three
// 8-bit sign masks, and one 4-bit state). NVQ3J-L stores three indices in
// a 96-bit record per gs24 group (six 10-bit indices, three 8-bit sign masks,
// and one 4-bit state).
// This replaces the two independently bit-packed streams on CUDA; it does not
// keep a second execution copy of the model weights.
static std::vector<uint8_t> repack_extended_jsc_group_metadata(
        const mfq::nvq::Data & c,
        const std::array<uint8_t, 16> * state_remap = nullptr,
        const std::array<std::array<uint16_t, 4096>, 4> * index_remap = nullptr) {
    if (c.format != 14 && c.format != 15) {
        throw std::runtime_error("extended JSC group metadata requires NVQ2J-XL or NVQ3J-L");
    }
    const int index_bits = c.format == 14 ? 12 : 10;
    const int vectors_per_group = c.format == 14 ? 3 : 6;
    const size_t record_bytes = c.format == 14 ? 8 : 12;
    const size_t group_count = static_cast<size_t>(c.out) * c.ng;
    std::vector<uint8_t> metadata(group_count * record_bytes, 0);
    mfq_parallel_for(0, static_cast<int64_t>(group_count), 4096, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t row = linear / c.ng;
            const int group = static_cast<int>(linear - row * c.ng);
            const size_t vector_base = static_cast<size_t>(row) * c.nvec +
                static_cast<size_t>(group) * vectors_per_group;
            const size_t sign_base = static_cast<size_t>(row) * c.nsign +
                static_cast<size_t>(group) * 3;
            const uint32_t source_state = load_compact_bits_cpu(
                c.sub_scale_packed, static_cast<size_t>(linear) * c.sub_bits,
                c.sub_bits);
            const uint32_t state = state_remap == nullptr
                ? source_state
                : (*state_remap)[source_state];
            uint8_t * destination = metadata.data() +
                static_cast<size_t>(linear) * record_bytes;
            if (c.format == 14) {
                uint64_t packed = 0;
                for (int local = 0; local < 3; ++local) {
                    const size_t vector = vector_base + local;
                    const uint32_t source_index = vector < static_cast<size_t>((row + 1) * c.nvec)
                        ? load_compact_bits_cpu(c.indices_packed, vector * index_bits, index_bits)
                        : 0;
                    const uint32_t source_bank = static_cast<uint8_t>(
                        c.codebook[36 + source_state]);
                    const uint32_t index = index_remap == nullptr
                        ? source_index
                        : (*index_remap)[source_bank][source_index];
                    const size_t sign = sign_base + local;
                    const uint32_t mask7 = sign < static_cast<size_t>((row + 1) * c.nsign)
                        ? load_compact_bits_cpu(c.aux_packed, sign * 7, 7)
                        : 0;
                    const uint32_t mask8 = mask7 | (compact_parity7(mask7) << 7);
                    const uint32_t segment = index | (mask8 << index_bits);
                    packed |= static_cast<uint64_t>(segment) << (local * 20);
                }
                packed |= static_cast<uint64_t>(state) << 60;
                std::memcpy(destination, &packed, sizeof(packed));
            } else {
                uint32_t words[3] = {0, 0, 0};
                for (int local = 0; local < 6; ++local) {
                    const size_t vector = vector_base + local;
                    const uint32_t index = vector < static_cast<size_t>((row + 1) * c.nvec)
                        ? load_compact_bits_cpu(c.indices_packed, vector * index_bits, index_bits)
                        : 0;
                    put_compact_bits96(words, local * index_bits, index_bits, index);
                }
                for (int local = 0; local < 3; ++local) {
                    const size_t sign = sign_base + local;
                    const uint32_t mask7 = sign < static_cast<size_t>((row + 1) * c.nsign)
                        ? load_compact_bits_cpu(c.aux_packed, sign * 7, 7)
                        : 0;
                    const uint32_t mask8 = mask7 | (compact_parity7(mask7) << 7);
                    put_compact_bits96(words, 60 + local * 8, 8, mask8);
                }
                put_compact_bits96(words, 84, 4, state);
                std::memcpy(destination, words, sizeof(words));
            }
        }
    });
    return metadata;
}

static bool nvq2_exec_layout_enabled(const CudaExecutionConfig& config) {
    return config.nvq2_exec;
}

static bool extended_jsc_group_layout_enabled(
        const CudaExecutionConfig& config) {
    return config.nvq_extended_group_exec;
}

static std::vector<int8_t> reorder_extended_e8_codebook_for_stage3(
        const mfq::nvq::Data & c,
        std::array<uint8_t, 16> & state_remap,
        std::array<std::array<uint16_t, 4096>, 4> & index_remap) {
    constexpr size_t metadata_bytes = 64;
    constexpr size_t bank_bytes = 4096 * 8;
    constexpr int bank_count = 4;
    if (c.format != 14 ||
        c.codebook.size() != metadata_bytes + bank_count * bank_bytes ||
        static_cast<uint8_t>(c.codebook[1]) != bank_count) {
        throw std::runtime_error("extended E8 runtime layout requires NVQ2J-XL");
    }

    uint64_t counts[bank_count] = {0, 0, 0, 0};
    std::vector<uint64_t> index_counts(bank_count * 4096, 0);
    const size_t groups = static_cast<size_t>(c.out) * c.ng;
    for (size_t linear = 0; linear < groups; ++linear) {
        const uint32_t state = load_compact_bits_cpu(
            c.sub_scale_packed, linear * c.sub_bits, c.sub_bits);
        const uint32_t bank = static_cast<uint8_t>(c.codebook[36 + state]);
        if (bank >= bank_count) {
            throw std::runtime_error("NVQ2J-XL state references an invalid bank");
        }
        ++counts[bank];
        const size_t row = linear / c.ng;
        const size_t group = linear - row * c.ng;
        const size_t vector_base = row * c.nvec + group * 3;
        const size_t vector_end = (row + 1) * c.nvec;
        for (int local = 0; local < 3; ++local) {
            const size_t vector = vector_base + local;
            if (vector >= vector_end) break;
            const uint32_t index = load_compact_bits_cpu(
                c.indices_packed, vector * 12, 12);
            ++index_counts[bank * 4096 + index];
        }
    }
    std::array<int, bank_count> order = {0, 1, 2, 3};
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
        return counts[left] > counts[right];
    });
    std::vector<int8_t> reordered = c.codebook;
    int runtime_bank[bank_count] = {0, 0, 0, 0};
    for (int destination = 0; destination < bank_count; ++destination) {
        const int source = order[destination];
        runtime_bank[source] = destination;
        std::array<int, 4096> index_order;
        std::iota(index_order.begin(), index_order.end(), 0);
        std::stable_sort(index_order.begin(), index_order.end(),
            [&](int left, int right) {
                return index_counts[source * 4096 + left] >
                    index_counts[source * 4096 + right];
            });
        for (int destination_index = 0;
             destination_index < 4096;
             ++destination_index) {
            const int source_index = index_order[destination_index];
            index_remap[source][source_index] =
                static_cast<uint16_t>(destination_index);
            std::memcpy(
                reordered.data() + metadata_bytes + destination * bank_bytes +
                    destination_index * 8,
                c.codebook.data() + metadata_bytes + source * bank_bytes +
                    source_index * 8,
                8);
        }
    }
    int bank_state_count[bank_count] = {0, 0, 0, 0};
    for (int source_state = 0; source_state < 16; ++source_state) {
        const uint8_t source_bank = static_cast<uint8_t>(
            c.codebook[36 + source_state]);
        const int rank_in_bank = bank_state_count[source_bank]++;
        if (rank_in_bank >= 4) {
            throw std::runtime_error(
                "NVQ2J-XL runtime bank has more than four scale states");
        }
        const int destination_bank = runtime_bank[source_bank];
        const int destination_state = rank_in_bank * bank_count + destination_bank;
        state_remap[source_state] = static_cast<uint8_t>(destination_state);
        std::memcpy(
            reordered.data() + 4 + destination_state * sizeof(uint16_t),
            c.codebook.data() + 4 + source_state * sizeof(uint16_t),
            sizeof(uint16_t));
        reordered[36 + destination_state] = static_cast<int8_t>(destination_bank);
    }
    for (int bank = 0; bank < bank_count; ++bank) {
        if (bank_state_count[bank] != 4) {
            throw std::runtime_error(
                "NVQ2J-XL runtime bank must have four scale states");
        }
    }
    return reordered;
}

static std::vector<uint8_t> remap_extended_e8_sub_scale_states(
        const mfq::nvq::Data & c,
        const std::array<uint8_t, 16> & state_remap) {
    if (c.format != 14 || c.sub_bits != 4) {
        throw std::runtime_error(
            "extended E8 scale-state remap requires 4-bit NVQ2J-XL states");
    }
    const size_t state_count = static_cast<size_t>(c.out) * c.ng;
    const size_t expected_bytes = (state_count + 1) / 2;
    if (c.sub_scale_packed.size() != expected_bytes) {
        throw std::runtime_error("NVQ2J-XL scale-state stream length mismatch");
    }
    std::vector<uint8_t> remapped(expected_bytes, 0);
    mfq_parallel_for(0, static_cast<int64_t>(expected_bytes), 4096,
        [&](int64_t begin, int64_t end) {
            for (int64_t byte = begin; byte < end; ++byte) {
                const uint8_t source = c.sub_scale_packed[byte];
                const size_t low_state = static_cast<size_t>(byte) * 2;
                const uint8_t low = state_remap[source & 0x0fu];
                const uint8_t high = low_state + 1 < state_count
                    ? state_remap[source >> 4]
                    : 0;
                remapped[byte] = low | static_cast<uint8_t>(high << 4);
            }
        });
    return remapped;
}

NvqWeight to_device_nvq(
        const mfq::nvq::Data & c,
        bool cuda,
        const CudaExecutionConfig& config) {
    NvqWeight w;
    std::array<uint8_t, 16> e8_state_remap = {};
    std::unique_ptr<std::array<std::array<uint16_t,4096>,4>> e8_index_remap;
    std::vector<int8_t> e8_runtime_codebook;
    std::vector<uint8_t> e8_runtime_sub_scale;
    const bool use_extended_e8 =
        cuda && c.format == 14 &&
        extended_jsc_group_layout_enabled(config);
    if (use_extended_e8) {
        e8_index_remap=std::make_unique<std::array<std::array<uint16_t,4096>,4>>();
        e8_runtime_codebook = reorder_extended_e8_codebook_for_stage3(
            c, e8_state_remap, *e8_index_remap);
        e8_runtime_sub_scale = remap_extended_e8_sub_scale_states(
            c, e8_state_remap);
    }
    w.format = c.format;
    w.kernel_format = c.format;
    w.sign_mode = c.sign_mode;
    w.sub_bits = c.sub_bits;
    w.gs = c.gs;
    w.out = c.out;
    w.ng = c.ng;
    w.neuron_len = c.neuron_len;
    w.shape = c.shape;
    if (cuda && (c.format == 14 || c.format == 15) &&
        extended_jsc_group_layout_enabled(config)) {
        auto metadata = repack_extended_jsc_group_metadata(
            c,
            use_extended_e8 ? &e8_state_remap : nullptr,
            use_extended_e8 ? e8_index_remap.get() : nullptr);
        w.indices_packed = cpu_u8_tensor(
            metadata, {(int64_t)metadata.size()});
        w.aux_packed = mfq_tensor_backend::empty(
            {0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8));
        w.kernel_format = c.format == 14
            ? kNvq2JscXlGroupExecKernelFormat
            : kNvq3JscLGroupExecKernelFormat;
    } else if (cuda && (c.format == 2 || c.format == 5) &&
               nvq2_exec_layout_enabled(config)) {
        auto metadata = repack_nvq2_exec_metadata(c);
        w.indices_packed = cpu_u8_tensor(
            metadata, {(int64_t)metadata.size()});
        w.aux_packed = mfq_tensor_backend::empty(
            {0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8));
        w.kernel_format = c.format == 5 ? 6 : 4;
    } else if (
        c.format == 10 && c.codebook.size() >= 64 &&
        c.codebook[3] == 1 && c.codebook[1] == 2) {
        w.indices_packed = cpu_u8_tensor(
            c.indices_packed, {(int64_t)c.indices_packed.size()});
        w.aux_packed = cpu_u8_tensor(
            c.aux_packed, {(int64_t)c.aux_packed.size()});
        w.kernel_format = 11;
    } else {
        w.indices_packed = cpu_u8_tensor(
            c.indices_packed, {(int64_t)c.indices_packed.size()});
        w.aux_packed = cpu_u8_tensor(
            c.aux_packed, {(int64_t)c.aux_packed.size()});
    }
    w.sub_scale_packed = use_extended_e8
        ? cpu_u8_tensor(
            e8_runtime_sub_scale, {(int64_t)e8_runtime_sub_scale.size()})
        : cpu_u8_tensor(
            c.sub_scale_packed, {(int64_t)c.sub_scale_packed.size()});
    w.neuron_scale = cpu_f16_to_f32_tensor(c.neuron_scale_h, c.out);
    if (
        c.format == 5 || c.format == 7 || c.format == 9 ||
        c.format == 10 || c.format == 12 ||
        c.format == 13 || c.format == 14 || c.format == 15) {
        if (use_extended_e8) {
            w.codebook = cpu_i8_tensor(
                e8_runtime_codebook, {(int64_t)e8_runtime_codebook.size()});
        } else {
            w.codebook = cpu_i8_tensor(
                c.codebook, {(int64_t)c.codebook.size()});
        }
    } else {
        const int dims = mfq::nvq::vector_size(c.format);
        const int entries = c.format == 1 ? 2048 : (c.format == 8 ? 1024 : 256);
        w.codebook = cpu_i8_tensor(c.codebook, {entries, dims});
    }
    // Retain the immutable map before upload. Decode and graph capture must
    // never copy codebook metadata back from the device.
    if ((w.kernel_format==5 || w.kernel_format==13 || w.kernel_format==14 ||
         w.kernel_format==10 || w.kernel_format==12 || w.kernel_format==15) &&
        w.codebook.numel()>=64) {
        const auto* metadata=w.codebook.data_ptr<int8_t>();
        for(unsigned state=0;state<16;++state)
            w.bank_map|=uint32_t(uint8_t(metadata[36+state])&3u)<<(2*state);
        if((w.kernel_format==12 || w.kernel_format==15) && metadata[1]==2 && metadata[3]==1) {
            static std::atomic<uint32_t> reported{0};const uint32_t bit=1u<<w.kernel_format;
            if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
                "mfe_d4_analytic_scale_metadata scope=first_validated_table format=%d banks=2 mode=1\n",
                int(w.kernel_format));
        }
    }
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        w.indices_packed = w.indices_packed.to(target).contiguous();
        w.aux_packed = w.aux_packed.to(target).contiguous();
        w.sub_scale_packed =
            w.sub_scale_packed.to(target).contiguous();
        if(w.kernel_format==16 || w.kernel_format==17)
            w.sub_scale_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
        w.neuron_scale = w.neuron_scale.to(target).contiguous();
        w.codebook = w.codebook.to(target).contiguous();
        prepare_nvq_decode_records(w,true);
        (void)resident_nvq_workspace(w,1);
    }
    return w;
}

NvqWeight to_gpu_nvq(
        const mfq::nvq::Data & c,
        const CudaExecutionConfig& config) {
    return to_device_nvq(c, true, config);
}

NvqWeight to_cuda_device_nvq(
        const mfq::nvq::Data & c,
        int device,
        const CudaExecutionConfig& config) {
    MfqCudaGuard guard(device);
    return to_device_nvq(c, true, config);
}

NvqWeight to_cpu_nvq(const mfq::nvq::Data & c) {
    return to_device_nvq(c, false);
}
static size_t packed_nbytes(size_t count, int bits) {
    return bits == 0 ? 0 : (count * (size_t)bits + 7) / 8;
}

NepqWeight to_device_nepq(const mfq::nepq::Data & cpu, bool cuda) {
    NepqWeight value;
    value.format = cpu.format;
    value.state_bits = cpu.state_bits;
    value.n_experts = cpu.n_experts;
    value.out_per_expert = cpu.out_per_expert;
    value.neuron_len = cpu.neuron_len;
    value.ng = cpu.ng;
    value.rotation_block = cpu.rotation_block;
    value.rotation_seed = cpu.rotation_seed;
    value.residual = cpu.residual;
    value.residual_position_bits = cpu.residual_position_bits;
    value.residual_block_vectors = cpu.residual_block_vectors;
    value.indices_packed = cpu_u8_tensor(
        cpu.indices_packed, {(int64_t)cpu.indices_packed.size()});
    value.aux_packed = cpu_u8_tensor(
        cpu.aux_packed, {(int64_t)cpu.aux_packed.size()});
    value.state_packed = cpu_u8_tensor(
        cpu.state_packed, {(int64_t)cpu.state_packed.size()});
    value.neuron_scale = cpu_f16_to_f32_tensor(
        cpu.neuron_scale_h, cpu.n_experts * cpu.out_per_expert);
    value.table_pool = cpu_i8_tensor(
        cpu.table_pool, {cpu.bank_count, cpu.runtime_table_bytes});
    const int grouped_stride =
        (cpu.profile == 0 || cpu.profile == 4)
        ? 2112 : cpu.runtime_table_bytes;
    value.grouped_table_pool = cpu_i8_tensor(
        cpu.grouped_table_pool, {cpu.bank_count, grouped_stride});
    value.bank_ids = cpu_u8_tensor(
        cpu.bank_ids, {
            (int64_t)cpu.n_experts * cpu.out_per_expert, cpu.nsuper});
    value.rotation_signs = cpu_i8_tensor(
        cpu.rotation_signs, {(int64_t)cpu.rotation_signs.size()});
    if (cpu.residual) {
        value.residual_codebook = cpu_f16_tensor(
            cpu.residual_codebook_h, {1024, 8});
        value.residual_first = mfq_tensor_backend::from_blob(
            (void *)cpu.residual_first.data(),
            {cpu.n_experts * cpu.out_per_expert,
             cpu.residual_blocks_per_row},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt16)).clone();
        value.residual_second = mfq_tensor_backend::from_blob(
            (void *)cpu.residual_second_dense.data(),
            {cpu.n_experts * cpu.out_per_expert,
             cpu.residual_blocks_per_row},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt16)).clone();
    }
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            mfq_current_cuda_device());
        value.indices_packed =
            value.indices_packed.to(target).contiguous();
        value.aux_packed =
            value.aux_packed.to(target).contiguous();
        value.state_packed =
            value.state_packed.to(target).contiguous();
        value.neuron_scale =
            value.neuron_scale.to(target).contiguous();
        value.table_pool =
            value.table_pool.to(target).contiguous();
        value.grouped_table_pool =
            value.grouped_table_pool.to(target).contiguous();
        value.bank_ids =
            value.bank_ids.to(target).contiguous();
        value.rotation_signs =
            value.rotation_signs.to(target).contiguous();
        if (cpu.residual) {
            value.residual_codebook =
                value.residual_codebook.to(target).contiguous();
            value.residual_first =
                value.residual_first.to(target).contiguous();
            value.residual_second =
                value.residual_second.to(target).contiguous();
        }
    }
    return value;
}

NepqWeight to_gpu_nepq(const mfq::nepq::Data & cpu) {
    return to_device_nepq(cpu, true);
}

NepqWeight to_cpu_nepq(const mfq::nepq::Data & cpu) {
    return to_device_nepq(cpu, false);
}
mfq::nvq::Data select_nvq_cpu_rows(
        const mfq::nvq::Data & source,
        const std::vector<int64_t> & rows) {
    if (source.shape.size() != 2 ||
        source.axis != 0 ||
        source.shape[0] != source.out ||
        rows.empty()) {
        throw std::runtime_error(
            "invalid NVQ MoE output shard source");
    }
    mfq::nvq::Data result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    result.neuron_scale_h.resize(rows.size());
    const int index_bits =
        mfq::nvq::index_bits(source.format);
    const bool delta =
        nvq_delta_format(source.format);
    const bool no_aux =
        nvq_no_aux_format(source.format);
    const int aux_items =
        delta ? source.ng :
        (no_aux ? 0 : source.nsign);
    const int aux_bits =
        delta ? 1 : (no_aux ? 0 : 7);
    result.sub_scale_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.sub_bits),
        0);
    result.indices_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.nvec),
            index_bits),
        0);
    result.aux_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(aux_items),
            aux_bits),
        0);
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source.out) {
            throw std::runtime_error(
                "NVQ MoE shard row is out of range");
        }
        copy_packed_bits(
            source.sub_scale_packed,
            static_cast<size_t>(source_row) *
                source.ng * source.sub_bits,
            result.sub_scale_packed,
            destination * source.ng *
                source.sub_bits,
            static_cast<size_t>(source.ng) *
                source.sub_bits);
        copy_packed_bits(
            source.indices_packed,
            static_cast<size_t>(source_row) *
                source.nvec * index_bits,
            result.indices_packed,
            destination * source.nvec *
                index_bits,
            static_cast<size_t>(source.nvec) *
                index_bits);
        if (aux_bits != 0) {
            copy_packed_bits(
                source.aux_packed,
                static_cast<size_t>(source_row) *
                    aux_items * aux_bits,
                result.aux_packed,
                destination * aux_items *
                    aux_bits,
                static_cast<size_t>(aux_items) *
                    aux_bits);
        }
        result.neuron_scale_h[destination] =
            source.neuron_scale_h[
                static_cast<size_t>(source_row)];
    }
    return result;
}

mfq::nepq::Data select_nepq_cpu_rows(
        const mfq::nepq::Data & source,
        const std::vector<int64_t> & rows,
        int output_per_expert,
        int selected_experts) {
    const int destination_experts = selected_experts < 0
        ? source.n_experts
        : selected_experts;
    const int source_rows =
        source.n_experts *
        source.out_per_expert;
    if (rows.empty() ||
        output_per_expert <= 0 ||
        destination_experts <= 0 ||
        static_cast<int>(rows.size()) !=
            destination_experts *
                output_per_expert) {
        throw std::runtime_error(
            "invalid NEPQ MoE output shard");
    }
    mfq::nepq::Data result = source;
    result.n_experts = destination_experts;
    result.out_per_expert = output_per_expert;
    result.neuron_scale_h.resize(rows.size());
    result.state_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.state_bits),
        0);
    result.indices_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.nvec),
            source.index_bits),
        0);
    result.aux_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.aux_bits),
        0);
    result.bank_ids.resize(
        rows.size() * source.nsuper);
    if (source.residual) {
        result.residual_first.resize(
            rows.size() * source.residual_blocks_per_row);
        result.residual_second_dense.resize(
            rows.size() * source.residual_blocks_per_row);
    }
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source_rows) {
            throw std::runtime_error(
                "NEPQ MoE shard row is out of range");
        }
        copy_packed_bits(
            source.state_packed,
            static_cast<size_t>(source_row) *
                source.ng * source.state_bits,
            result.state_packed,
            destination * source.ng *
                source.state_bits,
            static_cast<size_t>(source.ng) *
                source.state_bits);
        copy_packed_bits(
            source.indices_packed,
            static_cast<size_t>(source_row) *
                source.nvec * source.index_bits,
            result.indices_packed,
            destination * source.nvec *
                source.index_bits,
            static_cast<size_t>(source.nvec) *
                source.index_bits);
        if (source.aux_bits != 0) {
            copy_packed_bits(
                source.aux_packed,
                static_cast<size_t>(source_row) *
                    source.ng * source.aux_bits,
                result.aux_packed,
                destination * source.ng *
                    source.aux_bits,
                static_cast<size_t>(source.ng) *
                    source.aux_bits);
        }
        std::memcpy(
            result.bank_ids.data() +
                destination * source.nsuper,
            source.bank_ids.data() +
                static_cast<size_t>(source_row) *
                    source.nsuper,
            static_cast<size_t>(source.nsuper));
        result.neuron_scale_h[destination] =
            source.neuron_scale_h[
                static_cast<size_t>(source_row)];
        if (source.residual) {
            std::memcpy(
                result.residual_first.data()
                    + destination * source.residual_blocks_per_row,
                source.residual_first.data()
                    + static_cast<size_t>(source_row)
                        * source.residual_blocks_per_row,
                static_cast<size_t>(source.residual_blocks_per_row)
                    * sizeof(int16_t));
            std::memcpy(
                result.residual_second_dense.data()
                    + destination * source.residual_blocks_per_row,
                source.residual_second_dense.data()
                    + static_cast<size_t>(source_row)
                        * source.residual_blocks_per_row,
                static_cast<size_t>(source.residual_blocks_per_row)
                    * sizeof(int16_t));
        }
    }
    return result;
}
mfq_tensor_backend::Tensor nvq_dequant(const NvqWeight & w) {
    return nvq_dequant_cuda(
        w.indices_packed, w.aux_packed, w.sub_scale_packed,
        w.neuron_scale, w.codebook, w.neuron_len, w.gs,
        w.sub_bits, w.kernel_format, w.sign_mode);
}
bool nvq_fused_residual_format(std::int64_t kernel_format) {
    return kernel_format == kNvq2JscXlGroupExecKernelFormat ||
        kernel_format == kNvq3JscLGroupExecKernelFormat;
}
