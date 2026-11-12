#include "selected_attention.h"
#include "selected_raw.h"
#include "sparse_mma_raw.h"
#include "env.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
namespace mfq_selected_attention {
namespace raw = ::tyloi::cuda::selected_raw;
namespace mma = ::tyloi::cuda::sparse_mma_raw;
using AttentionLayout = raw::Layout;
using AttentionEpilogue = raw::Epilogue;
static mma::Hot hot_view(tyloi::cuda::HotKvView h) { return {h.keys,h.values,h.mapping,h.cells,h.cold_mapping,h.cold_cells}; }
static raw::Sigmoid sigmoid(bool high_precision) { return high_precision ? raw::Sigmoid::native_double : raw::Sigmoid::native_float; }
static DenseTensor run_mma(const DenseTensor& q,const DenseTensor& k,const DenseTensor& v,
    const DenseTensor& indices,const DenseTensor& mask,const DenseTensor& sinks,DenseTensor meta,
    double scale,const char* name,const DenseTensor& gate={},bool half_output=false,
    const DenseTensor& positions={},int64_t causal_columns=0,tyloi::cuda::HotKvView hot={}) {
    TYLOINFER_RUNTIME_CHECK(q.dim()==4 && k.dim()==4 && v.dim()==4 && q.size(0)<=INT_MAX && q.size(1)<=INT_MAX && q.size(2)<=INT_MAX && k.size(1)<=INT_MAX && k.size(2)<=INT_MAX,name,": attention dimensions exceed 32-bit indexing");
    TYLOINFER_RUNTIME_CHECK(q.size(3)==256 || q.size(3)==512,name,": tensor shapes disagree");
    const int B=int(q.size(0)),heads=int(q.size(1)),M=int(q.size(2)),kv_heads=int(k.size(1)),max_seq=int(k.size(2)),width=int(q.size(3));
    const bool causal=positions.defined();
    TYLOINFER_RUNTIME_CHECK(!causal || (causal_columns>0 && causal_columns<=INT_MAX-32),name,": causal selection exceeds 32-bit indexing");
    const int selected=causal ? int(((causal_columns+31)/32)*32) : int(indices.size(2));
    TYLOINFER_RUNTIME_CHECK(B>0 && M>0 && heads>0 && kv_heads>0 && heads%kv_heads==0 && max_seq>0 && selected>0,name,": invalid attention geometry");
    TYLOINFER_RUNTIME_CHECK((width==256 || width==512) && k.size(0)==B && k.size(3)==width && v.size(0)==B && v.size(1)==kv_heads && v.size(2)==max_seq && v.size(3)==width && (causal || (indices.dim()==3 && indices.size(0)==B && indices.size(1)==M)),name,": tensor shapes disagree");
    TYLOINFER_RUNTIME_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && (causal || indices.is_cuda()) && q.device()==k.device() && q.device()==v.device() && (causal || q.device()==indices.device()),name,": tensors must share one CUDA device");
    TYLOINFER_RUNTIME_CHECK(k.is_contiguous() && v.is_contiguous() && (causal || indices.is_contiguous()),name,": tensors must be contiguous");
    TYLOINFER_RUNTIME_CHECK((q.scalar_type()==::tyloi::kFloat32 || q.scalar_type()==::tyloi::kFloat16) && k.scalar_type()==::tyloi::kFloat16 && v.scalar_type()==::tyloi::kFloat16 && (causal || indices.scalar_type()==::tyloi::kInt32),name,": expected f32/f16/f16/i32 tensors");
    if(causal) TYLOINFER_RUNTIME_CHECK(positions.is_cuda() && positions.device()==q.device() && positions.is_contiguous() && positions.dim()==1 && positions.numel()==M && (positions.scalar_type()==::tyloi::kInt32 || positions.scalar_type()==::tyloi::kInt64) && !mask.defined() && !indices.defined(),name,": invalid implicit causal selection");
    if(mask.defined()) TYLOINFER_RUNTIME_CHECK(mask.is_cuda() && mask.device()==q.device() && mask.is_contiguous() && mask.scalar_type()==::tyloi::kFloat16 && mask.sizes()==indices.sizes(),name,": invalid mask");
    if(sinks.defined()) TYLOINFER_RUNTIME_CHECK(sinks.is_cuda() && sinks.device()==q.device() && sinks.is_contiguous() && sinks.scalar_type()==::tyloi::kFloat32 && sinks.numel()==heads,name,": invalid sinks");
    const MfqCudaGuard guard(q.device());
    auto plan=mma::plan(B,M,heads,kv_heads,max_seq,selected,width,width,hot.mapping!=nullptr);
    const size_t meta_elements=plan.workspace_bytes/sizeof(float);
    TYLOINFER_RUNTIME_CHECK(meta_elements<=size_t(std::numeric_limits<int64_t>::max()),name,": meta workspace is too large");
    if(!meta.defined()) meta=::tyloi::cuda::empty({int64_t(meta_elements)},q.options().dtype(::tyloi::kFloat32));
    TYLOINFER_RUNTIME_CHECK(meta.is_cuda() && meta.device()==q.device() && meta.is_contiguous() && meta.scalar_type()==::tyloi::kFloat32 && size_t(meta.numel())>=meta_elements,name,": meta workspace too small");
    auto out=::tyloi::cuda::empty({B,M,heads,width},q.options().dtype(::tyloi::kFloat32));
    const int64_t strides[]={q.stride(0),q.stride(1),q.stride(2),q.stride(3)};
    mma::launch(plan,q.data_ptr(),k.data_ptr(),v.data_ptr(),indices.defined()?indices.data_ptr<int32_t>():nullptr,mask.defined()?mask.data_ptr():nullptr,sinks.defined()?sinks.data_ptr<float>():nullptr,causal?positions.data_ptr():nullptr,strides,q.scalar_type()==::tyloi::kFloat16,q.is_contiguous(),causal && positions.scalar_type()==::tyloi::kInt64,int(causal_columns),hot_view(hot),float(scale),out.data_ptr<float>(),meta.data_ptr(),size_t(meta.numel())*sizeof(float),gate.defined(),mfq_current_cuda_stream());
    if(gate.defined()) return mma_reduce_gated(out,meta,gate,half_output,kv_heads,plan.blocks,plan.parts,1,16);
    return out;
}
void values(std::initializer_list<const DenseTensor*> tensors) {
    const auto* first = *tensors.begin();
    for (const auto* value : tensors) {
        TYLOINFER_RUNTIME_CHECK(value->defined() && value->is_cuda(),
            "selected attention requires CUDA tensors");
        TYLOINFER_RUNTIME_CHECK(value->device() == first->device(),
            "selected-attention tensors must share one CUDA device");
        const auto dtype = value->scalar_type();
        TYLOINFER_RUNTIME_CHECK(dtype == ::tyloi::kFloat32 || dtype == ::tyloi::kFloat16 ||
            dtype == ::tyloi::kBFloat16, "selected attention requires F32/F16/BF16 values");
    }
}

DenseTensor promoted_matmul(const DenseTensor& left, const DenseTensor& right) {
    // MLX matmul promotes mixed floating operands, unlike ATen matmul.
    const auto dtype = left.scalar_type() == right.scalar_type()
        ? left.scalar_type() : ::tyloi::kFloat32;
    return ::tyloi::cuda::matmul(left.to(dtype), right.to(dtype));
}

DenseTensor dots(const DenseTensor& query, const DenseTensor& pooled) {
    values({&query, &pooled});
    TYLOINFER_RUNTIME_CHECK(query.dim() == 4 && pooled.dim() == 3,
        "selected-attention scores require [B,T,H,D] query and [B,P,D] keys");
    TYLOINFER_RUNTIME_CHECK(query.size(0) == pooled.size(0) &&
        query.size(3) == pooled.size(2) && query.size(2) > 0 && query.size(3) > 0,
        "selected-attention score dimensions disagree");
    // QSA rotates [B,H,T,D] then permutes back to [B,T,H,D]. Native
    // batched GEMM requires the leading batch axes to have compact strides.
    return ::tyloi::cuda::matmul(query.to(::tyloi::kFloat32).contiguous(),
        pooled.to(::tyloi::kFloat32).transpose(-1, -2).unsqueeze(1)).clamp_min(0.0);
}

void attention_shapes(const DenseTensor& query, const DenseTensor& key, const DenseTensor& value) {
    values({&query, &key, &value});
    TYLOINFER_RUNTIME_CHECK(query.dim() == 4 && key.dim() == 4 && value.sizes() == key.sizes(),
        "selected attention requires [B,H,T,D] query/key/value");
    TYLOINFER_RUNTIME_CHECK(query.size(0) == key.size(0) && key.size(1) > 0 &&
        query.size(1) > 0 && query.size(1) % key.size(1) == 0 &&
        query.size(3) == key.size(3) && query.size(3) > 0,
        "selected-attention dimensions disagree");
}

static DenseTensor sparse_impl(const DenseTensor& q, const DenseTensor& k, const DenseTensor& v,
    const DenseTensor& selected, double scale, bool v_is_k_view,
    const DenseTensor& gate, bool half_output, bool fused_query = false, tyloi::cuda::HotKvView hot = {}) {
    attention_shapes(q, k, v);
    TYLOINFER_RUNTIME_CHECK(selected.defined() && selected.is_cuda() &&
        selected.device() == q.device() && selected.dim() == 3 &&
        selected.size(0) == q.size(0) && selected.size(1) == q.size(2) &&
        selected.size(2) > 0, "sparse indices must have [B,T,topk] shape");
    TYLOINFER_RUNTIME_CHECK(selected.scalar_type() == ::tyloi::kInt32 ||
        selected.scalar_type() == ::tyloi::kInt64, "sparse indices must be integer");
    TYLOINFER_RUNTIME_CHECK(std::isfinite(scale), "attention scale must be finite");
    MfqCudaGuard guard(q.device());
    const bool direct_query = fused_query && !v_is_k_view && q.size(3) == 256 &&
        (q.size(1) == 24 || selected.size(2) <= 512) &&
        (q.scalar_type() == ::tyloi::kFloat16 || q.scalar_type() == ::tyloi::kFloat32);
    auto query = direct_query ? q : q.to(::tyloi::kFloat32).contiguous();
    auto key = k.to(::tyloi::kFloat16).contiguous();
    auto value = v_is_k_view ? key : v.to(::tyloi::kFloat16).contiguous();
    auto indices = selected.to(::tyloi::kInt32).contiguous();
    if (q.size(0) == 0 || q.size(2) == 0) {
        return ::tyloi::cuda::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
            q.options().dtype(::tyloi::kFloat32));
    }
    if (key.size(2) == 0) {
        return ::tyloi::cuda::zeros({q.size(0), q.size(2), q.size(1), q.size(3)},
            q.options().dtype(::tyloi::kFloat32));
    }

    const bool use_qsa_mma = !v_is_k_view && q.size(3) == 256;
    const bool use_mla_mma = v_is_k_view && q.size(3) == 512;
    if (use_qsa_mma || use_mla_mma) {
        constexpr int index_tile = 32;
        const int64_t source_count = indices.size(2);
        TYLOINFER_RUNTIME_CHECK(
            source_count <= std::numeric_limits<int>::max() - index_tile &&
            key.size(2) <= std::numeric_limits<int>::max(),
            "sparse attention exceeds 32-bit index geometry");
        const int64_t padded_count =
            ((source_count + index_tile - 1) / index_tile) * index_tile;
        auto shape = indices.sizes().vec();
        shape[2] = padded_count;
        auto safe_indices = ::tyloi::cuda::empty(shape, indices.options());
        auto mask = ::tyloi::cuda::empty(shape, query.options().dtype(::tyloi::kFloat16));
        TYLOINFER_RUNTIME_CHECK(
            indices.size(0) <= std::numeric_limits<int64_t>::max() /
                indices.size(1),
            "sparse attention row count is too large");
        const int64_t rows = indices.size(0) * indices.size(1);
        TYLOINFER_RUNTIME_CHECK(
            rows <= std::numeric_limits<int64_t>::max() / padded_count,
            "sparse attention index workspace is too large");
        raw::prepare_indices(indices.data_ptr<int32_t>(),safe_indices.data_ptr<int32_t>(),mask.data_ptr(),rows,int(source_count),int(padded_count),int(key.size(2)),mfq_current_cuda_stream());
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
        if (use_qsa_mma) {
            if(hot.mapping)return run_mma(
                query,key,value,safe_indices,mask,DenseTensor{},DenseTensor{},scale,
                "qwen4_sparse_gqa_attention_hot",gate,half_output,DenseTensor{},0,hot);
            return run_mma(
                query, key, value, safe_indices, mask, DenseTensor{}, DenseTensor{},
                scale, "qwen4_sparse_gqa_attention", gate, half_output);
        }
        return run_mma(
            query, key, key, safe_indices, mask, DenseTensor{}, DenseTensor{},
            scale, "glm5_sparse_mla_attention");
    }

    auto output = ::tyloi::cuda::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
        q.options().dtype(::tyloi::kFloat32));
    if (output.numel() == 0) return output;
    const int64_t blocks = output.numel() / q.size(3);
    TYLOINFER_RUNTIME_CHECK(blocks <= std::numeric_limits<int>::max(),
        "sparse attention exceeds CUDA grid size");
    const auto stream = mfq_current_cuda_stream();
    raw::sparse(query.data_ptr(),key.data_ptr(),value.data_ptr(),indices.data_ptr<int32_t>(),output.data_ptr<float>(),q.size(0),q.size(1),k.size(1),q.size(2),q.size(3),k.size(2),selected.size(2),float(scale),false,nullptr,false,AttentionEpilogue{},hot_view(hot),stream);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

DenseTensor sparse(const DenseTensor& q, const DenseTensor& k, const DenseTensor& v,
    const DenseTensor& selected, double scale, bool v_is_k_view, tyloi::cuda::HotKvView hot) {
    return sparse_impl(q, k, v, selected, scale, v_is_k_view, DenseTensor{}, false, false, hot);
}

AttentionLayout attention_layout(const DenseTensor& x) {
    return {x.stride(0), x.stride(1), x.stride(2), x.stride(3)};
}

bool half_or_float(const DenseTensor& x) {
    return x.scalar_type() == ::tyloi::kFloat16 || x.scalar_type() == ::tyloi::kFloat32;
}

DenseTensor gated(const DenseTensor& attended, const DenseTensor& gate, bool half_output) {
    values({&attended, &gate});
    TYLOINFER_RUNTIME_CHECK(attended.dim() == 4 && attended.sizes() == gate.sizes(),
        "attention gate requires matching [B,T,H,D] inputs");
    if (!half_or_float(attended) || !half_or_float(gate))
        return (attended.to(::tyloi::kFloat32) * ::tyloi::cuda::sigmoid(gate.to(::tyloi::kFloat32)))
            .to(half_output ? ::tyloi::kFloat16 : ::tyloi::kFloat32);
    MfqCudaGuard guard(attended.device());
    auto output = ::tyloi::cuda::empty(attended.sizes(), attended.options().dtype(half_output ? ::tyloi::kFloat16 : ::tyloi::kFloat32));
    if (!output.numel()) return output;
    const AttentionEpilogue epilogue{gate.data_ptr(), output.data_ptr(),
        gate.scalar_type() == ::tyloi::kFloat32, half_output,
        sigmoid(gate.scalar_type() == ::tyloi::kFloat32 && !gate.is_contiguous()), attention_layout(gate)};
    raw::gated(attended.data_ptr(),attended.scalar_type()==::tyloi::kFloat16,attention_layout(attended),epilogue,output.numel(),output.size(2),output.size(1),output.size(3),mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

DenseTensor sparse_gated(const DenseTensor& q, const DenseTensor& k, const DenseTensor& v,
    const DenseTensor& indices, const DenseTensor& gate, double scale, bool half_output, tyloi::cuda::HotKvView hot) {
    attention_shapes(q, k, v);
    values({&q, &gate});
    TYLOINFER_RUNTIME_CHECK(gate.dim() == 4 && gate.size(0) == q.size(0) &&
        gate.size(1) == q.size(2) && gate.size(2) == q.size(1) &&
        gate.size(3) == q.size(3), "sparse attention gate geometry mismatch");
    // The QSA MMA epilogue combines stream-K normalization, gate and cast.
    // Keep the shared selected-attention arithmetic for all other widths.
    if (q.size(3) != 256 || !half_or_float(gate) || !q.numel() || !k.size(2))
        return gated(sparse(q, k, v, indices, scale, false,hot), gate, half_output);
    const char* fused_query = tyloi::cuda::env_lookup("TYLOINFER_QSA_SPARSE_QUERY_FUSED");
    return sparse_impl(q, k, v, indices, scale, false, gate, half_output,
        !fused_query || fused_query[0] != '0',hot);
}

DenseTensor block_indices_to_tokens(const DenseTensor& blocks, const DenseTensor& positions,
    int64_t pool, int64_t budget) {
    TYLOINFER_RUNTIME_CHECK(blocks.defined() && positions.defined() && blocks.is_cuda() &&
        positions.is_cuda() && blocks.device() == positions.device() &&
        blocks.dim() == 3 && positions.dim() == 1 && positions.numel() == blocks.size(1) &&
        (blocks.scalar_type() == ::tyloi::kInt32 || blocks.scalar_type() == ::tyloi::kInt64) &&
        (positions.scalar_type() == ::tyloi::kInt32 || positions.scalar_type() == ::tyloi::kInt64) &&
        pool > 0 && budget > 0 && pool <= INT_MAX && budget <= INT_MAX-pool &&
        blocks.size(1) <= INT_MAX && blocks.size(2) <= budget/pool,
        "block-selected token geometry mismatch");
    MfqCudaGuard guard(blocks.device());
    const int64_t columns = budget+pool-1;
    auto output = ::tyloi::cuda::empty({blocks.size(0),blocks.size(1),columns},blocks.options().dtype(::tyloi::kInt32));
    if (!output.numel()) return output;
    auto compact_blocks = blocks.contiguous(), compact_positions = positions.contiguous();
    raw::block_indices(compact_blocks.data_ptr(),blocks.scalar_type()==::tyloi::kInt64,compact_positions.data_ptr(),positions.scalar_type()==::tyloi::kInt64,output.data_ptr<int32_t>(),blocks.size(0)*blocks.size(1),int(blocks.size(1)),int(blocks.size(2)),int(pool),int(budget),int(columns),mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

DenseTensor mma_reduce_gated(const DenseTensor& partial,const DenseTensor& metadata,const DenseTensor& gate,
    bool half_output,int kv_heads,int total_blocks,int parts,int query_columns,int head_columns) {
    values({&partial,&metadata,&gate});
    TYLOINFER_RUNTIME_CHECK(partial.dim()==4 && partial.sizes()==gate.sizes() &&
        partial.scalar_type()==::tyloi::kFloat32 && metadata.scalar_type()==::tyloi::kFloat32 &&
        partial.is_contiguous() && metadata.is_contiguous() && half_or_float(gate) &&
        kv_heads>0 && partial.size(2)%kv_heads==0 && total_blocks>0 && parts>0 &&
        query_columns>0 && head_columns>0,"MMA gate reduction geometry mismatch");
    const int64_t heads=partial.size(2),tokens=partial.size(1),width=partial.size(3);
    const int64_t tiles=partial.size(0)*kv_heads*((heads/kv_heads+head_columns-1)/head_columns)*
        ((tokens+query_columns-1)/query_columns);
    TYLOINFER_RUNTIME_CHECK(tokens>0 && heads>0 && width>0 && tiles*parts==total_blocks &&
        (parts==1 || metadata.numel()>=int64_t(total_blocks)*query_columns*head_columns*(4+width)),
        "MMA gate reduction metadata mismatch");
    TYLOINFER_RUNTIME_CHECK(partial.numel()/width<=INT_MAX && width<=INT_MAX && tokens<=INT_MAX && heads<=INT_MAX,
        "MMA gate reduction exceeds CUDA geometry");
    const size_t shared=size_t(parts-1)*sizeof(float2);
    TYLOINFER_RUNTIME_CHECK(shared+sizeof(float)<=48*1024,"MMA gate reduction shared memory exceeded");
    MfqCudaGuard guard(partial.device());
    auto output=::tyloi::cuda::empty(partial.sizes(),partial.options().dtype(half_output?::tyloi::kFloat16: ::tyloi::kFloat32));
    const char* parallel=tyloi::cuda::env_lookup("TYLOINFER_QSA_PARALLEL_REDUCE");
    const bool parallel_reduce=!parallel || parallel[0]!='0';
    if(output.numel())raw::reduce_gated(partial.data_ptr<float>(),metadata.data_ptr(),gate.data_ptr(),output.data_ptr(),attention_layout(gate),output.numel()/width,int(tokens),int(heads),kv_heads,int(width),total_blocks,parts,query_columns,head_columns,gate.scalar_type()==::tyloi::kFloat32,half_output,sigmoid(gate.scalar_type()==::tyloi::kFloat32 && !gate.is_contiguous()),parallel_reduce,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

DenseTensor causal_gated(const DenseTensor& q, const DenseTensor& k, const DenseTensor& v,
    const DenseTensor& positions, const DenseTensor& gate, int64_t columns, double scale,
    bool half_output, bool fused_mma_gate, bool fused_mma_prepare, tyloi::cuda::HotKvView hot) {
    attention_shapes(q, k, v);
    values({&q, &gate});
    TYLOINFER_RUNTIME_CHECK(gate.dim() == 4 && gate.size(0) == q.size(0) && gate.size(1) == q.size(2) &&
        gate.size(2) == q.size(1) && gate.size(3) == q.size(3) && positions.defined() &&
        positions.is_cuda() && positions.device() == q.device() && positions.dim() == 1 &&
        positions.numel() == q.size(2) && (positions.scalar_type() == ::tyloi::kInt32 || positions.scalar_type() == ::tyloi::kInt64) &&
        columns > 0 && columns <= INT_MAX && std::isfinite(scale), "causal attention gate geometry mismatch");
    MfqCudaGuard guard(q.device());
    auto cache_positions = positions.contiguous();
    if (!half_or_float(q) || !half_or_float(gate) || (q.size(3) == 256 && !k.size(2))) {
        auto selected = ::tyloi::cuda::arange(columns, positions.options()).reshape({1, 1, columns})
            .expand({q.size(0), q.size(2), columns});
        selected = ::tyloi::cuda::where(selected <= cache_positions.reshape({1, q.size(2), 1}),
            selected, ::tyloi::cuda::full_like(selected, -1));
        return gated(sparse(q, k, v, selected, scale, false,hot), gate, half_output);
    }
    if (q.size(3) == 256) {
        auto key = k.to(::tyloi::kFloat16).contiguous(), value = v.to(::tyloi::kFloat16).contiguous();
        if(hot.mapping && q.numel()) {
            auto attended=run_mma(q,key,value,
                DenseTensor{},DenseTensor{},DenseTensor{},DenseTensor{},scale,"causal_gqa_attention_hot",
                fused_mma_gate?gate:DenseTensor{},half_output,cache_positions,columns,hot);
            return fused_mma_gate?attended:gated(attended,gate,half_output);
        }
        if (fused_mma_prepare && q.numel()) {
            auto attended = run_mma(
                q, key, value, DenseTensor{}, DenseTensor{}, DenseTensor{}, DenseTensor{}, scale,
                "causal_gqa_attention_gate", fused_mma_gate ? gate : DenseTensor{}, half_output,
                cache_positions, columns);
            return fused_mma_gate ? attended : gated(attended, gate, half_output);
        }
        // Prepare the original MMA query, safe indices and validity mask together.
        TYLOINFER_RUNTIME_CHECK(columns <= INT_MAX - 32, "causal MMA index geometry exceeds 32 bits");
        const int64_t padded_columns = ((columns + 31) / 32) * 32;
        auto query = ::tyloi::cuda::empty(q.sizes(), q.options().dtype(::tyloi::kFloat32));
        auto indices = ::tyloi::cuda::empty({q.size(0), q.size(2), padded_columns}, q.options().dtype(::tyloi::kInt32));
        auto mask = ::tyloi::cuda::empty(indices.sizes(), q.options().dtype(::tyloi::kFloat16));
        const int blocks = int(std::min<int64_t>(65535, (std::max(query.numel(), indices.numel()) + 255) / 256));
        if (!blocks) return ::tyloi::cuda::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
            q.options().dtype(half_output ? ::tyloi::kFloat16 : ::tyloi::kFloat32));
        const auto layout = AttentionLayout{q.stride(0), q.stride(2), q.stride(1), q.stride(3)};
        raw::prepare_causal(q.data_ptr(),q.scalar_type()==::tyloi::kFloat16,layout,query.data_ptr<float>(),query.numel(),q.size(1),q.size(2),q.size(3),cache_positions.data_ptr(),cache_positions.scalar_type()==::tyloi::kInt64,indices.data_ptr<int32_t>(),mask.data_ptr(),indices.numel(),columns,padded_columns,k.size(2),mfq_current_cuda_stream());
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
        auto attended = run_mma(
            query, key, value, indices, mask, DenseTensor{}, DenseTensor{}, scale, "qwen4_causal_gqa_attention_gate",
            fused_mma_gate?gate:DenseTensor{},half_output);
        if(fused_mma_gate)return attended;
        return gated(attended, gate, half_output);
    }
    auto query = q.contiguous();
    auto key = k.to(::tyloi::kFloat16).contiguous(), value = v.to(::tyloi::kFloat16).contiguous();
    auto output = ::tyloi::cuda::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
        q.options().dtype(half_output ? ::tyloi::kFloat16 : ::tyloi::kFloat32));
    if (!output.numel()) return output;
    auto scratch = q.size(3) > 1024 ? ::tyloi::cuda::empty(output.sizes(), q.options().dtype(::tyloi::kFloat32)) : DenseTensor{};
    const auto blocks = output.numel() / q.size(3);
    TYLOINFER_RUNTIME_CHECK(blocks <= INT_MAX, "causal attention exceeds CUDA grid size");
    const AttentionEpilogue epilogue{gate.data_ptr(), output.data_ptr(),
        gate.scalar_type() == ::tyloi::kFloat32, half_output,
        sigmoid(gate.scalar_type() == ::tyloi::kFloat32 && !gate.is_contiguous()), attention_layout(gate)};
    raw::sparse(query.data_ptr(),key.data_ptr(),value.data_ptr(),nullptr,scratch.defined()?scratch.data_ptr<float>():nullptr,q.size(0),q.size(1),k.size(1),q.size(2),q.size(3),k.size(2),columns,float(scale),q.scalar_type()==::tyloi::kFloat16,cache_positions.data_ptr(),cache_positions.scalar_type()==::tyloi::kInt64,epilogue,hot_view(hot),mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

DenseTensor dense(const DenseTensor& query, const DenseTensor& key, const DenseTensor& value,
                       int64_t offset, double scale, tyloi::cuda::HotKvView hot) {
    attention_shapes(query, key, value);
    const auto count = key.size(2);
    TYLOINFER_RUNTIME_CHECK(count > 0 && offset >= 0 && offset <= count &&
        query.size(2) <= count - offset, "attention query range is outside the cache");
    TYLOINFER_RUNTIME_CHECK(std::isfinite(scale), "attention scale must be finite");
    MfqCudaGuard guard(query.device());
    if(hot.mapping) {
        auto positions=::tyloi::cuda::arange(offset,offset+query.size(2),query.options().dtype(::tyloi::kInt64));
        if(query.size(3)==256)return run_mma(
            query,key,value,DenseTensor{},DenseTensor{},DenseTensor{},DenseTensor{},scale,"dense_gqa_attention_hot",
            DenseTensor{},false,positions,offset+query.size(2),hot);
        auto selected=::tyloi::cuda::arange(offset+query.size(2),positions.options()).reshape({1,1,-1}).expand({1,query.size(2),offset+query.size(2)});
        selected=::tyloi::cuda::where(selected<=positions.reshape({1,-1,1}),selected,::tyloi::cuda::full_like(selected,-1));
        return sparse(query,key,value,selected,scale,false,hot);
    }
    const auto options = query.options().dtype(::tyloi::kInt64);
    const auto kp = ::tyloi::cuda::arange(count, options).unsqueeze(0);
    const auto qp = ::tyloi::cuda::arange(query.size(2), options).unsqueeze(1) + offset;
    auto mask = (kp <= qp).unsqueeze(0).unsqueeze(0);
    return ::tyloi::cuda::scaled_dot_product_attention(query, key.to(query.scalar_type()),
        value.to(query.scalar_type()), mask, 0.0, false, scale, true).permute({0, 2, 1, 3});
}

} // namespace mfq_selected_attention
