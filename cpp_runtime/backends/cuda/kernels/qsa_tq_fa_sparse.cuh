#pragma once
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

// Sort only selected block IDs; each query then writes the masks used by this union.
template<int Items>
static __global__ void prepare_sparse_selected_union(half* mask_ptr,const int* inverse,
        const tyloinfer::qsa::SelectedBlock* selected,const int64_t* positions,int count,
        int* indices,int* counts,int rows,int queries,int index_rows) {
    static_assert(Items>0 && Items<=32);
    constexpr int threads=256;
    using Sort=cub::BlockRadixSort<int,threads,Items>;
    using Scan=cub::BlockScan<int,threads>;
    __shared__ union Storage {
        typename Sort::TempStorage sorting;
        typename Scan::TempStorage scanning;
        int sorted[threads*Items];
    } work;
    const int tid=threadIdx.x,group=blockIdx.x,blocks=rows/4;
    int ids[Items];
#pragma unroll
    for(int j=0;j<Items;++j) {
        const int at=tid*Items+j,token=group*8+at/count,slot=at%count;
        int id=INT_MAX;
        if(at<8*count && token<queries) {
            const auto block=selected[int64_t(token)*count+slot];
            const auto left=positions[token]-int64_t(block.id)*4+1;
            const auto visible=left<=0?0u:left>=4?15u:(1u<<int(left))-1u;
            if(block.id>=0 && block.id<blocks && (block.token_mask&visible))id=block.id;
        }
        ids[j]=id;
    }
    Sort(work.sorting).Sort(ids);
    __syncthreads();
#pragma unroll
    for(int j=0;j<Items;++j)work.sorted[tid*Items+j]=ids[j];
    __syncthreads();
    unsigned unique=0;int local_count=0;
#pragma unroll
    for(int j=0;j<Items;++j) {
        const int at=tid*Items+j;
        const bool first=ids[j]!=INT_MAX && (!at || ids[j]!=work.sorted[at-1]);
        unique|=unsigned(first)<<j;local_count+=first;
    }
    __syncthreads();
    int offset=0,total=0;
    Scan(work.scanning).ExclusiveSum(local_count,offset,total);
    int rank=offset;
    auto* mask=reinterpret_cast<uint8_t*>(mask_ptr);
#pragma unroll
    for(int j=0;j<Items;++j)if(unique&(1u<<j)) {
        const int id=ids[j];
        for(int t=0;t<8;++t) {
            const int token=group*8+t;
            if(token>=queries)break;
            const int64_t at=int64_t(token)*blocks+id;
            const int slot=inverse[at];
            unsigned bits=0;
            if(slot>=0 && slot<count) {
                const auto block=selected[int64_t(token)*count+slot];
                const auto left=positions[token]-int64_t(id)*4+1;
                const auto visible=left<=0?0u:left>=4?15u:(1u<<int(left))-1u;
                bits=block.token_mask&visible;
            }
            mask[int64_t(token)*blocks+id]=uint8_t(bits);
        }
        for(int r=0;r<4;++r)indices[int64_t(group)*index_rows+4*rank+r]=4*id+r;
        ++rank;
    }
    const int live=4*total;
    for(int r=live+tid;r<min(index_rows,(live+255)&~255);r+=threads)
        indices[int64_t(group)*index_rows+r]=-1;
    if(!tid)counts[group]=live;
}


// Split selected KV batches while retaining disjoint microblock history writes.
static __global__ void qsa_fa_split(tyloinfer::qsa::TqPrefillAttention a,const half* mask,
        const int* indices,int index_rows,float* partial,float* norms,int parts,uint3 query_shape,QsaFaHistory history) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && defined(TURING_MMA_AVAILABLE) && defined(FLASH_ATTN_AVAILABLE)
    constexpr int width=256,columns=64;
    constexpr int warps=ggml_cuda_fattn_mma_get_nthreads(width,width,columns)/32;
    constexpr int batch=ggml_cuda_fattn_mma_get_nbatch_fa(width,width,columns);
    const int groups=(a.queries+7)/8,gqa=a.heads/a.kv_heads,head_groups=(gqa+7)/8;
    const int group=blockIdx.x%groups,head_group=(blockIdx.x/groups)%head_groups;
    const int kv_head=blockIdx.x/(groups*head_groups),head=kv_head*gqa+head_group*8;
    const int part=blockIdx.y,count=indices[int64_t(groups)*index_rows+group];
    const int batches=(count+batch-1)/batch;
    const int begin=int64_t(part)*batches/parts,end=int64_t(part+1)*batches/parts;
    const int64_t rows=int64_t(a.queries)*a.heads;
    float* output=partial+int64_t(part)*rows*width;
    history.norm=norms+int64_t(part)*rows;
    if(begin==end) {
        for(int i=threadIdx.y*32+threadIdx.x;i<8*8*width;i+=warps*32) {
            const int q=group*8+i/(8*width),h=head+(i/width)%8;
            if(q<a.queries && h<kv_head*gqa+gqa)output[(int64_t(q)*a.heads+h)*width+i%width]=0.f;
        }
        for(int i=threadIdx.y*32+threadIdx.x;i<64;i+=warps*32) {
            const int q=group*8+i/8,h=head+i%8;
            if(q<a.queries && h<kv_head*gqa+gqa)history.norm[int64_t(q)*a.heads+h]=-INFINITY;
        }
        return;
    }
    const int stride=width/2;
    const auto* k=reinterpret_cast<const half2*>(a.keys)+kv_head*4*width/2;
    const auto* v=reinterpret_cast<const half2*>(a.values)+kv_head*4*width/2;
    const auto* q=reinterpret_cast<const float2*>(a.rotated)+head*width/2;
    auto* result=reinterpret_cast<float2*>(output)+head*width/2;
    flash_attn_ext_f16_process_tile<256,256,8,8,warps,false,false,true,false,false>(
        q,k,v,mask,indices+int64_t(group)*index_rows,nullptr,result,nullptr,
        1.f/sqrtf(float(a.width)),1.f,0.f,query_shape,a.heads,gqa,
        min(count,end*batch),a.heads*width/2,width/2,stride,stride,a.kv_rows,
        group,head_group,begin,end,history,head);
#endif
}

// Combine normalized partial outputs using their log-normalizers.
static __global__ void qsa_fa_merge(float* output,float* norm,const float* partial,const float* norms,
        int rows,int parts) {
    const int row=blockIdx.x,d=threadIdx.x;
    float maximum=-INFINITY;
    for(int p=0;p<parts;++p)maximum=fmaxf(maximum,norms[int64_t(p)*rows+row]);
    float sum=0,value=0;
    for(int p=0;p<parts;++p) {
        const float logsum=norms[int64_t(p)*rows+row];
        const float scale=isfinite(logsum)?__expf(logsum-maximum):0.f;
        sum+=scale;value+=scale*partial[(int64_t(p)*rows+row)*256+d];
    }
    output[int64_t(row)*256+d]=sum>0?value/sum:0.f;
    if(!d && norm)norm[row]=sum>0?maximum+__logf(sum):-INFINITY;
}
