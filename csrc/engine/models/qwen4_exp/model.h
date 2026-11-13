#include "env.h"
#include "engine/backends/cuda/executor/options.h"
#pragma once
#include "resident_allocations.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/models/qwen4_exp/causal_lm.h"
#include "engine/models/qwen4_exp/ngram.h"
#include "engine/models/common/attention_ops.h"
#include "engine/ops.h"
#include "engine/ops/gdn.h"
#include "engine/ops/qsa.h"
#include "engine/ops/ple.h"
#include "engine/backends/cuda/kernels/qwen4_exp.h"
#include "engine/backends/cuda/kernels/ple_fused.h"
#include "engine/weights/row_stage.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>

namespace tyloi::cuda::qwen4_exp {
using attention_ops::DenseTensor;
using attention_ops::Linear;
using attention_ops::rms_norm;

// The upstream PLE hash is a CPU uint64 algorithm, including signed remainder
// after bitwise mixing and EOS segment boundaries. Keep shards independent.
class NgramEmbedding {
  public:
    using Context = tyloi::models::qwen4_exp::NgramContext;
    struct State {
        State() = default;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        State(State&&) = default;
        State& operator=(State&&) = default;
        Context context;
        std::shared_ptr<tyloi::weights::RowPipeline> pipeline;
        bool hash_ready=false,hash_reported=false;
        int cached_tokens=0;
        std::array<int64_t,8> cached_ids{};
        tyloi::models::qwen4_exp::NgramHashes cached_hashes;
    };
    /** Own the row shards and validate their n-gram metadata. */
    NgramEmbedding(std::vector<::tyloi::Tensor> shards, int64_t rows, int64_t dimension, int64_t ngram,
                   int64_t heads_per_ngram, int64_t eos, std::vector<int64_t> multipliers,
                   std::vector<int64_t> offsets, std::vector<int64_t> vocab)
        : shards_(std::move(shards)), rows_(rows), dimension_(dimension), ngram_(ngram),
          heads_(heads_per_ngram), eos_(eos), multipliers_(std::move(multipliers)),
          offsets_(std::move(offsets)), vocab_(std::move(vocab)) {
        TYLOINFER_RUNTIME_CHECK(!shards_.empty() && rows_ > 0 && dimension_ > 0 && ngram_ > 1 &&
                              heads_ > 0 && multipliers_.size() == size_t(ngram_) &&
                              offsets_.size() == size_t((ngram_ - 1) * heads_) &&
                              vocab_.size() == offsets_.size(),
                          "invalid Qwen4 ngram metadata geometry");
        for (size_t h = 0; h < vocab_.size(); ++h)
            TYLOINFER_RUNTIME_CHECK(vocab_[h] > 0 && offsets_[h] >= 0 &&
                                  offsets_[h] <= int64_t(shards_.size()) * rows_ - vocab_[h],
                              "Qwen4 ngram vocabulary lies outside embedding shards");
        source_ = tyloi::weights::RowPipeline::source(shards_,rows_);
    }
    void reset(State& state) const { state.context = {};state.hash_ready=false; }
    /** Report whether all shards support stable graph row staging. */
    bool graph_supported() const {
        return bool(source_);
    }
    const Context &context(const State& state) const { return state.context; }
    void restore(State& state,Context context) const { state.context = std::move(context);state.hash_ready=false; }
    Context prefix_context(const State& state,const DenseTensor& host_ids,int64_t count) const {
        auto host=host_ids.narrow(1,0,count).to(::tyloi::kInt64).contiguous();
        return tyloi::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),host.size(0),count,
            ngram_,heads_,eos_,multipliers_,offsets_,vocab_,state.context,true).next;
    }
    /** Prefetch the rows selected by host token IDs without advancing the context. */
    void prefetch(State& state,const DenseTensor& host_ids,bool early_prefix=false) const {
        if (!graph_supported()) return;
        if (!state.pipeline) state.pipeline=std::make_shared<tyloi::weights::RowPipeline>(source_);
        TYLOINFER_RUNTIME_CHECK(host_ids.is_cpu() && host_ids.dim()==2,"PLE prefetch requires host token IDs");
        auto host=host_ids.to(::tyloi::kInt64).contiguous();const auto b=host.size(0),t=host.size(1);
        if(cacheable_hashes(b,t)) {
            const auto& hashes=graph_hashes(state,host);
            state.pipeline->prefetch(hashes.ids,{b,t,(ngram_-1)*heads_},early_prefix);
            return;
        }
        auto hashes=tyloi::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),b,t,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,state.context,true);
        state.pipeline->prefetch(hashes.ids,{b,t,(ngram_-1)*heads_},early_prefix);
}
    /** Prepare stable graph storage for the selected row geometry. */
    std::shared_ptr<tyloi::weights::RowStage> graph_stage(State& state,int64_t b,int64_t t,const ::tyloi::Device& device) const {
        const std::vector<int64_t> shape{b,t,(ngram_-1)*heads_};
        TYLOINFER_RUNTIME_CHECK(graph_supported(),"PLE graph requires prepared range rows");
        if (!state.pipeline) state.pipeline=std::make_shared<tyloi::weights::RowPipeline>(source_);
        return state.pipeline->stage(shape,device);
    }
    /** Compute selected graph rows and their next context without publishing it. */
    tyloi::models::qwen4_exp::NgramHashes graph_rows(State& state,const DenseTensor& host_ids) const {
        TYLOINFER_RUNTIME_CHECK(graph_supported() && host_ids.is_cpu() && host_ids.dim()==2,"PLE graph requires host token IDs");
        auto host=host_ids.to(::tyloi::kInt64).contiguous();const auto b=host.size(0),t=host.size(1);
        if(cacheable_hashes(b,t)) return graph_hashes(state,host);
        return tyloi::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),b,t,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,state.context,true);
    }
    void upload_graph(State& state,const DenseTensor& host_ids,tyloi::weights::RowStage& stage) const {
        auto hashes=graph_rows(state,host_ids);
        const auto b=host_ids.size(0),t=host_ids.size(1);
        stage.upload(hashes.ids,{b,t,(ngram_-1)*heads_});
        restore(state,std::move(hashes.next));
    }
    /** Gather hashed embedding rows and optionally advance the request context. */
    DenseTensor forward(State& state,const DenseTensor &ids, bool cache, DenseTensor *hashed_ids = nullptr,const DenseTensor& host_ids = {}) const {
        TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0,
                          "Qwen4 ngram IDs must be [B,T]");
        const auto b = ids.size(0), t = ids.size(1), nh = (ngram_ - 1) * heads_;
        auto host = host_ids.defined() ? host_ids.to(::tyloi::kInt64).contiguous() : ids.to(::tyloi::kInt64).contiguous().cpu();
        TYLOINFER_RUNTIME_CHECK(host.is_cpu() && host.sizes()==ids.sizes(),"PLE host token mirror differs");
        const auto *source = host.data_ptr<int64_t>();
        auto hashes = tyloi::models::qwen4_exp::ngram_hashes(
            source, b, t, ngram_, heads_, eos_, multipliers_, offsets_, vocab_, state.context, cache);
        auto &global = hashes.ids;
        if (graph_supported()) {
            if (!state.pipeline) state.pipeline=std::make_shared<tyloi::weights::RowPipeline>(source_);
            auto result=state.pipeline->collect(global,{b,t,nh},ids.device());
            if(hashed_ids)*hashed_ids=::tyloi::cuda::tensor(global).reshape({b,t,nh}).to(ids.device());
            if(cache)restore(state,std::move(hashes.next));
            return result.dense().reshape({b,t,nh*dimension_});
        }

        auto global_tensor =
            ::tyloi::cuda::from_blob(global.data(), {b, t, nh}, ::tyloi::TensorOptions().dtype(::tyloi::kInt64))
                .clone()
                .to(ids.device());
        if (hashed_ids)
            *hashed_ids = global_tensor;
        std::set<int64_t> used;
        for (auto value : global)
            used.insert(value / rows_);
        DenseTensor result;
        if(global_tensor.is_cuda()) {
            std::vector<int64_t> selected(used.begin(),used.end());
            result=tyloi::ops::sharded_embedding(shards_,global_tensor,selected,rows_).dense();
        } else
        {
        for (auto shard : used) {
            auto mask = (global_tensor >= shard * rows_) & (global_tensor < (shard + 1) * rows_);
            auto local =
                ::tyloi::cuda::where(mask, global_tensor - shard * rows_, ::tyloi::cuda::zeros_like(global_tensor));
            auto values = tyloi::ops::embedding(local,shards_.at(shard)).dense();
            TYLOINFER_RUNTIME_CHECK(values.sizes().vec() == std::vector<int64_t>({b, t, nh, dimension_}),
                              "Qwen4 ngram embedding shape mismatch");
            values = values * mask.unsqueeze(-1).to(values.scalar_type());
            result = result.defined() ? result + values : values;
        }
        }
        if (cache)
            restore(state,std::move(hashes.next));
        return result.reshape({b, t, nh * dimension_});
    }

  private:

    std::vector<::tyloi::Tensor> shards_;
    std::shared_ptr<const tyloi::weights::RowPipeline::Source> source_;
    int64_t rows_, dimension_, ngram_, heads_, eos_;
    std::vector<int64_t> multipliers_, offsets_, vocab_;
    bool reuse_hashes_=[] {
        const auto* option=tyloi::cuda::env_lookup("TYLOINFER_PLE_HASH_REUSE");return !option || option[0]!='0';
    }();
    bool cacheable_hashes(int64_t b,int64_t t) const {return reuse_hashes_ && b==1 && t>0 && t<=8;}
    tyloi::models::qwen4_exp::NgramHashes& graph_hashes(State& state,const DenseTensor& host) const {
        const auto tokens=host.size(1);const auto* ids=host.data_ptr<int64_t>();
        if(state.hash_ready && state.cached_tokens==tokens && std::equal(ids,ids+tokens,state.cached_ids.begin())) {
            if(!state.hash_reported) {
                state.hash_reported=true;
                std::fprintf(stderr,"ple_ngram_hash_reuse enabled=1 tokens=%lld\n",static_cast<long long>(tokens));
            }
            return state.cached_hashes;
        }
        state.cached_hashes=tyloi::models::qwen4_exp::ngram_hashes(ids,1,tokens,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,state.context,true);
        std::copy_n(ids,tokens,state.cached_ids.begin());state.cached_tokens=int(tokens);state.hash_ready=true;
        return state.cached_hashes;
    }
};

/** Coordinate n-gram context and row loading with the PLE numeric operation. */
class Ple {
public:
    /** Own numeric convolution state and the request's n-gram history. */
    struct State {
        NgramEmbedding::State embedding;
        tyloi::ops::Ple::State numeric;
        tyloi::Tensor input{DenseTensor{}}, rows{DenseTensor{}};
        NgramEmbedding::Context saved_context;
        std::vector<NgramEmbedding::Context> saved_prefix_context;
    };
    /** Bind immutable numeric weights to the n-gram row source. */
    Ple(NgramEmbedding embedding, tyloi::ops::PleWeights weights, int64_t hidden, int64_t streams,
        int64_t dilation, double eps, tyloi::Device target)
        : embedding_(std::move(embedding)), numeric_(std::move(weights),hidden,streams,dilation,eps,target),
          hidden_(hidden),streams_(streams) {}
    /** Reset the request history, optionally retaining graph storage. */
    void reset(State& state,bool retain=false) const {
        embedding_.reset(state.embedding); numeric_.reset(state.numeric,retain);
        state.saved_context={}; state.saved_prefix_context.clear();
        state.input.dense()={}; state.rows.dense()={};
    }
    /** Commit and release speculative history. */
    void commit(State& state) const {
        numeric_.commit(state.numeric); state.saved_context={}; state.saved_prefix_context.clear();
    }
    /** Save convolution and n-gram history for rollback. */
    void checkpoint(State& state) const {
        state.saved_context=embedding_.context(state.embedding); numeric_.checkpoint(state.numeric);
    }
    /** Restore the saved history or a selected retained prefix. */
    void rollback(State& state,int64_t keep=-1) const {
        auto context=state.saved_context;
        if(keep>=0 && state.numeric.saved_prefix_position>=0) {
            const auto index=keep-state.numeric.saved_prefix_position;
            TYLOINFER_RUNTIME_CHECK(index>=0 && index<int64_t(state.saved_prefix_context.size()),
                "Qwen4 PLE prefix checkpoint is out of range");
            context=state.saved_prefix_context[index];
        }
        numeric_.rollback(state.numeric,keep); embedding_.restore(state.embedding,std::move(context));
        state.saved_context={}; state.saved_prefix_context.clear();
    }
    /** Borrow the current convolution history. */
    const tyloi::Tensor& conv_state(const State& state) const { return state.numeric.conv; }
    /** Borrow stable numeric slots used by graph capture. */
    std::vector<tyloi::Tensor*> graph_state(State& state) const { return numeric_.graph_state(state.numeric); }
    /** Keep numeric and pending embedding backings alive through execution. */
    std::vector<tyloi::Tensor> storage(const State& state) const {
        auto result=numeric_.storage(state.numeric);
        if(state.input.defined())result.emplace_back(state.input.dense());
        if(state.rows.defined())result.emplace_back(state.rows.dense());
        return result;
    }
    /** Estimate missing convolution storage for a batch. */
    std::size_t growth_bytes(const State& state,int64_t batch) const { return numeric_.growth_bytes(state.numeric,batch); }
    /** Report whether the row source supports graph execution. */
    bool graph_supported() const { return embedding_.graph_supported(); }
    /** Borrow the committed n-gram context for a session snapshot. */
    const NgramEmbedding::Context& session_context(const State& state) const { return embedding_.context(state.embedding); }
    /** Publish a prepared n-gram context after graph replay. */
    void restore_graph_context(State& state,NgramEmbedding::Context context) const { embedding_.restore(state.embedding,std::move(context)); }
    /** Restore convolution and n-gram history from a session. */
    void restore_session(State& state,const DenseTensor& conv,NgramEmbedding::Context context,bool retain=false) const {
        numeric_.restore_session(state.numeric,conv,retain); embedding_.restore(state.embedding,std::move(context));
        state.saved_context={}; state.saved_prefix_context.clear();
    }
    /** Create a request-owned row transfer stage for graph execution. */
    std::shared_ptr<tyloi::weights::RowStage> graph_stage(State& state,int64_t tokens,const tyloi::Device& device) const {
        return embedding_.graph_stage(state.embedding,1,tokens,device);
    }
    /** Upload graph rows and update the n-gram context. */
    void upload_graph(State& state,const DenseTensor& ids,tyloi::weights::RowStage& stage) const { embedding_.upload_graph(state.embedding,ids,stage); }
    /** Compute graph row hashes without publishing the next context. */
    tyloi::models::qwen4_exp::NgramHashes graph_rows(State& state,const DenseTensor& ids) const { return embedding_.graph_rows(state.embedding,ids); }
    /** Compute the context corresponding to a graph verification prefix. */
    NgramEmbedding::Context graph_prefix_context(const State& state,const DenseTensor& ids,int64_t count) const {
        return embedding_.prefix_context(state.embedding,ids,count);
    }
    /** Bind numeric and n-gram snapshots for a graph verification window. */
    void bind_graph_checkpoint(State& state,const std::vector<tyloi::Tensor>& conv,
            std::vector<NgramEmbedding::Context> context,int64_t position) const {
        TYLOINFER_RUNTIME_CHECK(!conv.empty() && conv.size()==context.size(),"invalid PLE prefix checkpoints");
        numeric_.bind_graph_checkpoint(state.numeric,conv,position);
        state.saved_context=context.front(); state.saved_prefix_context=std::move(context);
    }
    /** Execute graph PLE with prepared rows and optional prefix checkpoints. */
    DenseTensor forward_graph(State& state,const DenseTensor& x,const DenseTensor&,const DenseTensor& rows,
            int64_t confirmed,std::vector<tyloi::Tensor>& conv) const {
        return numeric_.forward_graph(state.numeric,x,rows,confirmed,conv).dense();
    }
    /** Start request row reads for the supplied token IDs. */
    void prefetch_tokens(State& state,const DenseTensor& ids,bool early=false) const { embedding_.prefetch(state.embedding,ids,early); }
    /** Load embedding rows and execute one chunk with transactional context. */
    DenseTensor forward_chunk(State& state,const DenseTensor& x,const DenseTensor& ids,bool cache,
            const DenseTensor& host={},const DenseTensor& rows={}) const {
        auto context=embedding_.context(state.embedding);
        try {
            state.input.dense()=x;
            state.rows.dense()=rows.defined()?rows:embedding_.forward(state.embedding,ids,cache,nullptr,host);
            auto result=numeric_.forward_chunk(state.numeric,state.input,state.rows,cache).dense();
            state.input.dense()={}; state.rows.dense()={};
            return result;
        } catch(...) { embedding_.restore(state.embedding,std::move(context)); throw; }
    }
    /** Execute a request window and retain speculative prefix history when needed. */
    DenseTensor forward(State& state,const DenseTensor& x,const DenseTensor& ids,bool cache,
            int64_t confirmed=0,const DenseTensor& host={},int64_t position=-1) const {
        TYLOINFER_RUNTIME_CHECK(x.dim()==3 && ids.dim()==2 && x.size(0)==ids.size(0) &&
            x.size(1)==ids.size(1) && x.size(1)>0 && x.size(2)==hidden_*streams_ && confirmed>=0 &&
            confirmed<=x.size(1) && (!confirmed || cache),"invalid Qwen4 PLE input");
        if(cache && position>=0 && confirmed>0 && confirmed<x.size(1) && x.size(1)<=8) {
            TYLOINFER_RUNTIME_CHECK(!state.numeric.saved && state.numeric.conv.defined(),"PLE verification requires committed convolution state");
            auto host_ids=host.defined()?host:ids.to(tyloi::kCPU,tyloi::kInt64);
            state.numeric.saved_prefix_conv.clear(); state.saved_prefix_context.clear();
            for(int64_t i=confirmed;i<x.size(1);++i) {
                state.numeric.saved_prefix_conv.emplace_back(tyloi::cuda::empty_like(state.numeric.conv.dense()));
                state.saved_prefix_context.push_back(embedding_.prefix_context(state.embedding,host_ids,i));
            }
            auto context=embedding_.context(state.embedding);
            try {
                state.input.dense()=x;
                state.rows.dense()=embedding_.forward(state.embedding,ids,true,nullptr,host_ids);
                auto result=numeric_.forward_graph(state.numeric,state.input,state.rows,confirmed,state.numeric.saved_prefix_conv).dense();
                auto contexts=state.saved_prefix_context;
                bind_graph_checkpoint(state,state.numeric.saved_prefix_conv,std::move(contexts),position+confirmed);
                state.input.dense()={}; state.rows.dense()={};
                return result;
            } catch(...) { embedding_.restore(state.embedding,std::move(context)); throw; }
        }
        if(!confirmed || confirmed==x.size(1)) return forward_chunk(state,x,ids,cache,host);
        TYLOINFER_RUNTIME_CHECK(!state.numeric.saved,"resolve previous PLE checkpoint");
        auto first=forward_chunk(state,x.narrow(1,0,confirmed),ids.narrow(1,0,confirmed),cache,
            host.defined()?host.narrow(1,0,confirmed):DenseTensor{});
        checkpoint(state);
        try {
            auto second=forward_chunk(state,x.narrow(1,confirmed,x.size(1)-confirmed),ids.narrow(1,confirmed,x.size(1)-confirmed),cache,
                host.defined()?host.narrow(1,confirmed,x.size(1)-confirmed):DenseTensor{});
            return tyloi::cuda::cat({first,second},1);
        } catch(...) { rollback(state); throw; }
    }
private:
    NgramEmbedding embedding_;
    tyloi::ops::Ple numeric_;
    int64_t hidden_,streams_;
};

} // namespace tyloi::cuda::qwen4_exp
