#pragma once

#include "engine/models/qwen4_exp/block.h"
#include "engine/models/qwen4_exp/causal_lm.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/backends/cuda/executor/tq_moe.h"
#include "engine/backends/cuda/ops/projection/nint.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/weights/loader.h"
#include "engine/weights/group.h"
#include "engine/weights/encoding/quant/nint_rows.h"
#include "engine/weights/encoding/tq.h"
#include "engine/weights/tq/source.h"
#include "engine/weights/tq/rows.h"
#include "engine/weights/nint_row_pipeline.h"

namespace tyloi::cuda::qwen4_exp {
using Config = tyloi::models::qwen4_exp::Config;

struct BlockLoader : tyloi::weights::Loader {
    using GdnWeights = qwen4_exp::GdnWeights;
    using QsaWeights = qwen4_exp::QsaWeights;
    using PleWeights = qwen4_exp::PleWeights;
    CudaExecutionContext& execution;
    Linear* router_capture=nullptr;
    MoeFfnPrefetch* prefetch_capture=nullptr;
    bool* tq_graph_capture=nullptr;
    struct Embedding {
        attention_ops::Embedding function;
        std::shared_ptr<tyloi::NintRows> range;
        std::shared_ptr<tyloi::TqTable> tq;
        DenseTensor operator()(const DenseTensor& ids) const { return function(ids); }
    };

    auto residual_linear(const std::string &name) const {
        const auto part = name=="model.mhc.pre.down.weight" || name=="predictor.mhc.pre.down.weight"
            ? GatedResidual::Part::final_down
            : name=="model.mhc.pre.up.weight" || name=="predictor.mhc.pre.up.weight"
                ? GatedResidual::Part::final_up : GatedResidual::Part::generic;
#ifdef TYLOINFER_NATIVE_CUDA
        return GatedResidual::projection(weight(name), part);
#else
        return GatedResidual::projection(weight(name), execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)), execution.executor().profiler(),
                                          execution.executor().options().residual, part);
#endif
    }
#ifdef TYLOINFER_NATIVE_CUDA
    GatedResidual residual(std::shared_ptr<const ::tyloi::Tensor> norm, GatedResidual::Projection down,
#else
    GatedResidual residual(DenseTensor norm, GatedResidual::Projection down,
#endif
        GatedResidual::Projection up, GatedResidual::Projection injection, const Config &c) const {
        return {std::move(norm), std::move(down), std::move(up), std::move(injection),
#ifdef TYLOINFER_NATIVE_CUDA
                c.hidden, c.streams, c.eps};
#else
                c.hidden, c.streams, c.eps, execution.executor().options().residual, execution.executor().mode().ffn.batch_serial};
#endif
    }
    static auto final_mixer(GatedResidual weights) { return std::make_unique<GatedResidual>(std::move(weights)); }
    auto gdn_weights(const std::string &p) const {
        const bool independent = has(p + ".query.weight") || has(p + ".key.weight") || has(p + ".value.weight");
        TYLOINFER_RUNTIME_CHECK(!independent || !has(p + ".qkv.weight"), "ambiguous Qwen4 GDN input projections");
        GdnWeights w;
        if(independent)for(const auto* name:{"query","key","value"})
            w.input_projection.push_back(weight(p+"."+name+".weight"));
        else w.input_projection.push_back(weight(p+".qkv.weight"));
        w.input_projection.push_back(weight(p+".gate.weight"));
        w.input_projection.push_back(weight(p+".alpha.weight"));
        w.input_projection.push_back(weight(p+".beta.weight"));
        w.qkv_count=independent?3:1;
        w.output=weight(p+".output.weight");
#ifdef TYLOINFER_NATIVE_CUDA
        w.conv=dense(p+".conv.weight");
        w.dt_bias=dense(p+".dt_bias");
        w.a_log=dense(p+".a");
        w.norm=dense(p+".norm.weight");
#else
        w.conv=dynamic_cast<const ::tyloi::DenseTensor&>(*dense(p+".conv.weight"));
        w.dt_bias=dynamic_cast<const ::tyloi::DenseTensor&>(*dense(p+".dt_bias"));
        w.a_log=dynamic_cast<const ::tyloi::DenseTensor&>(*dense(p+".a"));
        w.norm=dynamic_cast<const ::tyloi::DenseTensor&>(*dense(p+".norm.weight"));
#endif
        return w;
    }
    auto qsa_weights(const std::string &p) const {
        QsaWeights w;
        for(const auto* name:{"query","key","value"})
            w.input_projection.push_back(weight(p+"."+name+".weight"));
        w.input_projection.push_back(weight(p+".indexer.query_key.weight"));
        w.output=weight(p+".output.weight");
#ifdef TYLOINFER_NATIVE_CUDA
        w.query_norm=dense(p+".query_norm.weight");
        w.key_norm=dense(p+".key_norm.weight");
        w.index_query_norm=dense(p+".indexer.query_norm.weight");
        w.index_key_norm=dense(p+".indexer.key_norm.weight");
#else
        w.query_norm=dynamic_cast<const ::tyloi::DenseTensor&>(*dense(p+".query_norm.weight"));
        w.key_norm=dynamic_cast<const ::tyloi::DenseTensor&>(*dense(p+".key_norm.weight"));
        w.index_query_norm=dynamic_cast<const ::tyloi::DenseTensor&>(*dense(p+".indexer.query_norm.weight"));
        w.index_key_norm=dynamic_cast<const ::tyloi::DenseTensor&>(*dense(p+".indexer.key_norm.weight"));
#endif
        if(has(p+".gate.weight")) {
            w.input_projection.push_back(weight(p+".gate.weight"));
            w.has_gate=true;
        }
        return w;
    }
    auto gdn(GdnWeights weights, const Config &c) const {
        weights.input_projection=pack_projections(std::move(weights.input_projection),
            options.grouped,placement.cpu);
        auto shared=std::make_shared<const GdnWeights>(std::move(weights));
        return std::make_unique<Gdn>(std::move(shared), c.key_heads, c.value_heads, c.linear_width,
                                     c.kernel, c.eps, c.silu_gate,true,execution.executor().mode().transposed,
                                     execution.executor().options().gdn_fused_output,true,execution.executor().options().gdn_fused_preparation,
                                     execution.executor().options().gdn_fused_core);
    }
    auto qsa(QsaWeights weights, const Config &c) const {
        weights.input_projection=pack_projections(std::move(weights.input_projection),
            options.grouped,placement.cpu);
        QsaConfig geometry{c.heads, c.kv_heads, c.width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.eps};
#ifdef TYLOINFER_NATIVE_CUDA
        constexpr bool default_rotary_fusion = true;
#else
        constexpr bool default_rotary_fusion = false;
#endif
        auto rotary = std::make_shared<RotaryEmbedding>(c.rotary, c.maximum, c.rope_base,
            c.sections, c.interleaved,
            ::tyloi::cuda::options::rotary_fusion().value_or(default_rotary_fusion), execution.config.yarn);
        for(auto* norm:{&weights.query_norm,&weights.key_norm,&weights.index_query_norm,&weights.index_key_norm})
#ifdef TYLOINFER_NATIVE_CUDA
            if((*norm)->defined())*norm=std::make_shared<DenseTensor>(
                dynamic_cast<const DenseTensor&>(**norm).to(::tyloi::kFloat32).contiguous());
#else
            if(norm->defined())*norm=norm->to(::tyloi::kFloat32).contiguous();
#endif
        auto shared=std::make_shared<const QsaWeights>(std::move(weights));
        return std::make_unique<Qsa>(std::move(shared), geometry, std::move(rotary), true, true, execution.offload, execution.kv);
    }
    auto moe(std::vector<Linear> gate_up, Linear down, Linear router,
                    Linear shared_gate, Linear sg, Linear su, Linear sd, const Config &c) const {
        if(router_capture)*router_capture=router;
        if(tq_graph_capture) {
            const auto banks = tq_banks(gate_up.empty() ? nullptr : gate_up.front(),
                gate_up.empty() ? nullptr : gate_up.back(), down);
            *tq_graph_capture = gate_up.size()==2 && down &&
                banks.gate && banks.up && banks.down;
        }
        return Moe(execution.executor().pipelines(), std::move(gate_up), std::move(down), std::move(router), std::move(shared_gate),
                        std::move(sg), std::move(su), std::move(sd), c, prefetch_capture);
    }
    auto embedding(const std::string &name) const {
        if(tyloi::is_tq_table(source,name)) {
            auto table=tyloi::tq_table(source,name);
            Embedding lookup{[table](const DenseTensor& ids){
                auto result=lookup_tq_table(*table,ids);
                return dynamic_cast<const DenseTensor&>(*result);
            },{},table};
            return std::pair{std::move(lookup),std::array<int64_t,2>{static_cast<int64_t>(table->rows()),table->width()}};
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (require_tensor(source, name).dtype == "NINT") {
            auto table = load_nint_row_table(source, name);
            Embedding lookup{[table](const DenseTensor &ids) {
                return nint_row_embedding_lookup(*table, ids);
            },table};
            return std::pair{std::move(lookup), std::array<int64_t, 2>{table->rows(), table->width()}};
        }
#endif

        auto weight = this->weight(name);
        Embedding lookup{[weight](const DenseTensor &ids) { return quant_embedding_lookup(*weight, ids); },{}};
        return std::pair{std::move(lookup), std::array<int64_t, 2>{weight->size(0), weight->size(1)}};
    }
    std::vector<int64_t> integers(const std::string &name) const {
        const auto &type = require_tensor(source, name).dtype;
        TYLOINFER_RUNTIME_CHECK(type == "I64" || type == "I32",
                          "Qwen4 PLE metadata must be a dense integer array: ", name);
        auto host = dynamic_cast<const ::tyloi::DenseTensor&>(*tensor(name)).to(::tyloi::kInt64).contiguous().cpu();
        TYLOINFER_RUNTIME_CHECK(host.dim() == 1, "Qwen4 PLE metadata must be a vector");
        return {host.data_ptr<int64_t>(), host.data_ptr<int64_t>() + host.numel()};
    }
    static auto ple(std::vector<Embedding> shards, int64_t rows, int64_t width, const Config &c,
                    std::vector<int64_t> multipliers, std::vector<int64_t> offsets,
                    std::vector<int64_t> sizes, PleWeights weights) {
        std::vector<attention_ops::Embedding> functions;std::vector<std::shared_ptr<tyloi::NintRows>> ranges;
        std::vector<std::shared_ptr<tyloi::TqTable>> tq;
        for(auto& shard:shards){functions.push_back(std::move(shard.function));ranges.push_back(std::move(shard.range));tq.push_back(std::move(shard.tq));}
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(!std::all_of(ranges.begin(),ranges.end(),[](const auto& p){return bool(p);})) ranges.clear();
#else
        ranges.clear();
#endif
        NgramEmbedding::RowLookup row_lookup;
        NgramEmbedding::StageFactory stage_factory;
        if(std::all_of(tq.begin(),tq.end(),[](const auto& p){return bool(p);})) {
            row_lookup=[tq,rows](const auto& ids,const auto& device) {
                return lookup_tq_shards(tq,rows,ids,device);
            };
            stage_factory=[tq=std::move(tq),rows](const auto& shape,const auto& device) {
                return make_tq_row_stage(tq,rows,shape,device);
            };
        }
        NgramEmbedding embedding(std::move(functions), rows, width, c.ngram, c.ngram_heads, c.eos,
                                 std::move(multipliers), std::move(offsets), std::move(sizes),std::move(ranges),std::move(row_lookup),std::move(stage_factory));
        return std::make_unique<Ple>(std::move(embedding), std::move(weights), c.hidden, c.streams,
                                     c.ngram, c.eps);
    }
    std::unique_ptr<Qwen4Block> block(const Config &c, int layer, bool predictor);
};


} // namespace tyloi::cuda::qwen4_exp
