#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/qwen4_exp/layer.h"
#include "engine/models/qwen4_exp/causal_lm.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/backends/cuda/ops/projection/nint.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/weights/loader.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/weights/row_source.h"
#include "engine/weights/encoding/quant/nint_rows.h"
#include "engine/weights/encoding/tq.h"
#include "engine/weights/tq/source.h"
#include "engine/weights/tq/rows.h"
#include "engine/weights/nint_row_pipeline.h"
#endif
#include "engine/weights/group.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "env.h"

#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/execution.h"
#endif
#include "engine/backends/cuda/native/scope.h"
#include "engine/models/common/check.h"
#include "engine/models/common/load.h"
#include "engine/models/load.h"
#include "engine/models/source/model_config.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/source.h"
#include "engine/weights/cache.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace tyloi::cuda::qwen4_exp {

using Config = tyloi::models::qwen4_exp::Config;

struct LayerLoader : tyloi::weights::Loader {
#ifndef TYLOINFER_NATIVE_CUDA
    using GdnWeights = qwen4_exp::GdnWeights;
#endif
#ifndef TYLOINFER_NATIVE_CUDA
    using QsaWeights = qwen4_exp::QsaWeights;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    using PleWeights = tyloi::ops::PleWeights;
    tyloi::ops::YarnScaling yarn;
    using Embedding = ::tyloi::Tensor;
#else
    using PleWeights = qwen4_exp::PleWeights;
    CudaExecutionContext& execution;
    struct Embedding {
        attention_ops::Embedding function;
        std::shared_ptr<tyloi::NintRows> range;
        std::shared_ptr<tyloi::TqTable> tq;
        DenseTensor operator()(const DenseTensor& ids) const { return function(ids); }
    };
#endif

    auto residual_linear(const std::string &name) const {
#ifdef TYLOINFER_NATIVE_CUDA
        return weight(name);
#else
        const auto part = name=="model.mhc.pre.down.weight" || name=="predictor.mhc.pre.down.weight"
            ? GatedResidual::Part::final_down
            : name=="model.mhc.pre.up.weight" || name=="predictor.mhc.pre.up.weight"
                ? GatedResidual::Part::final_up : GatedResidual::Part::generic;
        return GatedResidual::projection(weight(name), execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)), execution.executor().profiler(),
                                          execution.executor().options().residual, part);
#endif
    }
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::Residual residual(::tyloi::Tensor norm, ::tyloi::Tensor down,
        ::tyloi::Tensor up, ::tyloi::Tensor injection, const Config &c) const {
        const bool final=!injection.defined();
        return {tyloi::ops::ResidualWeights{std::move(norm),std::move(down),std::move(up),std::move(injection)},
                tyloi::ops::ResidualConfig{c.hidden,c.streams,c.eps,final}};
    }
    static auto final_mixer(tyloi::ops::Residual weights) { return std::make_unique<tyloi::ops::Residual>(std::move(weights)); }
#else
    GatedResidual residual(DenseTensor norm, GatedResidual::Projection down,
        GatedResidual::Projection up, GatedResidual::Projection injection, const Config &c) const {
        return {std::move(norm), std::move(down), std::move(up), std::move(injection),
                c.hidden, c.streams, c.eps, execution.executor().options().residual, execution.executor().mode().ffn.batch_serial};
    }
    static auto final_mixer(GatedResidual weights) { return std::make_unique<GatedResidual>(std::move(weights)); }
#endif
    auto gdn_weights(const std::string &p) const {
        const bool independent = has(p + ".query.weight") || has(p + ".key.weight") || has(p + ".value.weight");
        TYLOINFER_RUNTIME_CHECK(!independent || !has(p + ".qkv.weight"), "ambiguous Qwen4 GDN input projections");
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::ops::GdnWeights w;
#else
        GdnWeights w;
#endif
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
        w.conv=dense(p+".conv.weight").dense();
        w.dt_bias=dense(p+".dt_bias").dense();
        w.a_log=dense(p+".a").dense();
        w.norm=dense(p+".norm.weight").dense();
#endif
        return w;
    }
    auto qsa_weights(const std::string &p) const {
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::ops::QsaWeights w;
#else
        QsaWeights w;
#endif
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
        w.query_norm=dense(p+".query_norm.weight").dense();
        w.key_norm=dense(p+".key_norm.weight").dense();
        w.index_query_norm=dense(p+".indexer.query_norm.weight").dense();
        w.index_key_norm=dense(p+".indexer.key_norm.weight").dense();
#endif
        if(has(p+".gate.weight")) {
            w.input_projection.push_back(weight(p+".gate.weight"));
            w.has_gate=true;
        }
        return w;
    }
#ifdef TYLOINFER_NATIVE_CUDA
    auto gdn(tyloi::ops::GdnWeights weights, const Config &c) const {
#else
    auto gdn(GdnWeights weights, const Config &c) const {
#endif
        weights.input_projection=tyloi::weights::pack_projections(std::move(weights.input_projection),
            options.grouped,placement.cpu);
#ifdef TYLOINFER_NATIVE_CUDA
        return std::make_unique<tyloi::ops::Gdn>(std::move(weights),c.key_heads,c.value_heads,
            c.linear_width,c.kernel,c.eps,c.silu_gate);
#else
        auto shared=std::make_shared<const GdnWeights>(std::move(weights));
        auto& executor=execution.executor();
        return std::make_unique<Gdn>(std::move(shared), c.key_heads, c.value_heads, c.linear_width,
                                     c.kernel, c.eps, c.silu_gate,true,executor.mode().transposed,
                                     executor.options().gdn_fused_output,true,executor.options().gdn_fused_preparation,
                                     executor.options().gdn_fused_core);
#endif
    }
#ifdef TYLOINFER_NATIVE_CUDA
    auto qsa(tyloi::ops::QsaWeights weights, const Config &c) const {
#else
    auto qsa(QsaWeights weights, const Config &c) const {
#endif
        weights.input_projection=tyloi::weights::pack_projections(std::move(weights.input_projection),
            options.grouped,placement.cpu);
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::ops::QsaConfig geometry{c.heads,c.kv_heads,c.width,c.index_heads,c.index_width,
            c.pool,c.budget,c.maximum,c.eps,c.rotary,c.rope_base,c.sections,c.interleaved,yarn};
        geometry.device=tyloi::Device(tyloi::kCUDA,placement.device);
#else
        QsaConfig geometry{c.heads, c.kv_heads, c.width, c.index_heads, c.index_width,
                           c.pool, c.budget, c.maximum, c.eps};
        constexpr bool default_rotary_fusion = false;
        auto rotary = std::make_shared<RotaryEmbedding>(c.rotary, c.maximum, c.rope_base,
            c.sections, c.interleaved,
            ::tyloi::cuda::options::rotary_fusion().value_or(default_rotary_fusion),
            execution.config.yarn);
#endif
        for(auto* norm:{&weights.query_norm,&weights.key_norm,&weights.index_query_norm,&weights.index_key_norm})
#ifdef TYLOINFER_NATIVE_CUDA
            if(norm->defined())*norm=tyloi::ops::contiguous(tyloi::ops::cast(*norm,::tyloi::kFloat32));
#else
            if(norm->defined())*norm=norm->to(::tyloi::kFloat32).contiguous();
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        return std::make_unique<tyloi::ops::Qsa>(std::move(weights),std::move(geometry));
#else
        auto shared=std::make_shared<const QsaWeights>(std::move(weights));
        return std::make_unique<Qsa>(std::move(shared), geometry, std::move(rotary), true, true, execution.offload, execution.kv);
#endif
    }
    auto moe(std::vector<Linear> gate_up, Linear down, Linear router,
                    Linear shared_gate, Linear sg, Linear su, Linear sd, const Config &c) const {
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::ops::FfnWeights weights;
        weights.gate_up=std::move(gate_up); weights.down=std::move(down);
        weights.moe_router=std::move(router); weights.moe_shared_gate=std::move(shared_gate);
        weights.shared=std::make_unique<tyloi::ops::FfnWeights>();
        weights.shared->gate_up={std::move(sg),std::move(su)}; weights.shared->down=std::move(sd);
        const auto routing=c.routing();
        weights.moe_top_k=c.topk; weights.moe_use_sigmoid=routing.activation==tyloi::ops::RouterActivation::sigmoid;
        weights.moe_use_sqrt_softplus=routing.activation==tyloi::ops::RouterActivation::sqrt_softplus;
        weights.moe_normalize=routing.normalize; weights.moe_delayed_softmax=routing.delayed_softmax;
        weights.moe_router_scale=routing.scale; weights.policy=tyloi::ops::FfnPolicy::qwen4;
        weights.qwen_hidden=c.hidden; weights.qwen_moe_width=c.moe_width;
        return tyloi::ops::qwen4_ffn_weights(std::move(weights));
#else
        return Moe(std::move(gate_up), std::move(down), std::move(router), std::move(shared_gate),
                   std::move(sg), std::move(su), std::move(sd), c);
#endif
    }
    auto embedding(const std::string &name) const {
#ifdef TYLOINFER_NATIVE_CUDA
        auto table = this->rows(name);
        const std::array<int64_t,2> shape{table.size(0),table.size(1)};
        return std::pair{std::move(table),shape};
#else
        if(tyloi::is_tq_table(source,name)) {
            auto table=tyloi::tq_table(source,name);
            Embedding lookup{[table](const DenseTensor& ids){
                auto result=lookup_tq_table(*table,ids);
                return result.dense();
            },{},table};
            return std::pair{std::move(lookup),std::array<int64_t,2>{static_cast<int64_t>(table->rows()),table->width()}};
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (require_tensor(source, name).dtype == "NINT") {
            auto table = tyloi::weights::load_nint_rows(source, name);
            Embedding lookup{[table](const DenseTensor &ids) {
                return nint_row_embedding_lookup(*table, ids);
            },table};
            return std::pair{std::move(lookup), std::array<int64_t, 2>{table->rows(), table->width()}};
        }
#endif

        auto weight = this->weight(name);
        Embedding lookup{[weight](const DenseTensor &ids) { return quant_embedding_lookup(weight, ids); },{}};
        return std::pair{std::move(lookup), std::array<int64_t, 2>{weight.size(0), weight.size(1)}};
#endif
    }
    std::vector<int64_t> integers(const std::string &name) const {
        const auto &type = require_tensor(source, name).dtype;
        TYLOINFER_RUNTIME_CHECK(type == "I64" || type == "I32",
                          "Qwen4 PLE metadata must be a dense integer array: ", name);
        auto host = tensor(name).dense().to(::tyloi::kInt64).contiguous().cpu();
        TYLOINFER_RUNTIME_CHECK(host.dim() == 1, "Qwen4 PLE metadata must be a vector");
        return {host.data_ptr<int64_t>(), host.data_ptr<int64_t>() + host.numel()};
    }
    static auto ple(std::vector<Embedding> shards, int64_t rows, int64_t width, const Config &c,
                    std::vector<int64_t> multipliers, std::vector<int64_t> offsets,
                    std::vector<int64_t> sizes, PleWeights weights) {
#ifdef TYLOINFER_NATIVE_CUDA
        NgramEmbedding embedding(std::move(shards), rows, width, c.ngram, c.ngram_heads, c.eos,
                                 std::move(multipliers), std::move(offsets), std::move(sizes));
#else
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
#endif
        return std::make_unique<Ple>(std::move(embedding), std::move(weights), c.hidden, c.streams,
                                     c.ngram, c.eps);
    }
    std::unique_ptr<Layer> layer(const Config &c, int layer, bool predictor);
};

std::unique_ptr<Layer> LayerLoader::layer(const Config &c, int layer, bool predictor) {
    auto result = std::make_unique<Layer>();
#ifdef TYLOINFER_NATIVE_CUDA
    LayerLoader loader{*this, yarn};
#else
    LayerLoader loader{*this, execution};
#endif
    loader.role = "qwen4_exp";
    tyloi::models::qwen4_exp::load_layer(*result, loader, c, layer, predictor);
    return result;
}

#ifdef TYLOINFER_NATIVE_CUDA
std::optional<Predictor> Predictor::load_if_present(const tyloi::weights::Loader& weights,
                                                      const tyloi::models::qwen4_exp::Config &main,
                                                      ::tyloi::Tensor embedding, ::tyloi::Tensor output,
                                                      double softcap, const tyloi::ops::YarnScaling& yarn) {
#else
std::optional<Predictor> Predictor::load_if_present(CudaExecutionContext &execution,
                                                      const tyloi::weights::Loader& weights,
                                                      const tyloi::models::qwen4_exp::Config &main,
                                                      ::tyloi::Tensor embedding, ::tyloi::Tensor output,
                                                      double softcap) {
#endif
    Predictor result;
#ifndef TYLOINFER_NATIVE_CUDA
    result.execution = &execution;
#endif
    result.embedding = std::move(embedding);
    result.output = std::move(output);
    result.softcap = softcap;
    result.primary = weights.placement.device;
#ifdef TYLOINFER_NATIVE_CUDA
    const auto enabled = [](const char* name) {
        const auto* value = tyloi::cuda::env_lookup(name);
        return !value || value[0] != '0';
    };
    result.history_stream = enabled("TYLOINFER_MTP_STREAM_PROMPT_HISTORY");
    result.history_bootstrap = enabled("TYLOINFER_MTP_EARLY_BOOTSTRAP");
    result.history_early_flush = enabled("TYLOINFER_MTP_HISTORY_EARLY_FLUSH");
    result.history_cache_only = enabled("TYLOINFER_MTP_CACHE_ONLY_PRIME");
    if (const auto* host = tyloi::cuda::env_lookup("TYLOINFER_MTP_HOST_PROMPT_HISTORY"))
        result.history_host = host[0] != '0';
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    LayerLoader loader{weights,yarn};
#else
    LayerLoader loader{weights,execution};
#endif
    loader.role = "qwen4_exp";
    if (!tyloi::models::qwen4_exp::load_predictor(result, loader, main))
        return std::nullopt;
    return result;
}

using tyloi::models::require_model;

void read(Model& model, std::shared_ptr<const tyloi::ModelSource> model_source,
          std::string_view definition, std::int64_t context_size) {
    model.source = std::move(model_source);
    const auto& source = *model.source;
    validate_model_source(source);
    model.graph = tyloi::models::graph(source);
    model.plan = cuda_model_plan(model.graph);
    require_model(Model::accepts_backbone(model.graph.backbone),
                  "model backbone does not match the requested causal model");
    model.load_config(definition);
    require_model(model.num_hidden_layers() == model.graph.topology.text_layers,
                  "model graph/config text-layer topology mismatch");
    if (context_size > 0) {
        require_model(context_size <= model.max_position_embeddings(),
                      "--ctx-size exceeds max_position_embeddings");
        model.config.maximum = context_size;
        model.metadata.max_position_embeddings = context_size;
    }
}

#ifdef TYLOINFER_NATIVE_CUDA
void load(Model& model, const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
          const tyloi::models::LoadOptions& options, bool load_layers) {
    validate_native_placement(weights.placement.tensor, placement, options.gpu_layers);
    if (options.expert_parallel && model.metadata.num_experts <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    placement.prepare(model.num_hidden_layers());
    const int cpu_layers=options.gpu_layers < 0 ? 0 : static_cast<int>(
        std::max<std::int64_t>(model.num_hidden_layers()-options.gpu_layers, 0));
    model.primary_device=placement.primary_device();
    model.needs_host_tokens=options.host_tokens || cpu_layers>0;
    model.parallel=weights.placement.tensor.enabled() || options.expert_parallel;
    if (cpu_layers > 0) {
        if (weights.placement.tensor.enabled() || options.expert_parallel || placement.enabled())
            throw std::runtime_error("--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
        std::cerr << "dense_layer_placement cpu=0-" << cpu_layers-1 << " gpu=" << cpu_layers << '-'
                  << model.num_hidden_layers()-1 << " cpu_threads=" << mfq_get_num_threads() << std::endl;
    }
    if (model.plan.vision == CudaVisionAdapter::grid_vit && cpu_layers > 0)
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");
    MfqCudaGuard model_guard(weights.placement.device);
    constexpr auto embedding = "model.token_embedding.weight";
    constexpr auto output = "model.output.weight";
    const bool independent = !model.tie_word_embeddings() && weights.has(output);
    model.embed = weights.embedding(embedding, independent);
    LayerLoader loader{weights, options.yarn};
    loader.role = "qwen4_exp";
    model.final_mixer = loader.final_mixer(
        tyloi::models::qwen4_exp::load_residual(loader, model.config, "model.mhc.pre", false));
    model.lm_head = independent ? weights.weight(output, TensorParallelAxis::Output)
        : weights.placement.tensor.enabled() ? weights.weight(embedding, TensorParallelAxis::Output) : model.embed;
    if (load_layers) {
        model.layers.reserve(static_cast<std::size_t>(model.num_hidden_layers()));
        for (int layer=0; layer<model.num_hidden_layers(); ++layer) {
            const int device=placement.device_for_layer(layer);
            const bool cpu=layer<cpu_layers;
            MfqCudaGuard layer_guard(device);
            std::cerr << "loading layer " << layer << ' '
                      << model.metadata.layer_types.at(static_cast<std::size_t>(layer)) << ' '
                      << (cpu ? "CPU" : "CUDA") << std::endl;
            LayerLoader placed{weights.for_layer(device,cpu), options.yarn};
            auto block=placed.layer(model.config,layer,false);
            block->cuda_device=device;
            block->cpu_offloaded=cpu;
            model.layers.push_back(std::move(block));
        }
    }
}
#else
void load(Model& model, CudaExecutionContext& execution, std::int64_t context_size,
          bool defer_moe_cache_finalize, bool load_layers) {
    auto definition = tyloi::models::config_json(*model.source, {});
    definition = configure_yarn(execution.config.yarn, execution.config.yarn_context_extension,
        std::move(definition), model.graph.backbone, context_size);
    read(model, model.source, definition, context_size);
    model.execution = &execution;
    validate_native_placement(execution.config.tensor_parallel, execution.config.layer_placement,
                              execution.config.n_gpu_layers);
    if (execution.config.expert_parallel.enabled() && model.metadata.num_experts <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    execution.config.layer_placement.prepare(model.num_hidden_layers());
    execution.config.dense_cpu_layer_count = 0;
    if (execution.config.n_gpu_layers >= 0) {
        execution.config.dense_cpu_layer_count = static_cast<int>(
            std::max<std::int64_t>(model.num_hidden_layers() - execution.config.n_gpu_layers, 0));
        if (execution.config.dense_cpu_layer_count > 0) {
            if (model_parallel_enabled(execution.config) || execution.config.layer_placement.enabled())
                throw std::runtime_error(
                    "--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            std::cerr << "dense_layer_placement cpu=0-" << (execution.config.dense_cpu_layer_count - 1)
                      << " gpu=" << execution.config.dense_cpu_layer_count << '-'
                      << (model.num_hidden_layers() - 1) << " cpu_threads=" << mfq_get_num_threads()
                      << std::endl;
        }
    }
    CudaScope model_guard(execution.config.layer_placement.primary_device(), execution.executor().contexts());
    if (model.plan.vision == CudaVisionAdapter::grid_vit && execution.config.dense_cpu_layer_count > 0)
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");
    auto tensors = execution.weights.loader(*model.source, execution.config.weights,
        {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
         execution.config.tensor_parallel, moe_parallel_config(execution.config)});
    constexpr auto embedding = "model.token_embedding.weight";
    constexpr auto output = "model.output.weight";
    const bool independent = !model.tie_word_embeddings() && tensors.has(output);
    model.embed = tensors.embedding(embedding, independent);
    LayerLoader loader{tensors, execution};
    loader.role = "qwen4_exp";
    model.final_mixer = loader.final_mixer(
        tyloi::models::qwen4_exp::load_residual(loader, model.config, "model.mhc.pre", false));
    model.lm_head = independent ? tensors.weight(output, TensorParallelAxis::Output)
        : tensors.placement.tensor.enabled() ? tensors.weight(embedding, TensorParallelAxis::Output)
                                              : model.embed;
    if (load_layers) {
        model.layers.reserve(static_cast<std::size_t>(model.num_hidden_layers()));
        for (int layer = 0; layer < model.num_hidden_layers(); ++layer) {
            const int device = execution.config.layer_placement.device_for_layer(layer);
            const bool cpu = layer < execution.config.dense_cpu_layer_count;
            MfqCudaGuard layer_guard(device);
            std::cerr << "loading layer " << layer << ' '
                      << model.metadata.layer_types.at(static_cast<std::size_t>(layer)) << ' '
                      << (cpu ? "CPU" : "CUDA") << std::endl;
            LayerLoader placed{tensors.for_layer(device, cpu), execution};
            auto block = placed.layer(model.config, layer, false);
            block->cuda_device = device;
            block->cpu_offloaded = cpu;
            model.layers.push_back(std::move(block));
        }
    }
    if (!defer_moe_cache_finalize)
        execution.weights.finalize();
}
#endif

} // namespace tyloi::cuda::qwen4_exp
