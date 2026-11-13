#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/qwen4_exp/layer.h"
#include "engine/models/qwen4_exp/causal_lm.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/backends/cuda/ops/projection/nint.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/weights/loader.h"
#include "engine/weights/group.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "env.h"

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
    using PleWeights = tyloi::ops::PleWeights;
    tyloi::ops::YarnScaling yarn;
    using Embedding = ::tyloi::Tensor;

    auto residual_linear(const std::string &name) const {
        return weight(name);
    }
    tyloi::ops::Residual residual(::tyloi::Tensor norm, ::tyloi::Tensor down,
        ::tyloi::Tensor up, ::tyloi::Tensor injection, const Config &c) const {
        const bool final=!injection.defined();
        return {tyloi::ops::ResidualWeights{std::move(norm),std::move(down),std::move(up),std::move(injection)},
                tyloi::ops::ResidualConfig{c.hidden,c.streams,c.eps,final,tyloi::Device(tyloi::kCUDA,placement.device)}};
    }
    static auto final_mixer(tyloi::ops::Residual weights) { return std::make_unique<tyloi::ops::Residual>(std::move(weights)); }
    auto gdn_weights(const std::string &p) const {
        const bool independent = has(p + ".query.weight") || has(p + ".key.weight") || has(p + ".value.weight");
        TYLOINFER_RUNTIME_CHECK(!independent || !has(p + ".qkv.weight"), "ambiguous Qwen4 GDN input projections");
        tyloi::ops::GdnWeights w;
        if(independent)for(const auto* name:{"query","key","value"})
            w.input_projection.push_back(weight(p+"."+name+".weight"));
        else w.input_projection.push_back(weight(p+".qkv.weight"));
        w.input_projection.push_back(weight(p+".gate.weight"));
        w.input_projection.push_back(weight(p+".alpha.weight"));
        w.input_projection.push_back(weight(p+".beta.weight"));
        w.qkv_count=independent?3:1;
        w.output=weight(p+".output.weight");
        w.conv=dense(p+".conv.weight");
        w.dt_bias=dense(p+".dt_bias");
        w.a_log=dense(p+".a");
        w.norm=dense(p+".norm.weight");
        return w;
    }
    auto qsa_weights(const std::string &p) const {
        tyloi::ops::QsaWeights w;
        for(const auto* name:{"query","key","value"})
            w.input_projection.push_back(weight(p+"."+name+".weight"));
        w.input_projection.push_back(weight(p+".indexer.query_key.weight"));
        w.output=weight(p+".output.weight");
        w.query_norm=dense(p+".query_norm.weight");
        w.key_norm=dense(p+".key_norm.weight");
        w.index_query_norm=dense(p+".indexer.query_norm.weight");
        w.index_key_norm=dense(p+".indexer.key_norm.weight");
        if(has(p+".gate.weight")) {
            w.input_projection.push_back(weight(p+".gate.weight"));
            w.has_gate=true;
        }
        return w;
    }
    auto gdn(tyloi::ops::GdnWeights weights, const Config &c) const {
        weights.input_projection=tyloi::weights::pack_projections(std::move(weights.input_projection),
            options.grouped,placement.cpu);
        return std::make_unique<tyloi::ops::Gdn>(std::move(weights),c.key_heads,c.value_heads,
            c.linear_width,c.kernel,c.eps,c.silu_gate,tyloi::Device(tyloi::kCUDA,placement.device));
    }
    auto qsa(tyloi::ops::QsaWeights weights, const Config &c) const {
        weights.input_projection=tyloi::weights::pack_projections(std::move(weights.input_projection),
            options.grouped,placement.cpu);
        tyloi::ops::QsaConfig geometry{c.heads,c.kv_heads,c.width,c.index_heads,c.index_width,
            c.pool,c.budget,c.maximum,c.eps,c.rotary,c.rope_base,c.sections,c.interleaved,yarn};
        geometry.device=tyloi::Device(tyloi::kCUDA,placement.device);
        for(auto* norm:{&weights.query_norm,&weights.key_norm,&weights.index_query_norm,&weights.index_key_norm})
            if(norm->defined())*norm=tyloi::ops::contiguous(tyloi::ops::cast(*norm,::tyloi::kFloat32));
        return std::make_unique<tyloi::ops::Qsa>(std::move(weights),std::move(geometry));
    }
    auto moe(std::vector<Linear> gate_up, Linear down, Linear router,
                    Linear shared_gate, Linear sg, Linear su, Linear sd, const Config &c) const {
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
        weights.target=tyloi::Device(tyloi::kCUDA,placement.device);
        return tyloi::ops::qwen4_ffn_weights(std::move(weights));
    }
    auto embedding(const std::string &name) const {
        auto table = this->rows(name);
        const std::array<int64_t,2> shape{table.size(0),table.size(1)};
        return std::pair{std::move(table),shape};
    }
    std::vector<int64_t> integers(const std::string &name) const {
        const auto &type = require_tensor(source, name).dtype;
        TYLOINFER_RUNTIME_CHECK(type == "I64" || type == "I32",
                          "Qwen4 PLE metadata must be a dense integer array: ", name);
        auto host = tensor(name).dense().to(::tyloi::kInt64).contiguous().cpu();
        TYLOINFER_RUNTIME_CHECK(host.dim() == 1, "Qwen4 PLE metadata must be a vector");
        return {host.data_ptr<int64_t>(), host.data_ptr<int64_t>() + host.numel()};
    }
    auto ple(std::vector<Embedding> shards, int64_t rows, int64_t width, const Config &c,
             std::vector<int64_t> multipliers, std::vector<int64_t> offsets,
             std::vector<int64_t> sizes, PleWeights weights) const {
        NgramEmbedding embedding(std::move(shards), rows, width, c.ngram, c.ngram_heads, c.eos,
                                 std::move(multipliers), std::move(offsets), std::move(sizes));
        return std::make_unique<Ple>(std::move(embedding), std::move(weights), c.hidden, c.streams,
                                     c.ngram, c.eps
                                     ,tyloi::Device(tyloi::kCUDA,placement.device)
                                     );
    }
    std::unique_ptr<Layer> layer(const Config &c, int layer, bool predictor);
};

std::unique_ptr<Layer> LayerLoader::layer(const Config &c, int layer, bool predictor) {
    auto result = std::make_unique<Layer>();
    LayerLoader loader{*this, yarn};
    loader.role = "qwen4_exp";
    tyloi::models::qwen4_exp::load_layer(*result, loader, c, layer, predictor);
    return result;
}

std::optional<Predictor> Predictor::load_if_present(const tyloi::weights::Loader& weights,
                                                      const tyloi::models::qwen4_exp::Config &main,
                                                      ::tyloi::Tensor embedding, ::tyloi::Tensor output,
                                                      double softcap, const tyloi::ops::YarnScaling& yarn) {
    Predictor result;
    result.embedding = std::move(embedding);
    result.output = std::move(output);
    result.softcap = softcap;
    result.primary = weights.placement.device;
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
    LayerLoader loader{weights,yarn};
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
    model.plan = ::tyloi::models::model_plan(model.graph);
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
    if (model.plan.vision == ::tyloi::models::Vision::grid_vit && cpu_layers > 0)
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");
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

} // namespace tyloi::cuda::qwen4_exp
