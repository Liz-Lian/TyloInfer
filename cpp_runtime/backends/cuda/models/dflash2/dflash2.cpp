#include "dflash2.h"
#include "weight_quant.h"
#include "gguf_weight.h"
#include "models/qwen4_exp/ops.h"
#include "../../kernels/dflash2.h"
#include "mfq_cuda_sampling_ops.h"
#include "mfq_weight_memory.h"
#include "mfq_cuda_resident_allocations.h"
#include "runtime/decode_window.h"
#include "runtime/execution_options.h"
#include "tyloinfer/hf_safetensors_source.h"
#include "tyloinfer/runtime.h"
#include "ggml-quants.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <limits>
#include <map>
#include <functional>
#include <utility>
#include <nlohmann/json.hpp>

namespace tb = mfq_tensor_backend;
namespace mfq::cuda::dflash2 {
Config Config::load(const std::filesystem::path& directory) {
    std::ifstream input(directory / "config.json");
    nlohmann::json json;
    if (!input || !(input >> json)) throw std::invalid_argument("DFlash2 config is unavailable");
    const auto& d = json.at("dflash_config");
    Config c;
    c.hidden = json.at("hidden_size"); c.vocabulary = json.at("vocab_size");
    c.layers = json.at("num_hidden_layers"); c.intermediate = json.at("intermediate_size");
    c.heads = json.at("num_attention_heads"); c.kv_heads = json.at("num_key_value_heads");
    c.dimension = json.at("head_dim"); c.window = json.at("sliding_window");
    c.block = d.at("block_size"); c.group = d.at("conv_group_size");
    c.rank = d.at("selector_rank"); c.candidates = d.at("selector_top_k");
    c.mask = d.at("mask_token_id"); c.target_layers = d.at("target_layer_ids").get<std::vector<int64_t>>();
    c.theta = json.at("rope_parameters").at("rope_theta"); c.epsilon = json.at("rms_norm_eps");
    if (c.hidden <= 0 || c.hidden % 32 || c.vocabulary <= 0 || c.layers <= 0 || c.intermediate <= 0 ||
        c.intermediate % 32 || c.heads <= 0 || c.kv_heads <= 0 || c.heads % c.kv_heads || c.dimension <= 0 ||
        c.dimension > 512 || c.dimension % 2 || c.heads * int64_t(c.dimension) % 32 ||
        c.window < 1 || c.block < 2 || c.group <= 0 || c.hidden % c.group || d.at("conv_kernel_size") != 2 ||
        c.rank <= 0 || c.candidates < 1 || c.candidates > std::min(64, c.vocabulary) || c.mask < 0 || c.mask >= c.vocabulary ||
        c.target_layers.empty() || c.target_layers.front() < 0 ||
        !std::is_sorted(c.target_layers.begin(), c.target_layers.end()) ||
        std::adjacent_find(c.target_layers.begin(), c.target_layers.end()) != c.target_layers.end() ||
        !std::isfinite(c.theta) || c.theta <= 0 || !std::isfinite(c.epsilon) || c.epsilon <= 0)
        throw std::invalid_argument("DFlash2 checkpoint geometry is invalid");
    return c;
}
namespace {
using Tensor = tb::Tensor;
struct PreparedActivation {Tensor values,codes,scales;};
PreparedActivation prepared_storage(Tensor values) {
    PreparedActivation result;result.values=std::move(values);
    const auto width=result.values.size(-1),rows=result.values.numel()/width;
    if(rows<1 || rows>8 || width%32)return result;
    const auto padded=(width+511)/512*512;
    result.codes=tb::empty({8*padded},result.values.options().dtype(tb::kUInt8));
    result.scales=tb::empty({8*(padded/32)},result.values.options());
    return result;
}
// Strata's Q8_0 tile image replaces the original packed allocation. Large
// prompt seeding reconstructs a temporary GGUF view for the existing PP GEMM.
struct TiledLinear {
    Tensor tiles;
    int64_t out=0,columns=0,padded_out=0,padded_in=0;
    TiledLinear()=default;
    TiledLinear(const GgufWeight& weight) {
        if(weight.type!=GGML_TYPE_Q8_0 || !weight.packed.is_cuda() || weight.out<=0 ||
                weight.out>std::numeric_limits<int>::max()-31 || weight.neuron_len<=0 ||
                weight.neuron_len>std::numeric_limits<int>::max()-511 || weight.neuron_len%32)
            throw std::invalid_argument("DFlash2 tiled linear requires a valid Q8_0 matrix");
        const WeightAllocationScope allocation;
        out=weight.out;columns=weight.neuron_len;
        padded_out=(out+31)/32*32;padded_in=(columns+511)/512*512;
        tiles=tb::empty({padded_out*(padded_in/32)*34},weight.packed.options());
        const auto stream=mfq_current_cuda_stream();
        weight.packed.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        tiles.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        q8_tile(weight.packed.data_ptr(),tiles.data_ptr(),int(out),int(columns),stream);
    }
    static std::vector<Tensor> group(std::initializer_list<const TiledLinear*> weights,Tensor input,const PreparedActivation* prepared=nullptr) {
        if(weights.size()<1 || weights.size()>3)throw std::invalid_argument("DFlash2 projection group differs");
        const auto& first=**weights.begin();
        if(input.dim()<1 || input.size(-1)!=first.columns || input.numel()<first.columns)
            throw std::invalid_argument("DFlash2 projection input differs");
        input=input.to(first.tiles.device()).to(tb::kFloat32).contiguous();
        const auto tokens=input.numel()/first.columns;
        const auto stream=mfq_current_cuda_stream();
        const auto stream_id=reinterpret_cast<std::uintptr_t>(stream);
        input.record_stream(stream_id);
        std::vector<Tensor> outputs;
        if(tokens>std::numeric_limits<int>::max()-7)
            throw std::invalid_argument("DFlash2 padded projection rows overflow");
        const auto padded_tokens=(tokens+7)/8*8;
        auto codes=prepared && prepared->codes.defined()?prepared->codes:tb::empty({padded_tokens*first.padded_in},first.tiles.options());
        auto scales=prepared && prepared->scales.defined()?prepared->scales:tb::empty({padded_tokens*(first.padded_in/32)},input.options());
        if(!prepared || !prepared->codes.defined())
            q8_quantize(input.data_ptr<float>(),codes.data_ptr(),scales.data_ptr<float>(),int(tokens),int(first.columns),stream);
        Q8TiledMatrix matrices[3];int index=0;
        for(const auto* weight:weights) {
            if(weight->columns!=first.columns || weight->tiles.device()!=first.tiles.device())
                throw std::invalid_argument("DFlash2 projection group input differs");
            auto shape=input.sizes().vec();shape.back()=weight->out;
            outputs.push_back(tb::empty(shape,input.options()));
            weight->tiles.record_stream(stream_id);outputs.back().record_stream(stream_id);
            matrices[index++]={weight->tiles.data_ptr(),outputs.back().data_ptr<float>(),int(weight->out),int(weight->padded_out)};
        }
        codes.record_stream(stream_id);scales.record_stream(stream_id);
        q8_gemm(matrices,index,codes.data_ptr(),scales.data_ptr<float>(),int(tokens),int(first.columns),stream);
        return outputs;
    }
    Tensor forward(Tensor input,const PreparedActivation* prepared=nullptr) const {return group({this},std::move(input),prepared).front();}
};
struct Loader {
    mfq::HfSafetensorsSource source;
    int device;
    explicit Loader(const std::filesystem::path& directory, int selected) : source(directory), device(selected) {}
    Tensor host(const std::string& name, const std::vector<int64_t>& shape) {
        const auto* record = source.find_tensor(name);
        if (!record || !record->shape || *record->shape != shape)
            throw std::invalid_argument("DFlash2 tensor geometry differs: " + name);
        auto type = record->dtype == "BF16" ? tb::kBFloat16 : record->dtype == "F16" ? tb::kFloat16 : tb::kFloat32;
        if (record->dtype != "BF16" && record->dtype != "F16" && record->dtype != "F32")
            throw std::invalid_argument("DFlash2 tensor precision differs: " + name);
        auto result = tb::empty(shape, tb::TensorOptions().device(tb::kCPU).dtype(type));
        if (result.nbytes() != record->nbytes) throw std::invalid_argument("DFlash2 tensor payload differs: " + name);
        source.read_range_into(name, 0, static_cast<std::byte*>(result.data_ptr()), result.nbytes()); return result;
    }
    Tensor dense(const std::string& name, const std::vector<int64_t>& shape, bool codebook = false) {
        const WeightAllocationScope weights;
        return host(name, shape).to(codebook ? tb::kBFloat16 : tb::kFloat32).to(tb::Device(tb::kCUDA, device));
    }
    GgufWeight linear(const std::string& name, int64_t rows, int64_t columns) {
        const WeightAllocationScope weights;
        auto input = host(name + ".weight", {rows, columns});
        const auto stride = mfq::gguf::row_bytes(GGML_TYPE_Q8_0, columns);
        auto packed=pack_q8_weight_cpu(input);
        return {packed.to(tb::Device(tb::kCUDA, device)), GGML_TYPE_Q8_0, rows, columns, stride};
    }
};
Tensor normalized(const Tensor& input, const Tensor& norm, float epsilon) {
    auto x=input.to(tb::kFloat32).contiguous(),output=tb::empty_like(x);
    rms_norm(x.data_ptr<float>(),norm.data_ptr<float>(),output.data_ptr<float>(),
        int(x.numel()/x.size(-1)),int(x.size(-1)),epsilon,mfq_current_cuda_stream());
    return output;
}
PreparedActivation normalized_q8(const Tensor& input,const Tensor& norm,float epsilon) {
    PreparedActivation prepared;
    auto source=input.to(tb::kFloat32).contiguous();
    const auto width=source.size(-1),rows=source.numel()/width;
    if(rows<1 || rows>8 || width%32){prepared.values=normalized(source,norm,epsilon);return prepared;}
    const auto padded=(width+511)/512*512;
    prepared.values=tb::empty_like(source);
    prepared.codes=tb::empty({8*padded},source.options().dtype(tb::kUInt8));
    prepared.scales=tb::empty({8*(padded/32)},source.options());
    rms_norm_q8(source.data_ptr<float>(),norm.data_ptr<float>(),prepared.values.data_ptr<float>(),
        prepared.codes.data_ptr(),prepared.scales.data_ptr<float>(),int(rows),int(width),epsilon,mfq_current_cuda_stream());
    return prepared;
}
Tensor convolved(const Tensor& input, const Tensor& dynamic, const Tensor& base, const Tensor& residual,
    int group, bool finish) {
    auto output = tb::empty_like(input);
    convolution(input.data_ptr<float>(), dynamic.data_ptr<float>(), base.data_ptr<float>(),
        finish ? residual.data_ptr<float>() : nullptr, output.data_ptr<float>(), int(input.size(1)), int(input.size(2)), group, finish,
        mfq_current_cuda_stream()); return output;
}
PreparedActivation convolved_q8(const Tensor& input,const Tensor& dynamic,const Tensor& base,int group) {
    auto output=prepared_storage(tb::empty_like(input));
    if(output.codes.defined())
        convolution_q8(input.data_ptr<float>(),dynamic.data_ptr<float>(),base.data_ptr<float>(),nullptr,
            output.values.data_ptr<float>(),output.codes.data_ptr(),output.scales.data_ptr<float>(),
            int(input.size(1)),int(input.size(2)),group,false,mfq_current_cuda_stream());
    else output.values=convolved(input,dynamic,base,{},group,false);
    return output;
}
Tensor rotation_table(const Tensor& positions,const Config& c) {
    auto table=tb::empty({positions.numel(),c.dimension},positions.options().dtype(tb::kFloat32));
    rotary_table(positions.data_ptr<int64_t>(),table.data_ptr<float>(),int(positions.numel()),c.dimension,c.theta,mfq_current_cuda_stream());
    return table;
}
Tensor rotated(const Tensor& input, const Tensor& norm, const Tensor& table, int heads, const Config& c) {
    auto source = input.to(tb::kFloat32).contiguous(), output = tb::empty_like(source);
    norm_rope_tabled(source.data_ptr<float>(), norm.data_ptr<float>(), table.data_ptr<float>(), output.data_ptr<float>(),
        int(source.size(1)), heads, c.dimension, c.epsilon, mfq_current_cuda_stream());
    return output.reshape({1, source.size(1), heads, c.dimension});
}
struct Layer {
    Tensor input_norm, attention_base, q_norm, k_norm, post_norm, mlp_base;
    TiledLinear dynamic_attention, q, k, v, output, dynamic_mlp, gate, up, down;
    Layer(Loader& loader, const Config& c, int index) {
        const auto p = "layers." + std::to_string(index) + ".";
        input_norm = loader.dense(p + "input_layernorm.weight", {c.hidden});
        attention_base = loader.dense(p + "attention_conv.base_kernel", {2, 2, c.hidden});
        q_norm = loader.dense(p + "self_attn.q_norm.weight", {c.dimension});
        k_norm = loader.dense(p + "self_attn.k_norm.weight", {c.dimension});
        post_norm = loader.dense(p + "post_attention_layernorm.weight", {c.hidden});
        mlp_base = loader.dense(p + "mlp_conv.base_kernel", {2, 2, c.hidden});
        dynamic_attention = loader.linear(p + "attention_conv.kernel_projection", 4 * c.hidden / c.group, c.hidden);
        q = loader.linear(p + "self_attn.q_proj", c.heads * c.dimension, c.hidden);
        k = loader.linear(p + "self_attn.k_proj", c.kv_heads * c.dimension, c.hidden);
        v = loader.linear(p + "self_attn.v_proj", c.kv_heads * c.dimension, c.hidden);
        output = loader.linear(p + "self_attn.o_proj", c.hidden, c.heads * c.dimension);
        dynamic_mlp = loader.linear(p + "mlp_conv.kernel_projection", 4 * c.hidden / c.group, c.hidden);
        gate = loader.linear(p + "mlp.gate_proj", c.intermediate, c.hidden);
        up = loader.linear(p + "mlp.up_proj", c.intermediate, c.hidden);
        down = loader.linear(p + "mlp.down_proj", c.hidden, c.intermediate);
    }
};
struct Draft final : MtpModule {
    CudaExecutionContext& execution;
    Config c;
    TiledLinear context_projection, selector_projection;
    Tensor hidden_norm, final_norm, predecessor, successor, mask_embedding;
    const std::uint16_t* predecessor_address=nullptr;
    const std::uint16_t* successor_address=nullptr;
    int codebook_device=0;
    bool codebooks_prepared=false;
    bool compact_kv=false;
    bool stage_successor=false;
    Tensor successor_rows_host,successor_rows_device;
    mutable std::shared_ptr<Event> context_fence;
    mutable cudaStream_t context_stream=nullptr;
    bool context_enqueue_reported=false;
    bool context_overlap=[] {
        const auto* option=std::getenv("TYLOINFER_DFLASH2_CONTEXT_OVERLAP");
        return !option || option[0]!='0';
    }();
    bool combining_context=false;
    MfqCudaStream context_update_stream;
    std::shared_ptr<void> context_branch_pool;
    std::shared_ptr<Event> context_fork,context_join;
    std::vector<std::shared_ptr<Event>> context_layer_ready;
    std::vector<cudaGraphNode_t> context_layer_nodes;
    std::vector<bool> context_join_reported;
    void record_context_layer(int layer) {
        if(!combining_context)return;
        const auto stream=mfq_current_cuda_stream();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(context_layer_ready[layer]->get(),stream));
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            cudaStreamCaptureStatus capture;unsigned long long id=0;cudaGraph_t graph=nullptr;
            const cudaGraphNode_t* nodes=nullptr;std::size_t count=0;
#if CUDART_VERSION >= 13000
            const auto status=cudaStreamGetCaptureInfo(stream,&capture,&id,&graph,&nodes,nullptr,&count);
#else
            const auto status=cudaStreamGetCaptureInfo_v2(stream,&capture,&id,&graph,&nodes,&count);
#endif
            if(status==cudaSuccess && capture==cudaStreamCaptureStatusActive && count==1)
                context_layer_nodes[layer]=nodes[0];
        }
    }
    void wait_context_layer(int layer) {
        if(!combining_context)return;
        const auto stream=mfq_current_cuda_stream();
        TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,context_layer_ready[layer]->get(),0));
        if(context_join_reported[layer])return;
        const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");
        if(!trace || trace[0]!='1')return;
        cudaStreamCaptureStatus capture;unsigned long long id=0;cudaGraph_t graph=nullptr;
        const cudaGraphNode_t* nodes=nullptr;std::size_t count=0;
#if CUDART_VERSION >= 13000
        const auto status=cudaStreamGetCaptureInfo(stream,&capture,&id,&graph,&nodes,nullptr,&count);
#else
        const auto status=cudaStreamGetCaptureInfo_v2(stream,&capture,&id,&graph,&nodes,&count);
#endif
        if(status==cudaSuccess && capture==cudaStreamCaptureStatusActive &&
            context_layer_nodes[layer] && std::find(nodes,nodes+count,context_layer_nodes[layer])!=nodes+count) {
            context_join_reported[layer]=true;
            std::cerr<<"native_dflash2_context_join actual_match=1 same_graph=1 layer="<<layer
                <<" layers="<<c.layers<<" dependencies="<<count<<" capture_id="<<id<<'\n';
        }
    }
    void flush_pending_context() const {
        if(auto* stage=std::exchange(pending_context,nullptr))stage->flush();
    }
    void order_context_access() const {
        flush_pending_context();
        const auto stream=mfq_current_cuda_stream();
        if(context_fence && stream!=context_stream)
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,context_fence->get(),0));
    }
    void finish_context_access() const {
        if(!context_fence)context_fence=std::make_shared<Event>();
        context_stream=mfq_current_cuda_stream();context_fence->record(context_stream);
    }
    std::vector<Layer> layers;
    std::vector<Tensor> keys, values;
    Tensor selector_hidden, compact;
    std::unique_ptr<DecodeWindow> draft_graph;
    struct ContextGraph {
        Tensor hidden,positions,block_hidden;
        MfqCudaStream stream;
        std::unique_ptr<DecodeWindow> window,round;
        std::function<void()> flush;
        std::uint64_t round_replays=0;
    };
    std::map<int64_t,std::unique_ptr<ContextGraph>> context_graphs;
    mutable ContextGraph* pending_context=nullptr;
    std::shared_ptr<Context> draft_graph_context;
    cudaStream_t draft_graph_stream=nullptr;
    bool draft_graph_owns_pool=false;
    Tensor draft_ids,draft_positions,draft_hidden;
    int64_t position = 0;
    int selector_row = 0;
    std::filesystem::path parity_directory=[] {
        const auto* setting=std::getenv("TYLOINFER_DFLASH2_PARITY_DIRECTORY");
        return setting?std::filesystem::path(setting):std::filesystem::path{};
    }();
    int64_t parity_minimum_position=[] {
        const auto* setting=std::getenv("TYLOINFER_DFLASH2_PARITY_MIN_POSITION");
        return setting?std::stoll(setting):int64_t(0);
    }();
    Tensor parity_taps;
    int64_t parity_seen=0;
    bool parity_done=false;
    Tensor stage_clock;
    std::vector<std::string> stage_labels;
    std::map<std::string,double> stage_totals;
    uint64_t stage_samples=0;
    void stage_mark(int index,const char* label) {
        if(!stage_clock.defined())return;
        if(stage_labels.size()<=std::size_t(index))stage_labels.resize(index+1);
        stage_labels[index]=label;
        record_gpu_timestamp(reinterpret_cast<uint64_t*>(stage_clock.data_ptr<int64_t>()),index,mfq_current_cuda_stream());
    }
    void stage_report() {
        if(!stage_clock.defined())return;
        const auto clock=stage_clock.cpu();const auto* ticks=reinterpret_cast<const uint64_t*>(clock.data_ptr<int64_t>());
        for(std::size_t i=1;i<stage_labels.size();++i)
            if(!ticks[i-1] || ticks[i]<ticks[i-1])throw std::runtime_error("DFlash2 stage clock is not monotonic");
        for(std::size_t i=1;i<stage_labels.size();++i)stage_totals[stage_labels[i]]+=(ticks[i]-ticks[i-1])*1e-3;
        ++stage_samples;
        if(stage_samples==1 || stage_samples%16==0)for(const auto& [label,total]:stage_totals)
            std::cerr<<"runtime_dflash2_stage name="<<label<<" mean_us="<<total/stage_samples<<" samples="<<stage_samples<<'\n';
    }
    explicit Draft(CudaExecutionContext& e, Loader& loader, Config config) : execution(e), c(std::move(config)) {
        const auto* cache_setting=std::getenv("TYLOINFER_DFLASH2_KV_BF16");
        compact_kv=cache_setting && cache_setting[0]=='1';
        const auto* successor_setting=std::getenv("TYLOINFER_DFLASH2_STAGE_SUCCESSOR");
        stage_successor=successor_setting && successor_setting[0]=='1';
        context_projection = loader.linear("fc", c.hidden, c.hidden * c.target_layers.size());
        selector_projection = loader.linear("candidate_selector.hidden_projection", c.rank, c.hidden);
        hidden_norm = loader.dense("hidden_norm.weight", {c.hidden});
        final_norm = loader.dense("norm.weight", {c.hidden});
        codebook_device=loader.device;
        predecessor=loader.dense("candidate_selector.predecessor_codebook",{c.vocabulary,c.rank},true);
        if(stage_successor) {
            const WeightAllocationScope weights;
            successor=loader.host("candidate_selector.successor_codebook",{c.vocabulary,c.rank}).to(tb::kBFloat16);
        } else successor=loader.dense("candidate_selector.successor_codebook",{c.vocabulary,c.rank},true);
        predecessor_address=static_cast<const std::uint16_t*>(predecessor.data_ptr());
        successor_address=stage_successor?nullptr:static_cast<const std::uint16_t*>(successor.data_ptr());
        if (loader.source.find_tensor("mask_embedding")) mask_embedding = loader.dense("mask_embedding", {c.hidden});
        for (int index = 0; index < c.layers; ++index) layers.emplace_back(loader, c, index);
        if(const char* profile=std::getenv("TYLOINFER_PROFILE_DFLASH2_STAGES");profile && profile[0]=='1') {
            const ResidentAllocationScope resident;
            stage_clock=tb::zeros({7+12*c.layers},hidden_norm.options().dtype(tb::kInt64));
        }
        if(context_overlap) {
            for(int layer=0;layer<c.layers;++layer)context_layer_ready.push_back(std::make_shared<Event>());
            context_layer_nodes.resize(c.layers);context_join_reported.resize(c.layers);
        }
        reset();
    }
    void prepare_codebooks() {
        if(codebooks_prepared)return;
        codebooks_prepared=true;
        // Candidate-row staging needs neither a whole-table host mapping nor
        // additional pinned capacity ahead of mandatory KV allocations.
        if(stage_successor) {
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1')
                std::cerr<<"runtime_dflash2_codebooks actual_match=1 stage=successor_rows ram_bytes="<<successor.nbytes()
                    <<" vram_bytes="<<predecessor.nbytes()<<'\n';
            return;
        }
        const auto* setting=std::getenv("TYLOINFER_DFLASH2_CODEBOOK_RAM");
        if(!setting || setting[0]!='1')return;
        const MfqCudaGuard guard(codebook_device);
        int capable=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&capable,cudaDevAttrCanMapHostMemory,codebook_device));
        // Optional host weights must not precede mandatory KV and transfer
        // buffers during startup. Migrate only at a prepared request boundary.
        if(capable) {
            const auto migrate=[&](Tensor& values,const std::uint16_t*& address) {
                try {
                    auto pinned=tb::empty(values.sizes(),values.options().device(tb::kCPU).pinned_memory(true));
                    void* alias=nullptr;const auto status=cudaHostGetDevicePointer(&alias,pinned.data_ptr(),0);
                    if(status!=cudaSuccess){(void)cudaGetLastError();return;}
                    pinned.copy_(values);
                    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
                    values=std::move(pinned);address=static_cast<const std::uint16_t*>(alias);
                }catch(const MfqResourceExhausted&) {
                    // Retain the existing GPU image when optional pinning is
                    // unavailable; do not retry it at every draft window.
                }
            };
            migrate(predecessor,predecessor_address);migrate(successor,successor_address);
        }
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            const auto ram=(predecessor.is_cuda()?0:predecessor.nbytes())+(successor.is_cuda()?0:successor.nbytes());
            std::cerr<<"runtime_dflash2_codebooks actual_match=1 stage=prepared ram_bytes="<<ram
                <<" vram_bytes="<<predecessor.nbytes()+successor.nbytes()-ram<<'\n';
        }
    }
    void release_draft_graph() {
        if(!draft_graph_context)return;
        const MfqCudaGuard guard(draft_graph_context->device());
        context_graphs.clear();draft_graph.reset();draft_hidden={};draft_ids={};draft_positions={};
        context_branch_pool.reset();
        if(draft_graph_owns_pool)draft_graph_context->end_graph_pool(draft_graph_stream);
        draft_graph_context.reset();draft_graph_stream=nullptr;draft_graph_owns_pool=false;
    }
    void ensure_graph_pool(const std::shared_ptr<Context>& context,cudaStream_t stream) {
        if(draft_graph_context && draft_graph_stream!=stream)release_draft_graph();
        if(draft_graph_context)return;
        draft_graph_owns_pool=!context->has_graph_pool(stream);
        if(draft_graph_owns_pool)context->begin_graph_pool(stream);
        draft_graph_context=context;draft_graph_stream=stream;
    }
    bool graphs_enabled(const std::shared_ptr<Context>& context) const {
        const auto* setting=std::getenv("TYLOINFER_DFLASH2_GRAPH");
        return mfq_cuda_graph_capture_supported() && context->supports_async_allocations() &&
            (!setting || setting[0]!='0');
    }
    bool reserve_graph_workspace(std::size_t bytes) {
        if(!execution.gguf_expert_cache)return false;
        const auto context=default_context(hidden_norm.get_device());
        if(context->available_memory()>=bytes)return false;
        const auto before=gguf_expert_cache_budget_bytes(execution.gguf_expert_cache);
        reserve_gguf_expert_workspace(execution.gguf_expert_cache,bytes);
        return gguf_expert_cache_budget_bytes(execution.gguf_expert_cache)<before;
    }
    void finish_graph_preparation(bool returned) {
        if(decode_workspace_changed)decode_workspace_changed();
        else if(returned)restore_gguf_decode_workspace_after_capture(execution.gguf_expert_cache);
    }
    std::optional<std::size_t> extra_decode_workspace_bytes(int maximum_depth) const override {
        bool prepared=draft_graph && draft_graph->valid();
        for(const auto& [rows,stage]:context_graphs)prepared|=stage->round && stage->round->valid();
        if(!prepared || !draft_graph_context || !codebooks_prepared)return {};
        const auto depth=std::size_t(std::clamp(maximum_depth,1,maximum_draft_depth()));
        const auto features=std::size_t(std::min(c.window,kQ8TileRows))*c.hidden*c.target_layers.size();
        // Short context inputs/positions and selector outputs remain outside
        // the draft body. Vocabulary logits are covered by the target budget;
        // committed KV/codebooks/captured scratch are already resident.
        const auto context=features*sizeof(float)*2+std::size_t(kQ8TileRows)*sizeof(int64_t);
        const auto selector=depth*c.candidates*(sizeof(int64_t)+2*sizeof(float)+
            (stage_successor?std::size_t(c.rank)*sizeof(std::uint16_t):0));
        return context+selector;
    }
    ~Draft() override {
        pending_context=nullptr;
        if(context_fence)try {context_fence->synchronize();}catch(...) {}
        // Mapped host weights outlive all queued selector reads, including a
        // consumer that releases its last draft output without a host wait.
        if(predecessor.defined() && (!predecessor.is_cuda() || !successor.is_cuda()))
            try {const MfqCudaGuard guard(codebook_device);(void)cudaDeviceSynchronize();}catch(...) {}
        try {release_draft_graph();}catch(...) {}
    }
    void reset(int64_t batch = 1) override {
        if (batch != 1) throw std::invalid_argument("DFlash2 draft context requires a single sequence");
        pending_context=nullptr;
        order_context_access();
        const ResidentAllocationScope resident;
        auto options = hidden_norm.options();
        if (keys.empty()) for (int i = 0; i < c.layers; ++i) {
            keys.push_back(tb::zeros({c.window,c.kv_heads,c.dimension},options.dtype(compact_kv?tb::kBFloat16:tb::kFloat32)));
            values.push_back(tb::zeros({c.window,c.kv_heads,c.dimension},options.dtype(compact_kv?tb::kBFloat16:tb::kFloat32)));
        } else { for (auto& key : keys) key.zero_(); for (auto& value : values) value.zero_(); }
        position = 0; selector_hidden = {}; compact = {}; selector_row = 0;
        parity_seen=0;
        finish_context_access();
    }
    Tensor forward(const MtpTarget&, Tensor, Tensor) override { throw std::logic_error("DFlash2 uses block drafting"); }
    int64_t cache_position() const noexcept override { return position; }
    void trim_cache_to(int64_t value) override {
        if (value != position) throw std::invalid_argument("DFlash2 stores only committed target context");
    }
    bool teacher_forced_prompt_prime() const noexcept override { return false; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return true; }
    bool blockwise_drafting() const noexcept override { return true; }
    int preferred_context_update_batch_size() const noexcept override {return kQ8TileRows;}
    bool retains_partial_target_prefix() const noexcept override { return true; }
    bool supports_session_state() const noexcept override { return true; }
    MtpSessionState capture_session_state(int64_t target_position, const Tensor& hidden) const override {
        if (target_position != position || position < 1 || hidden.dim() != 3 || hidden.size(0) != 1 ||
            hidden.size(1) != 1 || hidden.size(2) != c.hidden * int64_t(c.target_layers.size()))
            throw std::invalid_argument("DFlash2 session is outside the committed context boundary");
        order_context_access();
        MtpSessionState state;
        state.cache_pos = position; state.last_target_hidden = hidden.clone();
        state.bytes = state.last_target_hidden.nbytes();
        for (int layer = 0; layer < c.layers; ++layer) {
            Qwen4BlockSessionState saved;
            saved.keys = keys[layer].clone(); saved.values = values[layer].clone();
            state.bytes += saved.keys.nbytes() + saved.values.nbytes();
            state.qwen4_blocks.push_back(std::move(saved));
            state.qwen4_positions.emplace_back(); state.qwen4_lengths.push_back(position);
        }
        finish_context_access();return state;
    }
    void restore_session_state(const MtpSessionState& state) override {
        const std::vector<int64_t> shape{c.window, c.kv_heads, c.dimension};
        if (state.cache_pos < 1 || state.qwen4_blocks.size() != std::size_t(c.layers) ||
            state.qwen4_lengths.size() != std::size_t(c.layers) || !state.blocks.empty() ||
            !state.last_target_hidden.defined() || state.last_target_hidden.sizes().vec() !=
                std::vector<int64_t>{1, 1, c.hidden * int64_t(c.target_layers.size())})
            throw std::invalid_argument("DFlash2 session geometry differs from the draft model");
        for (int layer = 0; layer < c.layers; ++layer) {
            const auto& saved = state.qwen4_blocks[layer];
            if (!saved.keys.defined() || !saved.values.defined() || saved.keys.sizes().vec() != shape ||
                saved.values.sizes().vec() != shape ||
                (saved.keys.scalar_type()!=tb::kBFloat16 && saved.keys.scalar_type()!=tb::kFloat32) ||
                (saved.values.scalar_type()!=tb::kBFloat16 && saved.values.scalar_type()!=tb::kFloat32) || state.qwen4_lengths[layer] != state.cache_pos)
                throw std::invalid_argument("DFlash2 session has incompatible KV rings");
        }
        reset(1);
        for (int layer = 0; layer < c.layers; ++layer) {
            keys[layer].copy_(state.qwen4_blocks[layer].keys);
            values[layer].copy_(state.qwen4_blocks[layer].values);
        }
        position = state.cache_pos;
        finish_context_access();
    }
    int maximum_draft_depth() const noexcept override { return std::min(c.block - 1, mfq::engine::mtp::kMaximumBlockwiseDraftDepth); }
    void seed_target_context(Tensor hidden, int64_t count) override {
        if (position || hidden.dim() != 3 || hidden.size(1) < std::min<int64_t>(count, c.window) || count < hidden.size(1))
            throw std::invalid_argument("DFlash2 prompt context is incomplete");
        prepare_codebooks();
        position = count - hidden.size(1); append_target_context(std::move(hidden), position);
    }
    void append_context_rows(const Tensor& features,const Tensor& pos) {
        // Prompt seeding and later updates share the same Q8 arithmetic.
        // The batched kernel keeps eight-row tiles and handles the tail on
        // device, without changing quantization when prompt chunking changes.
        if(pos.numel()>kQ8TileRows) {
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true))std::cerr<<"runtime_dflash2_context_seed actual_match=1 rows="<<pos.numel()
                    <<" tile_rows="<<kQ8TileRows<<" q8_fp32_scales=1 bf16_projection=1 batched=1\n";
            }
        }
        // Preserve raw F32 target taps through their first Q8 projection.
        auto prepared=normalized_q8(context_projection.forward(features),hidden_norm,c.epsilon);
        const auto table=rotation_table(pos,c);
        for(int layer=0;layer<c.layers;++layer) {
            auto kv=TiledLinear::group({&layers[layer].k,&layers[layer].v},prepared.values,&prepared);
            const auto stream=mfq_get_current_cuda_stream();
            for(const auto& tensor:{kv[0],kv[1],layers[layer].k_norm,table,pos,keys[layer],values[layer]})
                mfq_cuda_record_stream(tensor,stream);
            store_context(kv[0].data_ptr<float>(),kv[1].data_ptr<float>(),layers[layer].k_norm.data_ptr<float>(),
                table.data_ptr<float>(),pos.data_ptr<int64_t>(),keys[layer].data_ptr(),values[layer].data_ptr(),
                int(pos.numel()),c.kv_heads,c.dimension,c.window,c.epsilon,stream.stream(),compact_kv);
            record_context_layer(layer);
        }
    }
    void execute_context_stage(ContextGraph& stage) {
        const auto context=default_context(hidden_norm.get_device());
        if(!graphs_enabled(context)) {append_context_rows(stage.hidden,stage.positions);return;}
        const bool preparing=!stage.window || !stage.window->valid();
        const auto capture_bytes=(std::size_t(64)<<20)+stage.hidden.numel()*sizeof(float)*32;
        const bool returned=preparing && reserve_graph_workspace(capture_bytes);
        if(!stage.window)stage.window=std::make_unique<DecodeWindow>(stage.stream.stream());
        if(!stage.window->valid()) {
            stage.window->capture([&]{append_context_rows(stage.hidden,stage.positions);},[]{},[]{});
            std::cerr<<"runtime_dflash2_context_graph rows="<<stage.positions.numel()<<" nodes="<<stage.window->nodes()<<'\n';
        }
        if(preparing)finish_graph_preparation(returned);
        const bool queued=stage.window->enqueue_if_device_only();if(!queued)stage.window->run();
        if(!context_enqueue_reported) {
            context_enqueue_reported=true;
            std::cerr<<"runtime_dflash2_context_enqueue asynchronous="<<int(queued)<<'\n';
        }
    }
    Tensor combined_context_block(const MtpTarget& target,ContextGraph& stage) {
        const auto parent=mfq_get_current_cuda_stream();
        if(!context_update_stream) {
            context_update_stream=mfq_get_stream_from_pool(false,parent.device_index());
            context_fork=std::make_shared<Event>();context_join=std::make_shared<Event>();
            for(const auto& tensor:{context_projection.tiles,hidden_norm})mfq_cuda_record_stream(tensor,context_update_stream);
            for(int layer=0;layer<c.layers;++layer)
                for(const auto& tensor:{layers[layer].k.tiles,layers[layer].v.tiles,layers[layer].k_norm,keys[layer],values[layer]})
                    mfq_cuda_record_stream(tensor,context_update_stream);
        }
        auto* window=DecodeWindow::recording();
        if(!window)throw std::logic_error("DFlash2 combined block requires its recording window");
        window->prepare_branch_stream(context_update_stream,context_branch_pool);
        mfq_cuda_record_stream(stage.hidden,context_update_stream);mfq_cuda_record_stream(stage.positions,context_update_stream);
        combining_context=true;
        try {
            context_fork->record(parent.stream());
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(context_update_stream.stream(),context_fork->get(),0));
            {
                const MfqCudaStreamGuard selected(context_update_stream);
                append_context_rows(stage.hidden,stage.positions);
                context_join->record(context_update_stream.stream());
            }
            auto hidden=block_hidden(target,draft_ids,draft_positions);
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(parent.stream(),context_join->get(),0));
            combining_context=false;return hidden;
        }catch(...) {combining_context=false;throw;}
    }
    void append_target_context(Tensor hidden,int64_t start) override {
        append_context(std::move(hidden),start,false);
    }
    void append_target_context_for_draft(Tensor hidden,int64_t start) override {
        append_context(std::move(hidden),start,true);
    }
    void append_context(Tensor hidden,int64_t start,bool for_draft) {
        if (start != position || hidden.dim() != 3 || hidden.size(0) != 1 || hidden.size(1) < 1 ||
            hidden.size(2) != c.hidden * int64_t(c.target_layers.size())) throw std::invalid_argument("DFlash2 target features differ");
        order_context_access();
        const auto count = hidden.size(1), retained = std::min<int64_t>(count, c.window), offset = count - retained;
        if(!parity_directory.empty() && !parity_done) {
            if(!parity_taps.defined())parity_taps=tb::empty({c.window,hidden.size(2)},hidden.options().dtype(tb::kFloat32));
            // Keep diagnostic features on device until the actual draft has
            // executed. A host read here would serialize context updates and
            // could hide an ordering error in the graph under examination.
            const auto first=(start+offset)%c.window;
            const auto span=std::min<int64_t>(retained,c.window-first);
            parity_taps.narrow(0,first,span).copy_(hidden.narrow(1,offset,span).reshape({span,hidden.size(2)}));
            if(span<retained)parity_taps.narrow(0,0,retained-span).copy_(
                hidden.narrow(1,offset+span,retained-span).reshape({retained-span,hidden.size(2)}));
            parity_seen=std::min<int64_t>(c.window,parity_seen+retained);
        }
        const auto stream=mfq_current_cuda_stream();
        const auto context=default_context(hidden_norm.get_device());
        auto positions=tb::arange(start+offset,start+count,hidden_norm.options().dtype(tb::kInt64));
        auto input=hidden.narrow(1,offset,retained).to(hidden_norm.device());
        // The block GEMM's eight activation lanes cover ordinary/MTP commits.
        // Keep large prompt seeding transient instead of retaining PP workspaces
        // in a persistent graph for each possible prompt length.
        if(retained<=8 && graphs_enabled(context)) {
            ensure_graph_pool(context,stream);
            auto& stage=context_graphs[retained];
            if(!stage) {
                const ResidentAllocationScope resident;
                stage=std::make_unique<ContextGraph>();
                stage->hidden=tb::empty(input.sizes(),input.options().dtype(tb::kFloat32));
                stage->positions=tb::empty(positions.sizes(),positions.options());
                stage->stream=mfq_get_current_cuda_stream();
                stage->flush=[this,ptr=stage.get()] {
                    const MfqCudaStreamGuard selected(ptr->stream);
                    execute_context_stage(*ptr);finish_context_access();
                };
            }
            stage->hidden.copy_(input);stage->positions.copy_(positions);
            if(context_overlap && for_draft)pending_context=stage.get();
            else execute_context_stage(*stage);
        }else append_context_rows(input,positions);
        position += count;
        finish_context_access();
    }
    Tensor block_hidden(const MtpTarget& target,const Tensor& ids,const Tensor& positions) {
        stage_mark(0,"block.begin");
        auto hidden = target.embed(ids).to(tb::kBFloat16).to(tb::kFloat32);
        if (mask_embedding.defined()) hidden.narrow(1, 1, c.block - 1).copy_(mask_embedding.reshape({1,1,c.hidden}).expand({1,c.block-1,c.hidden}));
        stage_mark(1,"block.embedding");
        const auto table=rotation_table(positions,c);
        stage_mark(2,"block.rotary_table");
        const int chunks = (c.window - 1 + c.block + 127) / 128;
        auto partial = tb::empty({chunks, c.block, c.heads, c.dimension + 2}, hidden.options());
        for (int i = 0; i < c.layers; ++i) {
            const auto& layer = layers[i];
            const int phase=3+12*i;
            auto prepared=normalized_q8(hidden,layer.input_norm,c.epsilon);auto input=prepared.values;stage_mark(phase,"block.norm");
            auto dynamic = layer.dynamic_attention.forward(input,&prepared);stage_mark(phase+1,"block.dynamic_attention");
            auto conv = convolved_q8(input,dynamic,layer.attention_base,c.group);stage_mark(phase+2,"block.conv_attention");
            auto qkv=TiledLinear::group({&layer.q,&layer.k,&layer.v},conv.values,&conv);stage_mark(phase+3,"block.qkv");
            auto query = rotated(qkv[0], layer.q_norm, table, c.heads, c);
            auto key = rotated(qkv[1], layer.k_norm, table, c.kv_heads, c);
            auto value = qkv[2];stage_mark(phase+4,"block.rope");
            auto attended = tb::empty({1, c.block, c.heads * c.dimension}, hidden.options());
            wait_context_layer(i);
            attention(query.data_ptr<float>(),keys[i].data_ptr(),values[i].data_ptr(), key.data_ptr<float>(), value.data_ptr<float>(),
                partial.data_ptr<float>(), attended.data_ptr<float>(), c.block, c.heads, c.kv_heads, c.dimension, c.window,
                0,mfq_current_cuda_stream(),positions.data_ptr<int64_t>(),compact_kv);
            stage_mark(phase+5,"block.attention");
            auto output = layer.output.forward(attended);stage_mark(phase+6,"block.output");
            auto residual = convolved(output, dynamic, layer.attention_base, hidden, c.group, true);
            prepared=normalized_q8(residual,layer.post_norm,c.epsilon);input=prepared.values;stage_mark(phase+7,"block.residual_norm");
            dynamic = layer.dynamic_mlp.forward(input,&prepared);
            conv = convolved_q8(input,dynamic,layer.mlp_base,c.group);stage_mark(phase+8,"block.mlp_input");
            auto gu=TiledLinear::group({&layer.gate,&layer.up},conv.values,&conv);stage_mark(phase+9,"block.gate_up");
            auto shape=conv.values.sizes().vec();shape.back()=c.intermediate;
            auto activated=prepared_storage(tb::empty(shape,conv.values.options()));
            if(activated.codes.defined())
                swiglu_float_q8(gu[0].data_ptr<float>(),gu[1].data_ptr<float>(),activated.values.data_ptr<float>(),
                    activated.codes.data_ptr(),activated.scales.data_ptr<float>(),c.block,c.intermediate,mfq_current_cuda_stream());
            else swiglu_float(gu[0].data_ptr<float>(),gu[1].data_ptr<float>(),activated.values.data_ptr<float>(),activated.values.numel(),mfq_current_cuda_stream());
            output = layer.down.forward(activated.values,&activated);stage_mark(phase+10,"block.down");
            hidden = convolved(output, dynamic, layer.mlp_base, residual, c.group, true);stage_mark(phase+11,"block.residual");
        }
        return hidden;
    }
    void compare_parity(const MtpTarget& target,const Tensor& anchor,const Tensor& body,int requested) {
        if(parity_directory.empty() || parity_done || position<parity_minimum_position || parity_seen<c.window)return;
        parity_done=true;
        std::filesystem::create_directories(parity_directory);
        const auto saved=body.clone();
        const auto actual=saved.cpu();
        auto ids=tb::full({1,c.block},c.mask,anchor.options());ids.narrow(1,0,1).copy_(anchor);
        auto positions=tb::arange(position,position+c.block,anchor.options());
        const auto reference=block_hidden(target,ids,positions);
        const auto expected=reference.cpu();
        double peak=0,error=0,magnitude=0;
        for(int64_t i=0;i<actual.numel();++i) {
            const double a=actual.data_ptr<float>()[i],b=expected.data_ptr<float>()[i],delta=a-b;
            peak=std::max(peak,std::abs(delta));error+=delta*delta;magnitude+=b*b;
        }
        const auto actual_logits=target.logits(normalized(saved.narrow(1,1,requested),final_norm,c.epsilon)).to(tb::kFloat32).contiguous().cpu();
        const auto expected_logits=target.logits(normalized(reference.narrow(1,1,requested),final_norm,c.epsilon)).to(tb::kFloat32).contiguous().cpu();
        nlohmann::json report{{"position",position},{"requested",requested},
            {"body_bitwise",actual.nbytes()==expected.nbytes() && std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0},
            {"body_max_error",peak},{"body_relative_l2",std::sqrt(error/std::max(1e-30,magnitude))},
            {"logits_bitwise",std::memcmp(actual_logits.data_ptr(),expected_logits.data_ptr(),actual_logits.nbytes())==0}};
        std::vector<int64_t> actual_ids,expected_ids;
        for(int row=0;row<requested;++row) {
            const auto* a=actual_logits.data_ptr<float>()+int64_t(row)*c.vocabulary;
            const auto* b=expected_logits.data_ptr<float>()+int64_t(row)*c.vocabulary;
            actual_ids.push_back(std::max_element(a,a+c.vocabulary)-a);
            expected_ids.push_back(std::max_element(b,b+c.vocabulary)-b);
        }
        report["graph_argmax"]=actual_ids;report["eager_argmax"]=expected_ids;
        std::ofstream(parity_directory/"graph-eager.json")<<report.dump(2)<<'\n';
        const auto write=[&](const char* name,const Tensor& source) {
            const auto host=source.contiguous().cpu();
            std::ofstream file(parity_directory/name,std::ios::binary);
            file.write(static_cast<const char*>(host.data_ptr()),host.nbytes());
            if(!file)throw std::runtime_error("could not write DFlash2 parity tensor");
        };
        write("body.f32",saved);write("logits.f32",actual_logits);
        const auto normalized_rows=normalized(saved,final_norm,c.epsilon).contiguous().cpu();
        // Reuse Strata's existing parity case format. The fourth header word
        // records the absolute context start; zero preserves older cases.
        const int32_t header[]={c.window,int32_t(anchor.item<int64_t>()),1024,int32_t(position-c.window)};
        std::ofstream file(parity_directory/"case.bin",std::ios::binary);
        file.write(reinterpret_cast<const char*>(header),sizeof(header));
        const auto first=(position-c.window)%c.window;
        const auto taps=tb::cat({parity_taps.narrow(0,first,c.window-first),parity_taps.narrow(0,0,first)},0).contiguous().cpu();
        file.write(static_cast<const char*>(taps.data_ptr()),taps.nbytes());
        auto pair_ids=tb::full({1,2},c.mask,anchor.options());pair_ids.narrow(1,0,1).copy_(anchor);
        auto embeddings=target.embed(pair_ids).to(tb::kBFloat16).to(tb::kFloat32);
        if(mask_embedding.defined())embeddings.narrow(1,1,1).copy_(mask_embedding.reshape({1,1,c.hidden}));
        const auto embedded=embeddings.contiguous().cpu();file.write(static_cast<const char*>(embedded.data_ptr()),embedded.nbytes());
        std::vector<int32_t> vocabulary(1024);for(int i=0;i<1024;++i)vocabulary[i]=i;
        file.write(reinterpret_cast<const char*>(vocabulary.data()),vocabulary.size()*sizeof(int32_t));
        std::vector<char> zero_head(int64_t(1024)*(c.hidden/32)*34);
        file.write(zero_head.data(),zero_head.size());
        std::vector<char> placeholder((c.block-1)*(sizeof(int32_t)+sizeof(float)));
        file.write(placeholder.data(),placeholder.size());
        file.write(static_cast<const char*>(normalized_rows.data_ptr()),normalized_rows.nbytes());
        if(!file)throw std::runtime_error("could not write DFlash2 parity case");
        std::cerr<<"runtime_dflash2_graph_eager_parity "<<report.dump()<<'\n';
    }
    Tensor draft_block(const MtpTarget& target, Tensor anchor, int requested) override {
        if (!position || requested < 1 || requested > maximum_draft_depth()) throw std::invalid_argument("DFlash2 draft depth differs");
        prepare_codebooks();
        const auto stream=mfq_current_cuda_stream();
        const auto context=default_context(anchor.get_device());
        const bool use_graph=graphs_enabled(context);
        if(pending_context && (!use_graph || !context_overlap || pending_context->stream.stream()!=stream))order_context_access();
        auto* update=context_overlap && use_graph?std::exchange(pending_context,nullptr):nullptr;
        if(!update)order_context_access();
        Tensor hidden;
        if(use_graph) {
            ensure_graph_pool(context,stream);
            if(!draft_ids.defined()) {
                const ResidentAllocationScope resident;
                draft_ids=tb::full({1,c.block},c.mask,anchor.options());
                draft_positions=tb::empty({c.block},anchor.options());
            }
            draft_ids.narrow(1,0,1).copy_(anchor);
            draft_positions.copy_(tb::arange(position,position+c.block,anchor.options()));
            if(update) {
                try {
                    const bool preparing=!update->round || !update->round->valid();
                    const auto extra=(std::size_t(64)<<20)+update->hidden.numel()*sizeof(float)*32;
                    const bool returned=preparing && reserve_graph_workspace(runtime_options::workspace_reserve_bytes()+extra);
                    if(!update->round)update->round=std::make_unique<DecodeWindow>(stream);
                    if(!update->round->valid()) {
                        update->round->capture([&]{update->block_hidden=combined_context_block(target,*update);},
                            []{},[&]{update->block_hidden={};});
                        std::cerr<<"runtime_dflash2_graph enabled=1 rows="<<c.block<<" nodes="<<update->round->nodes()
                            <<" combined_context=1 commit_rows="<<update->positions.numel()<<'\n';
                    }
                    if(preparing)finish_graph_preparation(returned);
                    const bool queued=update->round->enqueue_if_device_only();if(!queued)update->round->run();
                    ++update->round_replays;
                    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1' && update->round_replays%16==0)
                        std::cerr<<"runtime_dflash2_context_round_replay same_graph=1 rows="<<update->positions.numel()
                            <<" calls="<<update->round_replays<<" asynchronous="<<int(queued)<<'\n';
                    hidden=update->block_hidden;
                }catch(...) {pending_context=update;throw;}
            } else {
                const bool preparing=!draft_graph || !draft_graph->valid();
                const bool returned=preparing && reserve_graph_workspace(runtime_options::workspace_reserve_bytes());
                if(!draft_graph)draft_graph=std::make_unique<DecodeWindow>(stream);
                if(!draft_graph->valid()) {
                    draft_graph->capture([&] {draft_hidden=block_hidden(target,draft_ids,draft_positions);},
                        [] {},[&] {draft_hidden={};});
                    std::cerr<<"runtime_dflash2_graph enabled=1 rows="<<c.block<<" nodes="<<draft_graph->nodes()<<'\n';
                }
                if(preparing)finish_graph_preparation(returned);
                if(!draft_graph->enqueue_if_device_only())draft_graph->run();
                hidden=draft_hidden;
            }
        } else {
            auto ids=tb::full({1,c.block},c.mask,anchor.options());ids.narrow(1,0,1).copy_(anchor);
            auto positions=tb::arange(position,position+c.block,anchor.options());
            hidden=block_hidden(target,ids,positions);
        }
        compare_parity(target,anchor,hidden,requested);
        hidden = normalized(hidden.narrow(1, 1, requested), final_norm, c.epsilon);
        const int phase=3+12*c.layers;stage_mark(phase,"head.norm");
        auto logits = target.logits(hidden).to(tb::kFloat32).contiguous();
        stage_mark(phase+1,"head.projection");
        compact = sample_top_k_compact_cuda(logits.reshape({requested, c.vocabulary}), c.candidates);
        stage_mark(phase+2,"head.top_k");
        selector_hidden = selector_projection.forward(hidden).to(tb::kFloat32).contiguous();
        if(stage_successor) {
            // Only the requested candidates are read by the selector. Keep
            // their immutable BF16 rows; every predecessor remains available
            // on the GPU, including a caller-supplied token outside the list.
            const auto ids=compact.cpu();
            const auto rows=int64_t(requested)*c.candidates;
            if(!successor_rows_host.defined() || successor_rows_host.size(0)<rows) {
                const ResidentAllocationScope resident;
                successor_rows_host=tb::empty({rows,c.rank},successor.options());
                successor_rows_device=tb::empty({rows,c.rank},successor.options().device(predecessor.device()));
            }
            const auto* indices=ids.data_ptr<int64_t>();
            auto* destination=static_cast<std::uint16_t*>(successor_rows_host.data_ptr());
            const auto* table=static_cast<const std::uint16_t*>(successor.data_ptr());
            for(int64_t row=0;row<rows;++row) {
                const auto token=std::uint32_t(indices[row]);
                auto* output=destination+row*c.rank;
                if(token<std::uint64_t(c.vocabulary))std::memcpy(output,table+int64_t(token)*c.rank,c.rank*sizeof(std::uint16_t));
                else std::memset(output,0,c.rank*sizeof(std::uint16_t));
            }
            successor_rows_device.narrow(0,0,rows).copy_(successor_rows_host.narrow(0,0,rows));
        }
        stage_mark(phase+3,"head.selector_projection");stage_report();
        finish_context_access();selector_row = 0; return logits;
    }
    std::vector<int32_t> draft_greedy_tokens(Tensor anchor) override {
        return draft_greedy_chain(std::move(anchor),false).tokens;
    }
    MtpGreedyChain draft_greedy_chain(Tensor anchor,bool probabilities) override {
        if(!compact.defined() || selector_row || compact.size(0)<1)
            throw std::logic_error("DFlash2 greedy selector has no fresh draft block");
        const auto rows=compact.size(0);
        auto selected=tb::empty({rows},anchor.options().dtype(tb::kInt64));
        selector_greedy_walk(compact.data_ptr<int64_t>(),int(rows),c.candidates,anchor.data_ptr<int64_t>(),
            predecessor_address,stage_successor?static_cast<const std::uint16_t*>(successor_rows_device.data_ptr()):successor_address,
            selector_hidden.data_ptr<float>(),selected.data_ptr<int64_t>(),c.rank,c.vocabulary,mfq_current_cuda_stream(),stage_successor,probabilities);
        const auto host=selected.cpu();
        MtpGreedyChain result;
        result.tokens.resize(std::size_t(rows));
        if(probabilities)result.probabilities.resize(std::size_t(rows));
        for(int64_t row=0;row<rows;++row) {
            const auto packet=uint64_t(host.data_ptr<int64_t>()[row]);
            result.tokens[std::size_t(row)]=int32_t(uint32_t(packet));
            if(probabilities) {
                const uint32_t bits=uint32_t(packet>>32);
                std::memcpy(&result.probabilities[std::size_t(row)],&bits,sizeof(bits));
            }
        }
        selector_row=int(rows);
        const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");
        if(trace && trace[0]=='1') {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true))std::cerr<<"runtime_dflash2_selector_walk enabled=1 readbacks=1 rows="<<rows<<'\n';
        }
        return result;
    }
    Tensor draft_next(Tensor logits, Tensor previous) override {
        if (!compact.defined() || selector_row >= compact.size(0)) throw std::logic_error("DFlash2 selector has no draft row");
        auto output = tb::full(logits.sizes(), -std::numeric_limits<float>::infinity(), logits.options().dtype(tb::kFloat32));
        selector_logits(compact.data_ptr<int64_t>() + int64_t(selector_row) * c.candidates, c.candidates, previous.data_ptr<int64_t>(),
            predecessor_address,stage_successor?static_cast<const std::uint16_t*>(successor_rows_device.data_ptr())+
                int64_t(selector_row)*c.candidates*c.rank:successor_address,
            selector_hidden.data_ptr<float>() + int64_t(selector_row) * c.rank, output.data_ptr<float>(), c.rank, c.vocabulary,
            mfq_current_cuda_stream(),stage_successor);
        ++selector_row; return output;
    }
};
}
std::unique_ptr<MtpModule> load(CudaExecutionContext& execution, Qwen4CausalLm& target, const std::filesystem::path& directory) {
    auto config = Config::load(directory);
    if (config.hidden != target.hidden_size() || config.vocabulary != target.vocab_size() || config.target_layers.back() >= target.num_hidden_layers())
        throw std::invalid_argument("DFlash2 checkpoint is incompatible with the target");
    Loader loader(directory, execution.layer_placement.primary_device());
    auto draft = std::make_unique<Draft>(execution, loader, config);
    qwen4_set_hidden_capture_layers(target, config.target_layers);
    return draft;
}
}
