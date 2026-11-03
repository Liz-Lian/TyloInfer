#pragma once

#include "storage/weight_loader.h"

#include "core/mtp.h"
#include "core/full_block.h"
#include "models/qwen35/causal_lm.h"
#include "ops.h"
#include "runtime/decode_window.h"
#include "mfq_cuda_context.h"
#include "runtime/graph_pool_lease.h"
#include "models/qwen35/mtp_graph.h"
#include <map>
#include <tuple>
#include <iostream>

// Qwen3.5's predictor shares the main embedding/output head and owns only
// its fusion/norm/attention/FFN weights and an independent attention history.
struct Qwen35Mtp final : MtpModule {
    mfq::cuda::qwen35::Config config;
    QuantLinear fusion;
    mfq_tensor_backend::Tensor hidden_norm, embedding_norm, output_norm;
    std::vector<std::unique_ptr<Block>> blocks;
    CudaExecutionContext *execution = nullptr;
    int64_t cache_pos = 0;
    struct PredictorGraphs {
        using Tensor=mfq_tensor_backend::Tensor;
        struct Cell {
            mfq::cuda::DecodeWindow graph;
            Tensor hidden,ids,positions,physical_positions,lengths,output,logits;
            explicit Cell(cudaStream_t stream):graph(stream){}
        };
        std::shared_ptr<mfq::cuda::Context> context;
        cudaStream_t stream;
        std::shared_ptr<mfq::cuda::GraphPoolLease> pool;
        const void* target_identity=nullptr;
        std::vector<std::pair<const void*,const void*>> cache_addresses;
        std::map<std::tuple<int,int64_t,int>,std::unique_ptr<Cell>> cells;
        uint64_t captures=0,replays=0;
        PredictorGraphs():context(mfq::cuda::default_context(mfq_current_cuda_device())),
            stream(mfq_current_cuda_stream()),pool(mfq::cuda::GraphPoolLease::acquire(context,stream)) {}
        ~PredictorGraphs() {
            cells.clear();
            std::cerr<<"runtime_qwen35_predictor_graph captures="<<captures<<" replays="<<replays<<'\n';
        }
    };
    std::unique_ptr<PredictorGraphs> predictor_graphs;
    bool graph_initialized=false;
    std::shared_ptr<mfq::cuda::qwen35::MtpTargetGraphs> target_graphs;

    static std::optional<Qwen35Mtp> load_if_present(const mfq::ModelSource &file,
                                                    const mfq::cuda::qwen35::Config &main,
                                                    CudaExecutionContext &execution) {
        const bool fusion_present = has_tensor(file, "predictor.fusion.weight");
        const bool any = fusion_present || has_tensor(file, "predictor.hidden_norm.weight") ||
                         has_tensor(file, "predictor.embedding_norm.weight") ||
                         has_tensor(file, "predictor.output_norm.weight") ||
                         has_tensor(file, "predictor.block.0.attention.query.weight");
        if (!fusion_present) {
            TYLOINFER_RUNTIME_CHECK(!any, "Qwen model source contains an incomplete MTP head");
            return std::nullopt;
        }
        TYLOINFER_RUNTIME_CHECK(main.mtp_num_hidden_layers > 0 && !main.mtp_use_dedicated_embeddings,
                          "Qwen MTP requires declared predictor layers and shared embeddings");
        Qwen35Mtp result;
        result.config = main;
        result.execution = &execution;
        result.hidden_norm = load_dense_gpu(execution, file, "predictor.hidden_norm.weight");
        result.embedding_norm = load_dense_gpu(execution, file, "predictor.embedding_norm.weight");
        result.output_norm = load_dense_gpu(execution, file, "predictor.output_norm.weight");
        result.fusion = load_quant_linear(execution, file, "predictor.fusion.weight");
        TYLOINFER_RUNTIME_CHECK(result.fusion.neuron_len() == 2 * main.hidden_size &&
                              result.fusion.out() == main.hidden_size &&
                              result.hidden_norm.numel() == main.hidden_size &&
                              result.embedding_norm.numel() == main.hidden_size &&
                              result.output_norm.numel() == main.hidden_size,
                          "Qwen MTP component dimensions disagree with the backbone");
        for (int layer = 0; layer < main.mtp_num_hidden_layers; ++layer) {
            auto block = mfq::cuda::qwen35::load_block(execution, file, result.config, layer,
                                                       "full_attention", "predictor");
            block->cuda_device = execution.layer_placement.primary_device();
            if (auto* full = dynamic_cast<FullBlock*>(block.get())) full->preallocate_kv();
            result.blocks.push_back(std::move(block));
        }
        return result;
    }

    void reset(int64_t batch = 1) override {
        if(batch!=1){predictor_graphs.reset();target_graphs.reset();}
        cache_pos = 0;
        for (auto &block : blocks)
            block->reset(batch);
    }

    mfq_tensor_backend::Tensor forward(const MtpTarget &target, mfq_tensor_backend::Tensor hidden,
                                       mfq_tensor_backend::Tensor next_ids) override {
        return evaluate(target, std::move(hidden), std::move(next_ids), {});
    }

    bool take_graph_initialization() noexcept override {return std::exchange(graph_initialized,false);}
    uint64_t graph_replay_count() const noexcept override {return predictor_graphs?predictor_graphs->replays:0;}

    MtpStep step(const MtpTarget& target,mfq_tensor_backend::Tensor hidden,
                 mfq_tensor_backend::Tensor ids) override {
        return step_positioned(target,std::move(hidden),std::move(ids),{});
    }

    std::optional<MtpStep> graph_step(const MtpTarget& target,
            const mfq_tensor_backend::Tensor& hidden,const mfq_tensor_backend::Tensor& ids,
            const mfq_tensor_backend::Tensor& positions, int history_mode = 0) {
        namespace tb=mfq_tensor_backend;
        if(cache_pos<=0 || hidden.dim()!=3 || ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 ||
                ids.size(1)>1+mfq::engine::mtp::kMaximumReportedDraftDepth ||
                hidden.size(0)!=1 || hidden.size(1)!=ids.size(1) || !hidden.is_cuda() || ids.device()!=hidden.device() ||
                target.rope->sections.numel()!=0 || execution->layer_placement.enabled() ||
                execution->dense_cpu_layer_count || (positions.defined() && positions.numel()!=ids.size(1)))return {};
        std::vector<std::pair<const void*,const void*>> addresses;
        for(const auto& block:blocks) {
            auto* full=dynamic_cast<FullBlock*>(block.get());
            if(!full || full->cpu_offloaded || full->cuda_device!=hidden.get_device() ||
                    !full->cache.defined() || full->cache.is_paged() || full->cache.quantization.enabled())return {};
            addresses.emplace_back(full->cache.k.data_ptr(),full->cache.v.data_ptr());
        }
        const auto identity=std::visit([](const auto* p){return static_cast<const void*>(p);},target.model);
        if(predictor_graphs && predictor_graphs->stream!=mfq_current_cuda_stream())predictor_graphs.reset();
        if(!predictor_graphs)predictor_graphs=std::make_unique<PredictorGraphs>();
        auto& registry=*predictor_graphs;
        if(registry.target_identity!=identity || registry.cache_addresses!=addresses) {
            registry.cells.clear();registry.target_identity=identity;registry.cache_addresses=std::move(addresses);
        }
        const auto tokens=ids.size(1);
        auto& owner=registry.cells[{int(hidden.scalar_type()),tokens,history_mode}];
        if(!owner)owner=std::make_unique<PredictorGraphs::Cell>(registry.stream);
        auto& cell=*owner;const auto start=cache_pos;
        auto physical=tb::arange(start,start+tokens,ids.options());
        auto pos=positions.defined()?positions.reshape({tokens}).to(ids.device(),tb::kInt64).contiguous():physical;
        if(!cell.ids.defined()) {
            cell.hidden=hidden.contiguous().clone();cell.ids=ids.clone();cell.positions=pos.clone();
            cell.physical_positions=physical.clone();cell.lengths=tb::full({1},start+tokens,ids.options());
        } else {
            cell.hidden.copy_(hidden);cell.ids.copy_(ids);cell.positions.copy_(pos);
            cell.physical_positions.copy_(physical);cell.lengths.fill_(start+tokens);
        }
        if(!cell.graph.valid()) {
            cell.graph.capture([&] {
                if(history_mode==2 && tokens>1) {
                    (void)evaluate(target,cell.hidden.narrow(1,0,tokens-1),cell.ids.narrow(1,0,tokens-1),
                        cell.positions.narrow(0,0,tokens-1),cell.physical_positions.narrow(0,0,tokens-1),{},true);
                    cell.output=evaluate(target,cell.hidden.narrow(1,tokens-1,1),cell.ids.narrow(1,tokens-1,1),
                        cell.positions.narrow(0,tokens-1,1),cell.physical_positions.narrow(0,tokens-1,1),cell.lengths);
                } else cell.output=evaluate(target,cell.hidden,cell.ids,cell.positions,
                    cell.physical_positions,cell.lengths,history_mode==1);
                if(history_mode!=1)cell.logits=target.logits(cell.output.narrow(1,cell.output.size(1)-1,1)
                    .to(tb::kFloat16).contiguous()).reshape({1,-1});
            },[&]{cache_pos=start;},[&]{cell.output={};cell.logits={};});
            ++registry.captures;graph_initialized=true;
            std::cerr<<"runtime_qwen35_predictor_graph_capture rows="<<tokens<<" history_mode="<<history_mode
                <<" nodes="<<cell.graph.nodes()<<'\n';
        }
        cell.graph.run();++registry.replays;cache_pos=start+tokens;
        if(history_mode==1)return MtpStep{};
        auto output=cell.output.clone();
        return MtpStep{output,output,cell.logits};
    }

    bool cache_only_history_supported() const noexcept {
        // With several predictor layers, earlier outputs feed the next layer.
        return blocks.size()==1 && dynamic_cast<const FullBlock*>(blocks.front().get());
    }

    void history_step(const MtpTarget& target, mfq_tensor_backend::Tensor hidden,
                      mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor positions = {}) {
        if(!cache_only_history_supported()) {
            (void)step_positioned(target,std::move(hidden),std::move(ids),std::move(positions));return;
        }
        if(graph_step(target,hidden,ids,positions,1))return;
        (void)evaluate(target,std::move(hidden),std::move(ids),std::move(positions),{},{},true);
    }

    std::optional<MtpStep> history_tail_step(const MtpTarget& target,
            const mfq_tensor_backend::Tensor& hidden,const mfq_tensor_backend::Tensor& ids,
            const mfq_tensor_backend::Tensor& positions = {}) {
        if(!cache_only_history_supported() || ids.size(1)<=1)return {};
        return graph_step(target,hidden,ids,positions,2);
    }

    MtpStep step_positioned(const MtpTarget &target, mfq_tensor_backend::Tensor hidden,
                            mfq_tensor_backend::Tensor next_ids,
                            mfq_tensor_backend::Tensor positions) override {
        if(auto graph=graph_step(target,hidden,next_ids,positions))return std::move(*graph);
        auto output =
            evaluate(target, std::move(hidden), std::move(next_ids), std::move(positions));
        return {output, output};
    }

    mfq_tensor_backend::Tensor evaluate(const MtpTarget &target, mfq_tensor_backend::Tensor hidden,
                                        mfq_tensor_backend::Tensor next_ids,
                                        mfq_tensor_backend::Tensor positions,
                                        mfq_tensor_backend::Tensor physical_positions={},
                                        mfq_tensor_backend::Tensor dynamic_lengths={}, bool history_only=false) {
        TYLOINFER_RUNTIME_CHECK(!history_only || cache_only_history_supported(),
                              "cache-only Qwen predictor history requires one attention layer");
        struct Ops {
            using Tensor = mfq_tensor_backend::Tensor;
            Qwen35Mtp &model;
            const MtpTarget &target;
            const Tensor& physical_positions;
            const Tensor& dynamic_lengths;
            bool history_only;
            static int rank(const Tensor &x) { return x.dim(); }
            static int64_t size(const Tensor &x, int axis) { return x.size(axis); }
            static int64_t elements(const Tensor &x) { return x.numel(); }
            static bool defined(const Tensor &x) { return x.defined(); }
            Tensor embed(Tensor ids, const Tensor &like) {
                return target.embed(ids).to(like.scalar_type());
            }
            Tensor normalize(Tensor x, mfq::models::qwen35::PredictorNorm role) {
                using Norm = mfq::models::qwen35::PredictorNorm;
                if(history_only && role==Norm::output)return {};
                const auto &weight = role == Norm::embedding ? model.embedding_norm
                                     : role == Norm::hidden  ? model.hidden_norm
                                                             : model.output_norm;
                return qwen_rms_norm(x.reshape({-1, model.config.hidden_size})
                                         .to(mfq_tensor_backend::kFloat32),
                                     weight, model.config.rms_norm_eps,
                                     model.config.legacy_tensor_layout.norm_weight_offset)
                    .reshape(x.sizes());
            }
            void trace(const char *stage, const Tensor &x) {
                if(x.defined())trace_gemma_stage(*model.execution, 0, stage, x);
            }
            Tensor fuse(Tensor e, Tensor h) {
                return model.fusion.forward(*model.execution, mfq_tensor_backend::cat({e, h}, -1));
            }
            Tensor positions(Tensor explicit_pos, int64_t start, int64_t tokens) {
                return explicit_pos.defined()
                           ? tensor_to_cuda_device(
                                 model.execution->model_parallel_collectives, explicit_pos,
                                 model.execution->layer_placement.primary_device())
                                 .to(mfq_tensor_backend::kInt64)
                                 .contiguous()
                           : mfq_tensor_backend::arange(start, start + tokens,
                                                        mfq_tensor_backend::TensorOptions()
                                                            .device(mfq_tensor_backend::kCUDA)
                                                            .dtype(mfq_tensor_backend::kInt64));
            }
            Tensor lengths(int64_t batch, int64_t end, const Tensor &positions) {
                return dynamic_lengths.defined()?dynamic_lengths:
                    mfq_tensor_backend::full({batch}, end, positions.options());
            }
            Tensor cache_positions(const Tensor &positions, int64_t start, int64_t tokens) {
                return physical_positions.defined()?physical_positions:
                    mfq_tensor_backend::arange(start, start + tokens, positions.options());
            }
            Tensor layer(std::unique_ptr<Block> &block, Tensor hidden, const Tensor &pos,
                         const std::optional<Tensor> &lengths,
                         const std::optional<Tensor> &cache_positions) {
                Block::Context context;
                context.positions=pos;context.full_positions=pos;context.cache_position=model.cache_pos;
                context.sequence_lengths=dynamic_lengths.defined()?std::optional<Tensor>(dynamic_lengths):lengths;
                context.cache_positions=cache_positions;
                if(const auto* full=dynamic_cast<const FullBlock*>(block.get());full && full->cache.k.defined() &&
                        !full->cache.is_paged() && !full->cache.quantization.enabled() && full->cache.k.dim()>=3)
                    context.planned_kv_length=full->cache.k.size(2);
                if(history_only) {
                    static_cast<FullBlock*>(block.get())->append_history(*model.execution,std::move(hidden),context,*target.rope);
                    return {};
                }
                return block->forward_context(*model.execution,std::move(hidden),context,*target.rope);
            }
        } ops{*this, target,physical_positions,dynamic_lengths,history_only};
        return mfq::models::qwen35::mtp_predictor(ops, std::move(hidden), std::move(next_ids),
                                                  std::move(positions));
    }

    int64_t cache_position() const noexcept override { return cache_pos; }

    void trim_cache_to(int64_t position) override {
        TYLOINFER_RUNTIME_CHECK(position >= 0 && position <= cache_pos,
                          "Qwen MTP cache trim position is invalid");
        // Full-attention cache storage is position addressed. Lowering the
        // logical boundary makes the next predictor pass overwrite the
        // discarded draft suffix without copying history-sized tensors.
        cache_pos = position;
    }

    bool supports_session_state() const noexcept override { return true; }

    MtpSessionState
    capture_session_state(int64_t target_cache_position,
                          const mfq_tensor_backend::Tensor &last_target_hidden) const override {
        const int64_t predictor_position = target_cache_position - 1;
        if (predictor_position <= 0 || predictor_position > cache_pos ||
            !last_target_hidden.defined() || last_target_hidden.dim() != 3 ||
            last_target_hidden.size(0) != 1 || last_target_hidden.size(1) != 1 ||
            last_target_hidden.size(2) != config.hidden_size) {
            throw std::runtime_error("Qwen MTP session boundary is unavailable");
        }
        MtpSessionState state;
        state.cache_pos = predictor_position;
        state.blocks.reserve(blocks.size());
        for (const auto &block : blocks) {
            MfqCudaGuard guard(block->cuda_device);
            const auto *full = dynamic_cast<const FullBlock *>(block.get());
            if (full == nullptr) {
                throw std::runtime_error("Qwen MTP session layer type changed");
            }
            state.blocks.push_back(
                capture_full_attention_session_state(*full, predictor_position, state.bytes));
        }
        state.last_target_hidden = last_target_hidden.clone();
        state.bytes += session_tensor_bytes(state.last_target_hidden);
        return state;
    }

    void restore_session_state(const MtpSessionState &state) override {
        if (state.cache_pos <= 0 || state.blocks.size() != blocks.size() ||
            !state.last_target_hidden.defined() || state.last_target_hidden.dim() != 3 ||
            state.last_target_hidden.size(0) != 1 || state.last_target_hidden.size(1) != 1 ||
            state.last_target_hidden.size(2) != config.hidden_size) {
            throw std::runtime_error("Qwen MTP session state is incompatible");
        }
        for (std::size_t index = 0; index < blocks.size(); ++index) {
            auto &block = blocks[index];
            MfqCudaGuard guard(block->cuda_device);
            auto *full = dynamic_cast<FullBlock *>(block.get());
            if (full == nullptr) {
                throw std::runtime_error("Qwen MTP session layer type changed");
            }
            restore_full_attention_session_state(*full, state.blocks[index]);
        }
        cache_pos = state.cache_pos;
    }

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return false; }
    bool retains_partial_target_prefix() const noexcept override { return true; }
};
