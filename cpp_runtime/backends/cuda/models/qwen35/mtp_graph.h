#pragma once

#include "models/qwen35/linear_attention.h"
#include "models/qwen35/ops.h"
#include "core/full_block.h"
#include "runtime/decode_window.h"
#include "mfq_cuda_context.h"
#include "runtime/graph_pool_lease.h"

#include <map>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mfq::cuda::qwen35 {

// A verification graph retains both projected recurrent inputs and the state
// before the window. Commit/rollback uses the same block transaction as eager
// verification; replay only rebinds its logical position to the current request.
class MtpTargetGraphs {
    using Tensor = mfq_tensor_backend::Tensor;
    struct Checkpoint {
        LinearAttentionBlock* block = nullptr;
        Tensor convolution, recurrent;
        LinearRecurrentInputs inputs;
    };
    struct Cell {
        DecodeWindow graph;
        Tensor ids, positions, lengths, raw, hidden, logits;
        std::vector<Checkpoint> checkpoints;
        std::map<int64_t,std::unique_ptr<DecodeWindow>> rollbacks;
        explicit Cell(cudaStream_t stream) : graph(stream) {}
    };
    Qwen35CausalLm& model_;
    std::shared_ptr<mfq::cuda::Context> context_;
    cudaStream_t stream_;
    std::shared_ptr<GraphPoolLease> pool_;
    std::map<std::pair<int64_t,int64_t>,std::unique_ptr<Cell>> cells_;
    uint64_t captures_ = 0, replays_ = 0;
    Cell* active_ = nullptr;
    int64_t active_start_ = -1, active_confirmed_ = 0, active_tokens_ = 0;
    uint64_t rollback_captures_ = 0, rollback_replays_ = 0;
    std::vector<const void*> addresses_;

    std::vector<const void*> state_addresses() const {
        std::vector<const void*> result;
        for(const auto& block:model_.blocks) {
            if(const auto* b=dynamic_cast<const LinearAttentionBlock*>(block.get())) {
                result.push_back(b->mixer.convolution.data_ptr());result.push_back(b->mixer.recurrent.data_ptr());
                result.push_back(b->decode_core_workspace.defined()?b->decode_core_workspace.data_ptr():nullptr);
            } else if(const auto* b=dynamic_cast<const FullBlock*>(block.get())) {
                result.push_back(b->cache.k.defined()?b->cache.k.data_ptr():nullptr);
                result.push_back(b->cache.v.defined()?b->cache.v.data_ptr():nullptr);
            }
        }
        return result;
    }

public:
    explicit MtpTargetGraphs(Qwen35CausalLm& model) : model_(model),
        context_(mfq::cuda::default_context(mfq_current_cuda_device())),
        stream_(mfq_current_cuda_stream()),pool_(GraphPoolLease::acquire(context_,stream_)) {}
    ~MtpTargetGraphs() {
        cells_.clear();
    }
    uint64_t captures() const noexcept {return captures_;}
    uint64_t replays() const noexcept {return replays_;}
    uint64_t rollback_captures() const noexcept {return rollback_captures_;}
    uint64_t rollback_replays() const noexcept {return rollback_replays_;}
    bool compatible(const Qwen35CausalLm& model) const noexcept {
        return &model_==&model && stream_==mfq_current_cuda_stream();
    }

    static bool eligible(const Qwen35CausalLm& model, const Tensor& ids) {
        if(!ids.is_cuda() || ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 ||
                model.cache_pos<=0 || model.speculative_start>=0 ||
                model.execution->layer_placement.enabled() ||
                model.execution->dense_cpu_layer_count ||
                model.execution->moe_expert_cache || model.execution->gguf_expert_cache ||
                model.num_experts()!=0 || model.decode_position_delta!=0 ||
                model.rope.sections.numel()!=0) return false;
        for(const auto& block:model.blocks) {
            if(block->cpu_offloaded || block->cuda_device!=ids.get_device())return false;
            if(auto* linear=dynamic_cast<LinearAttentionBlock*>(block.get())) {
                if(!linear->mixer.defined() || linear->mixer.speculative_pending)return false;
            } else if(dynamic_cast<FullBlock*>(block.get())==nullptr)return false;
        }
        return true;
    }

    std::optional<Tensor> run(const Tensor& ids, int64_t confirmed, Tensor* raw,
                              Tensor* logits, bool* initialized) {
        if(!eligible(model_,ids))return {};
        const auto addresses=state_addresses();
        if(addresses!=addresses_) {
            cells_.clear();active_=nullptr;addresses_=addresses;
        }
        const auto tokens=ids.size(1),start=model_.cache_pos;
        TYLOINFER_RUNTIME_CHECK(confirmed>=0 && (confirmed==0 || confirmed<tokens) &&
            start+tokens<=model_.max_position_embeddings(),"invalid Dense MTP graph window");
        const auto stream=mfq_get_current_cuda_stream(ids.get_device());
        auto& owner=cells_[{tokens,confirmed}];
        if(!owner)owner=std::make_unique<Cell>(stream.stream());
        auto& cell=*owner;
        if(!cell.ids.defined()) {
            cell.ids=ids.clone();
            cell.positions=mfq_tensor_backend::arange(start,start+tokens,ids.options());
            cell.lengths=mfq_tensor_backend::full({1},start+tokens,ids.options());
        } else {
            cell.ids.copy_(ids);
            auto positions=mfq_tensor_backend::arange(start,start+tokens,ids.options());
            cell.positions.copy_(positions);cell.lengths.fill_(start+tokens);
        }
        if(!cell.graph.valid()) {
            struct Saved {LinearAttentionBlock* block;Tensor convolution,recurrent;};
            std::vector<Saved> saved;
            for(const auto& block:model_.blocks)if(auto* b=dynamic_cast<LinearAttentionBlock*>(block.get()))
                saved.push_back({b,b->mixer.convolution.clone(),b->mixer.recurrent.clone()});
            const auto restore=[&] {
                for(const auto& value:saved) {
                    value.block->mixer.convolution.copy_(value.convolution);
                    value.block->mixer.recurrent.copy_(value.recurrent);
                    value.block->clear_speculative();
                }
            };
            const auto release=[&] {
                cell.raw={};cell.hidden={};cell.logits={};cell.checkpoints.clear();
            };
            cell.graph.capture([&] {
                auto hidden=model_.adapter_prepare_hidden(model_.embed_forward(cell.ids),1,tokens);
                model_.adapter_begin_forward(true);
                Block::Context context;
                context.token_ids=cell.ids;context.positions=cell.positions;
                context.full_positions=cell.positions;context.cache_positions=cell.positions;
                context.cache_position=start;context.confirmed_prefix=confirmed;
                context.planned_kv_length=model_.max_position_embeddings();
                context.sequence_lengths=cell.lengths;
                for(const auto& block:model_.blocks) {
                    block->set_token_ids(cell.ids);
                    hidden=block->forward_context(*model_.execution,std::move(hidden),context,model_.rope);
                }
                model_.adapter_finish_forward(cell.positions,1,tokens);
                cell.hidden=model_.finalize_hidden(hidden,1,tokens);
                cell.raw=model_.adapter_raw_hidden(hidden,cell.hidden);
                cell.logits=model_.logits_from_hidden(cell.hidden);
                if(confirmed)for(const auto& block:model_.blocks)
                    if(auto* b=dynamic_cast<LinearAttentionBlock*>(block.get()))
                        cell.checkpoints.push_back({b,b->mixer.speculative_convolution,
                            b->mixer.speculative_recurrent,b->speculative_inputs});
            },restore,release);
            ++captures_;
            addresses_=state_addresses();
            if(initialized)*initialized=true;
            std::cerr<<"runtime_qwen35_mtp_graph_capture rows="<<tokens<<" confirmed="<<confirmed
                <<" nodes="<<cell.graph.nodes()<<" count="<<captures_<<'\n';
        }
        cell.graph.run();++replays_;
        if(confirmed) {
            for(const auto& checkpoint:cell.checkpoints) {
                auto& b=*checkpoint.block;
                b.mixer.speculative_convolution=checkpoint.convolution;
                b.mixer.speculative_recurrent=checkpoint.recurrent;
                b.mixer.speculative_pending=true;b.speculative_inputs=checkpoint.inputs;
                b.speculative_start=start;b.speculative_confirmed=confirmed;b.speculative_tokens=tokens;
            }
            model_.speculative_start=start;model_.speculative_confirmed=confirmed;
            active_=&cell;active_start_=start;active_confirmed_=confirmed;active_tokens_=tokens;
        }
        model_.cache_pos+=tokens;
        if(raw)*raw=cell.raw.clone();
        if(logits)*logits=cell.logits;
        return cell.hidden;
    }

    bool rollback(int64_t accepted, bool* initialized=nullptr) {
        if(!active_ || model_.speculative_start<0)return false;
        const auto retained=active_confirmed_+accepted;
        TYLOINFER_RUNTIME_CHECK(model_.speculative_start==active_start_ && accepted>=0 &&
            retained>0 && retained<active_tokens_,"invalid Dense MTP graph rollback");
        auto& owner=active_->rollbacks[retained];
        if(!owner)owner=std::make_unique<DecodeWindow>(stream_);
        auto& graph=*owner;
        if(!graph.valid()) {
            struct Saved {LinearAttentionBlock* block;Tensor convolution,recurrent;};
            std::vector<Saved> saved;
            for(const auto& checkpoint:active_->checkpoints) {
                auto* b=checkpoint.block;
                saved.push_back({b,b->mixer.convolution.clone(),b->mixer.recurrent.clone()});
            }
            const auto restore=[&] {
                for(const auto& value:saved) {
                    value.block->mixer.convolution.copy_(value.convolution);
                    value.block->mixer.recurrent.copy_(value.recurrent);
                }
            };
            graph.capture([&] {
                for(const auto& checkpoint:active_->checkpoints) {
                    auto& b=*checkpoint.block;
                    b.mixer.convolution.copy_(checkpoint.convolution);
                    b.mixer.recurrent.copy_(checkpoint.recurrent);
                    b.replay_recurrent_cuda(checkpoint.inputs,retained);
                }
            },restore);
            ++rollback_captures_;
            if(initialized)*initialized=true;
            std::cerr<<"runtime_qwen35_mtp_rollback_capture rows="<<active_tokens_
                <<" retained="<<retained<<" nodes="<<graph.nodes()<<'\n';
        }
        graph.run();++rollback_replays_;
        const auto keep=active_start_+retained;
        for(const auto& block:model_.blocks)block->commit_speculative();
        model_.adapter_rollback_speculative(keep);model_.cache_pos=keep;
        model_.speculative_start=-1;model_.speculative_confirmed=0;active_=nullptr;
        return true;
    }
};
} // namespace mfq::cuda::qwen35
