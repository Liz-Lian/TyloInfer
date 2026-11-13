#include "engine/weights/cache.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/batch.h"
#include "engine/ops.h"

#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/budget.h"
#include "components/error.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::cuda::qwen4_exp {
namespace {

class Batch final : public tyloi::models::Batch {
    struct GraphPlan final : Plan {
        const Batch* owner;
        std::vector<Qwen4BatchRequestState*> states;
        bool greedy;
        Qwen4BatchDecodeGraph::Plan layout;
        GraphPlan(const Batch& owner, std::vector<Qwen4BatchRequestState*> states, bool greedy,
                  Qwen4BatchDecodeGraph::Plan layout)
            : owner(&owner), states(std::move(states)), greedy(greedy), layout(std::move(layout)) {
            if (this->layout.growth) growth.emplace_back(tyloi::Device(tyloi::kCUDA,owner.model_.primary_device),this->layout.growth);
            rebuild=this->layout.rebuild;
        }
    };
public:
    /** Own the optional graph for batches using this model. */
    Batch(Model& model, bool graph) : model_(model) {
        if(graph) graph_=std::make_unique<Qwen4BatchDecodeGraph>(model);
    }
    /** Create an independent request cache for this model. */
    std::unique_ptr<tyloi::models::BatchState> make_state(std::int64_t) override {
        return std::make_unique<Qwen4BatchRequestState>(model_);
    }
    /** Support packed independent prompt spans. */
    bool supports_packed() const override { return true; }
    /** Return the per-row prefill workspace budget. */
    std::size_t prefill_row_bytes() const override {
        return std::size_t(model_.metadata.hidden_size) * model_.metadata.hc_mult * sizeof(float) * 8;
    }
    /** Prefill the validated request states with their token rows. */
    tyloi::StepSequence<std::vector<tyloi::Tensor>> prefill(std::vector<tyloi::models::BatchState*> states,
                                      std::vector<tyloi::Tensor> ids) override {
        return qwen4_batch_prefill(model_,checked(states),std::move(ids));
    }
    /** Decode one batch under its current execution policy. */
    tyloi::Tensor decode(std::span<const tyloi::models::BatchRow> rows, tyloi::Tensor ids,
                         const tyloi::Tensor& host_ids) override {
        auto requests=checked_rows(rows);
        tyloi::ops::FfnScope scope(requests.size());
        return qwen4_batch_decode(model_,requests,std::move(ids),host_ids);
    }
    /** Plan an eligible graph for these request states. */
    tyloi::StepSequence<std::unique_ptr<Plan>> plan(std::span<const tyloi::models::BatchRow> rows, bool greedy) override {
        return plan_job(checked_rows(rows),greedy);
    }
    /** Capture or replay a graph using the matching plan. */
    std::optional<tyloi::Tensor> capture(std::span<const tyloi::models::BatchRow> rows,
            const tyloi::Tensor& ids,const tyloi::Tensor& host_ids,bool greedy,const Plan& value) override {
        auto requests=checked_rows(rows);
        const auto* plan=dynamic_cast<const GraphPlan*>(&value);
        if(!graph_ || !plan || plan->owner!=this || plan->states!=requests || plan->greedy!=greedy)
            throw std::invalid_argument("Qwen4 batch graph plan belongs to another request or model");
        tyloi::ops::FfnScope scope(requests.size());
        return graph_->run(requests,ids,host_ids,greedy,plan->layout);
    }
    /** Retire graph references before releasing request states. */
    void clear() override { if(graph_) graph_->clear(); }
    /** Retire graphs that borrow a request before its backing is released. */
    void release(tyloi::models::BatchState&) override { clear(); }
    /** Report graph capture and replay counts. */
    tyloi::engine::Metrics metrics() const override {
        return {{"continuous_batching_cuda_graph_captures",double(graph_?graph_->captures():0)},
                {"continuous_batching_cuda_graph_replays",double(graph_?graph_->replays():0)}};
    }
private:
    std::vector<Qwen4BatchRequestState*> checked_rows(std::span<const tyloi::models::BatchRow> rows) const {
        std::vector<tyloi::models::BatchState*> states;
        for (const auto& row : rows) if (row.eligible) states.push_back(row.state);
        return checked(states);
    }
    tyloi::StepSequence<std::unique_ptr<Plan>> plan_job(std::vector<Qwen4BatchRequestState*> requests, bool greedy) {
        if(!graph_) co_return;
        auto job=graph_->plan(requests,greedy);
        while(auto step=job.next()) {
            if(step.layer) throw std::logic_error("unexpected batch graph layer event");
            if(step.graph) co_yield *step.graph;
            if(step.value) {
                std::unique_ptr<Plan> result=std::make_unique<GraphPlan>(*this,std::move(requests),greedy,std::move(*step.value));
                co_yield std::move(result);
                co_return;
            }
        }
    }
    std::vector<Qwen4BatchRequestState*> checked(std::span<tyloi::models::BatchState* const> states) const {
        std::vector<Qwen4BatchRequestState*> result;
        result.reserve(states.size());
        for(auto* value:states) {
            auto* state=dynamic_cast<Qwen4BatchRequestState*>(value);
            if(!state || !state->belongs_to(model_))
                throw std::invalid_argument("Qwen4 batch state belongs to another model");
            result.push_back(state);
        }
        return result;
    }
    Model& model_;
    std::unique_ptr<Qwen4BatchDecodeGraph> graph_;
};

} // namespace

std::unique_ptr<tyloi::models::Batch> Model::batch(const tyloi::models::BatchConfig& config, tyloi::models::State*) {
    return std::make_unique<Batch>(*this,config.graph);
}
} // namespace tyloi::cuda::qwen4_exp
