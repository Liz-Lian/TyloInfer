#include "engine/generation/batch_ops.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "components/error.h"
#include "engine/budget.h"
#include "engine/config.h"
#include "engine/executor.h"
#include "engine/generation/batch.h"
#include "engine/generation/batch_driver.h"
#include "engine/generation/decode.h"
#include "engine/generation/prefill_budget.h"
#include "engine/generation/sampling.h"
#include "engine/models/batch.h"
#include "engine/models/causal_model.h"
#include "engine/ops.h"
#include "engine/weights/cache.h"

#include <algorithm>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::engine {
namespace {

struct BatchRow : BatchRequest {
    using BatchRequest::BatchRequest;
    std::optional<Sampler> sampler;
    Tensor counts, ids;
    std::vector<std::unique_ptr<Executor::Timer>> timers;
    std::unique_ptr<models::BatchState> state;
    std::int64_t row = -1;
    double packed_ms = 0;
};

std::exception_ptr prefill_failure(std::exception_ptr failure) {
    try { std::rethrow_exception(failure); }
    catch (const std::exception& error) {
        const std::string message = error.what();
        if (dynamic_cast<const MfqResourceExhausted*>(&error) ||
            message.starts_with("CUDA process VRAM limit exceeded:") ||
            message.starts_with("CUDA VRAM limit exceeded:") ||
            message.starts_with("CUDA pinned allocation failed:"))
            return std::make_exception_ptr(InferenceInputError(InferenceInputErrorCode::ContextLength,
                "insufficient GPU or pinned RAM for prefill workspace; reduce concurrent requests or the packed prefill token budget"));
    } catch (...) {}
    return failure;
}

void merge_growth(models::BatchGrowth& target, const models::BatchGrowth& source) {
    for (const auto& [device, bytes] : source) {
        auto it=std::find_if(target.begin(),target.end(),[&](const auto& item) { return item.first==device; });
        if (it==target.end()) target.emplace_back(device,bytes);
        else it->second=prefill_budget_add(it->second,bytes);
    }
}

class Ops {
public:
    using Request = BatchRow;
    using State = BatchState<Request>;
    using Sample = models::BatchSample;
    struct Decoded { Tensor logits; std::vector<std::int64_t> greedy; };
    static constexpr bool isolate_prefill_failures = true;
    static constexpr bool isolate_context_growth_failures = true;

    Ops(models::CausalModel& model, std::unique_ptr<models::Batch> batch, Executor& executor,
        weights::Cache& weights, Budget& budget, ContinuousBatchConfig config,
        std::size_t workspace, bool packed, std::int64_t packed_tokens)
        : model_(model), batch_(std::move(batch)), executor_(executor), weights_(weights), budget_(budget),
          config_(std::move(config)), workspace_(workspace), packed_(packed), packed_tokens_(packed_tokens) {
        if (!batch_ || config_.max_sequences == 0 || config_.max_sequences > 32)
            throw std::invalid_argument("continuous batching supports 1..32 requests");
    }

    std::int64_t vocab_size() const { return model_.info().vocab_size; }
    std::int64_t max_context() const { return model_.info().max_context; }
    Metrics metrics() const {
        std::size_t bytes = 0, contexts = 0;
        for (const auto& weak : requests_)
            if (const auto request = weak.lock(); request && request->state) {
                bytes += request->state->resident_bytes();
                contexts += request->state->position() > 0;
            }
        auto result = batch_->metrics();
        result.emplace_back("continuous_batching_decode_batches", decode_batches_);
        result.emplace_back("continuous_batching_decode_tokens", decode_tokens_);
        result.emplace_back("continuous_batching_batched_greedy_batches", greedy_batches_);
        result.emplace_back("continuous_batching_packed_prefill_batches", packed_batches_);
        result.emplace_back("continuous_batching_packed_prefill_requests", packed_requests_);
        result.emplace_back("continuous_batching_packed_prefill_tokens", packed_rows_);
        if (std::none_of(result.begin(),result.end(),[](const auto& metric) {
            return metric.first=="continuous_batching_moe_cached_row_serial";
        })) result.emplace_back("continuous_batching_moe_cached_row_serial", 0.0);
        result.emplace_back("continuous_batching_request_kv_bytes", bytes);
        result.emplace_back("continuous_batching_request_contexts", contexts);
        return result;
    }
    void suspend_decode() {
        std::exception_ptr failure;
        try { weights_.finish_exchange(); } catch (...) { failure=std::current_exception(); }
        try { batch_->clear(); } catch (...) { if (!failure) failure=std::current_exception(); }
        if (failure) std::rethrow_exception(failure);
    }
    void resume_decode(std::int64_t position) {
        auto scope=executor_.enter();
        batch_->resume(position);
    }
    PrefillChunk prefill_chunk(const std::shared_ptr<Request>& request, PrefillChunk chunk) {
        auto scope = executor_.enter();
        if (!request->state) weights_.finish_exchange();
        initialize(request, chunk);
        std::size_t primary=0;
        for (const auto& [device, bytes] : request->state->prefill_growth_bytes(chunk.count)) {
            if (device==executor_.device()) primary=prefill_budget_add(primary,bytes);
            else if (bytes) budget_.reserve_kv(device,prefill_budget_add(bytes,std::size_t(64)<<20));
        }
        return budget_.batch(executor_.device(), batch_->prefill_row_bytes(), chunk,
            primary);
    }
    std::size_t prefill_batch_limit() const { return packed_ && batch_->supports_packed() ? config_.max_sequences : 1; }
    std::int64_t prefill_batch_token_budget() const { return packed_tokens_; }

    Sample prefill(const std::shared_ptr<Request>& request, PrefillChunk chunk) {
        try {
            auto scope = executor_.enter();
            initialize(request, chunk);
            auto timer = executor_.timer();
            if (chunk.count > 1024) weights_.workspace(workspace_);
            reserve(request->state->prefill_growth_bytes(chunk.count),std::size_t(256)<<20,true);
            std::vector<models::BatchState*> states{request->state.get()};
            std::vector<Tensor> ids{ops::contiguous(ops::slice(request->ids,1,chunk.offset,chunk.offset+chunk.count))};
            auto hidden = run_prefill(std::move(states), std::move(ids)).at(0);
            Tensor next;
            if (chunk.offset + chunk.count == std::int64_t(request->prompt.size()))
                next = sample_logits(*request->sampler,
                    ops::reshape(model_.logits(ops::select(hidden,1,-1)), {1,-1}),
                    request->counts, request->token_constraint);
            timer->stop();
            request->timers.push_back(std::move(timer));
            return finish(request, std::move(next));
        } catch (...) { std::rethrow_exception(prefill_failure(std::current_exception())); }
    }

    std::vector<PrefillBatchResult<Sample>> prefill_batch(
        const std::vector<std::pair<std::shared_ptr<Request>, PrefillChunk>>& items) {
        auto scope = executor_.enter();
        weights_.retire_decode();
        std::vector<PrefillBatchResult<Sample>> result(items.size());
        std::vector<std::size_t> ready;
        for (std::size_t i=0;i<items.size();++i) {
            try { initialize(items[i].first, items[i].second); ready.push_back(i); }
            catch (...) { result[i].failure = prefill_failure(std::current_exception()); }
        }
        const auto execute = [&](auto&& self, const std::vector<std::size_t>& rows) -> void {
            if (rows.empty()) return;
            models::BatchGrowth growth;
            for (auto row : rows) merge_growth(growth,items[row].first->state->prefill_growth_bytes(items[row].second.count));
            try { reserve(growth,std::size_t(256)<<20,true); }
            catch (const InferenceInputError& error) {
                if (error.code != InferenceInputErrorCode::ContextLength) throw;
                if (rows.size()==1) { result[rows.front()].failure=std::current_exception(); return; }
                const auto middle=rows.begin()+rows.size()/2;
                self(self, std::vector<std::size_t>(rows.begin(),middle));
                self(self, std::vector<std::size_t>(middle,rows.end()));
                return;
            }
            try {
                if (rows.size()==1) {
                    const auto row=rows.front();
                    result[row].sample=prefill(items[row].first,items[row].second);
                    return;
                }
                std::vector<models::BatchState*> states;
                std::vector<Tensor> ids;
                std::int64_t tokens=0;
                for (auto row : rows) {
                    const auto& [request,chunk]=items[row];
                    states.push_back(request->state.get());
                    ids.push_back(ops::contiguous(ops::slice(request->ids,1,chunk.offset,chunk.offset+chunk.count)));
                    tokens+=chunk.count;
                }
                auto timer=executor_.timer();
                if (tokens>1024) weights_.workspace(workspace_);
                auto hidden=run_prefill(std::move(states),std::move(ids));
                if (hidden.size()!=rows.size()) throw std::runtime_error("invalid packed prefill output count");
                std::vector<Tensor> last, next(items.size());
                std::vector<std::size_t> finished;
                for (std::size_t i=0;i<rows.size();++i) {
                    const auto& [request,chunk]=items[rows[i]];
                    if (chunk.offset+chunk.count==std::int64_t(request->prompt.size())) {
                        last.push_back(hidden[i]); finished.push_back(rows[i]);
                    }
                }
                if (!last.empty()) {
                    auto logits=ops::reshape(model_.logits(ops::cat(last,0)), {std::int64_t(last.size()),-1});
                    for (std::size_t i=0;i<finished.size();++i) {
                        const auto row=finished[i];
                        const auto& request=items[row].first;
                        next[row]=sample_logits(*request->sampler,ops::slice(logits,0,i,i+1),
                            request->counts,request->token_constraint);
                    }
                }
                timer->stop();
                const auto milliseconds=timer->elapsed_ms();
                for (auto row : rows) {
                    items[row].first->packed_ms += milliseconds*items[row].second.count/tokens;
                    result[row].sample=finish(items[row].first,std::move(next[row]));
                }
                ++packed_batches_; packed_requests_+=rows.size(); packed_rows_+=tokens;
            } catch (...) {
                const auto error=prefill_failure(std::current_exception());
                for (auto row : rows) result[row].failure=error;
            }
        };
        execute(execute,ready);
        return result;
    }

    void activate(const std::shared_ptr<Request>& request, const Sample& sample) {
        auto scope=executor_.enter();
        reserve(batch_->activate_growth(*request->state),std::size_t(64)<<20);
        batch_->activate(*request->state);
        accept(request,sample);
    }
    void discard_prefill(const std::shared_ptr<Request>& request) {
        auto scope=executor_.enter();
        suspend_decode();
        executor_.drain();
        if (request->state) batch_->release(*request->state);
        request->state.reset(); request->ids={}; request->timers.clear(); request->packed_ms=0;
    }
    void retire(const std::vector<std::shared_ptr<Request>>& retired, std::int64_t position) {
        auto scope=executor_.enter();
        weights_.finish_exchange();
        executor_.drain();
        for (const auto& request : retired) {
            if (request->state) batch_->release(*request->state);
            request->state.reset(); request->ids={}; request->timers.clear(); request->packed_ms=0;
        }
        batch_->retire(position);
        const bool active=std::any_of(requests_.begin(),requests_.end(),[](const auto& weak) {
            const auto request=weak.lock(); return request && request->state;
        });
        if (!active) { weights_.retire_decode(); executor_.trim(executor_.device()); }
    }
    Decoded decode(State& state) {
        auto scope=executor_.enter();
        std::vector<models::BatchRow> rows;
        std::vector<std::int64_t> tokens;
        for (const auto& request : state.active) {
            request->row=-1;
            if (!request->state || request->state->position()!=request->cache_length)
                throw std::runtime_error("batch decode position mismatch");
            const bool row_greedy=request->sampler->greedy() && !request->sampler->has_penalties() && !request->token_constraint;
            rows.push_back({request->state.get(),request->pending_token,request->cache_length,
                request->generation_limit-request->produced,request->eligible,row_greedy});
            if (!request->eligible) continue;
            request->row=std::int64_t(tokens.size());
            tokens.push_back(request->pending_token);
        }
        auto host=ops::reshape(ops::tensor(tokens,Device(kCPU)), {-1,1});
        auto ids=ops::copy(host,executor_.device());
        const bool greedy=config_.greedy && std::all_of(state.active.begin(),state.active.end(),[](const auto& request) {
            return !request->eligible || (request->sampler->greedy() && !request->sampler->has_penalties() && !request->token_constraint);
        });
        std::optional<Tensor> captured;
        try {
            reserve(batch_->decode_growth(rows),std::size_t(64)<<20);
            batch_->prepare_decode(rows);
            if (config_.cuda_graph) {
                auto plan=plan_graph(rows,greedy);
                if (plan) {
                    if (plan->rebuild) {
                        executor_.trim(executor_.device());
                        reserve(plan->growth,std::size_t(64)<<20);
                    }
                    captured=batch_->capture(rows,ids,host,greedy,*plan);
                }
            }
        } catch (const std::exception& error) {
            const std::string message=error.what();
            if (!dynamic_cast<const MfqResourceExhausted*>(&error) &&
                !message.starts_with("CUDA process VRAM limit exceeded:") &&
                !message.starts_with("CUDA VRAM limit exceeded:") &&
                !message.starts_with("CUDA pinned allocation failed:")) throw;
            throw InferenceInputError(InferenceInputErrorCode::ContextLength,
                "insufficient GPU or pinned RAM for the batch decode workspace; reduce concurrent requests");
        }
        if (!captured) {
            models::BatchGrowth growth;
            for (const auto& row:rows) if (row.eligible) merge_growth(growth,row.state->prefill_growth_bytes(1));
            reserve(growth,std::size_t(64)<<20);
        }
        auto output=captured ? std::move(*captured) : batch_->decode(rows,std::move(ids),host);
        std::vector<std::int64_t> selected;
        if (greedy) {
            selected=ops::integers(captured ? output : ops::sample_greedy(ops::contiguous(output)));
            ++greedy_batches_;
        }
        ++decode_batches_; decode_tokens_+=tokens.size();
        return {std::move(output),std::move(selected)};
    }
    Sample sample(const std::shared_ptr<Request>& request, const Decoded& decoded) {
        if (request->row<0) throw std::runtime_error("batch sampled an inactive row");
        if (!decoded.greedy.empty()) return {{},decoded.greedy.at(std::size_t(request->row)),{}};
        auto next=sample_logits(*request->sampler,ops::slice(decoded.logits,0,request->row,request->row+1),
            request->counts,request->token_constraint);
        const auto token=ops::item(next);
        return {std::move(next),token,{}};
    }
    void accept(const std::shared_ptr<Request>& request, const Sample& sample) {
        if (request->counts) ops::add_token_counts(request->counts,ops::contiguous(sample.next));
    }
    void recover(State& state) {
        auto scope=executor_.enter();
        std::exception_ptr failure;
        try { weights_.drain(); } catch (...) { failure=std::current_exception(); }
        try { executor_.drain(); } catch (...) { if (!failure) failure=std::current_exception(); }
        for (const auto& request:state.active) discard_prefill(request);
        for (const auto& request:state.prefilling) discard_prefill(request);
        batch_->recover();
        if (failure) std::rethrow_exception(failure);
    }

private:
    void reserve(const models::BatchGrowth& values, std::size_t margin, bool primary=false) {
        reserve_batch(budget_,executor_,values,margin,primary);
    }
    std::vector<Tensor> run_prefill(std::vector<models::BatchState*> states, std::vector<Tensor> ids) {
        return tyloi::engine::prefill_batch(*batch_,executor_,weights_,std::move(states),std::move(ids));
    }
    std::unique_ptr<models::Batch::Plan> plan_graph(const std::vector<models::BatchRow>& rows, bool greedy) {
        return plan_batch(*batch_,executor_,weights_,rows,greedy);
    }
    void initialize(const std::shared_ptr<Request>& request, PrefillChunk chunk) {
        if (!request->state) {
            if (chunk.offset!=0) throw std::runtime_error("batch lost request state");
            std::erase_if(requests_,[](const auto& weak){return weak.expired();});
            const bool active=std::any_of(requests_.begin(),requests_.end(),[](const auto& weak) {
                const auto row=weak.lock(); return row && row->state;
            });
            if (!active) { weights_.retire_decode(); executor_.trim(executor_.device()); }
            requests_.push_back(request);
            auto state=batch_->make_state(static_cast<std::int64_t>(request->prompt.size()));
            reserve(state->initial_growth_bytes(),std::size_t(64)<<20);
            request->state=std::move(state);
            request->state->initialize();
            request->ids=ops::reshape(ops::tensor(request->prompt,executor_.device()),{1,-1});
            request->sampler.emplace(request->sampling,executor_.device());
            if (request->sampler->has_penalties()) {
                request->counts=ops::zeros({vocab_size()},TensorOptions().device(executor_.device()).dtype(kInt32));
                ops::add_token_counts(request->counts,request->ids);
            }
        }
        if (request->state->position()!=chunk.offset) throw std::runtime_error("batch prefill position mismatch");
    }
    Sample finish(const std::shared_ptr<Request>& request, Tensor next) {
        if (!next) return {};
        const auto token=ops::item(next);
        double ms=request->packed_ms;
        for (const auto& timer:request->timers) ms+=timer->elapsed_ms();
        request->timers.clear(); request->ids={}; request->packed_ms=0;
        return {std::move(next),token,{request->prompt.size(),ms,0.0,ms}};
    }
    models::CausalModel& model_;
    std::unique_ptr<models::Batch> batch_;
    Executor& executor_;
    weights::Cache& weights_;
    Budget& budget_;
    ContinuousBatchConfig config_;
    std::size_t workspace_;
    bool packed_;
    std::int64_t packed_tokens_;
    std::vector<std::weak_ptr<Request>> requests_;
    std::int64_t decode_batches_=0,decode_tokens_=0,greedy_batches_=0;
    std::int64_t packed_batches_=0,packed_requests_=0,packed_rows_=0;
};
} // namespace

std::unique_ptr<Batch> make_batch(models::CausalModel& model, models::State* serial, Executor& executor,
    weights::Cache& weights, Budget& budget, const Config& config) {
    const auto& settings=config.continuous_batch;
    auto scope=executor.enter();
    auto batch = model.batch({settings.max_sequences,settings.paged_kv,settings.preallocate,
        settings.cpu_offload,settings.cached_slots,settings.cuda_graph,settings.greedy,
        settings.cuda_graph_minimum_tokens},serial);
    if (!batch) throw std::invalid_argument("continuous batching is unavailable for this model");
    try {
        for (const auto& [device,bytes] : batch->initial_growth_bytes())
            if (bytes) budget.reserve_kv(device,prefill_budget_add(bytes,std::size_t(64)<<20));
        batch->initialize();
        weights.finish_exchange();
        executor.drain();
    } catch (...) {
        const auto failure=std::current_exception();
        try { weights.finish_exchange(); } catch (...) {}
        try { executor.drain(); } catch (...) {}
        batch.reset();
        std::rethrow_exception(failure);
    }
    return std::make_unique<ContinuousBatch<Ops>>(config.generation.prefill_chunk_size,
        config.continuous_batch.prefill_token_budget,model,std::move(batch),executor,weights,budget,
        config.continuous_batch,config.workspace_reserve,config.continuous_batch.packed_prefill,
        config.continuous_batch.packed_tokens);
}
} // namespace tyloi::engine
#endif
