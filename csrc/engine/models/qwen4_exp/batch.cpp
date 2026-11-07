#include "engine/weights/cache.h"
#include "engine/models/qwen4_exp/batch.h"
#include "engine/models/batch.h"

#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/executor/execution.h"
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

using tyloi::cuda::internal::PrefillCudaTimer;
using DenseTensor = ::tyloi::DenseTensor;

namespace {

class CacheScope {
  public:
    CacheScope(bool& serial, bool enabled) : serial_(serial), previous_(serial) {
        serial_ = enabled;
    }
    ~CacheScope() { serial_ = previous_; }

    CacheScope(const CacheScope&) = delete;
    CacheScope& operator=(const CacheScope&) = delete;

  private:
    bool& serial_;
    bool previous_ = false;
};

} // namespace

BatchOps::BatchOps(qwen4_exp::Model& model, CudaExecutionContext& execution,
                   tyloi::engine::Budget& budget, tyloi::engine::ContinuousBatchConfig config)
    : model_(model), execution_(execution), budget_(budget), config_(std::move(config)) {
    TYLOINFER_RUNTIME_CHECK(config_.max_sequences > 0 && config_.max_sequences <= 32,
                            "Flash-Next continuous batching supports 1..32 requests");
    model_.active->reset(1);
    if (config_.cuda_graph)
        graph_ = std::make_unique<Qwen4BatchDecodeGraph>(model_);
}

BatchOps::~BatchOps() = default;

int64_t BatchOps::vocab_size() const { return model_.metadata.vocab_size; }

int64_t BatchOps::max_context() const { return model_.max_position_embeddings(); }

tyloi::engine::Metrics BatchOps::metrics() const {
    std::size_t bytes = 0, contexts = 0;
    for (const auto& weak : requests_)
        if (const auto request = weak.lock(); request && request->cache) {
            bytes += request->cache->resident_bytes();
            contexts += request->cache->position() > 0;
        }
    return {{"continuous_batching_decode_batches", double(decode_batches_)},
            {"continuous_batching_decode_tokens", double(decode_tokens_)},
            {"continuous_batching_batched_greedy_batches", double(greedy_batches_)},
            {"continuous_batching_packed_prefill_batches", double(packed_prefill_batches_)},
            {"continuous_batching_packed_prefill_requests", double(packed_prefill_requests_)},
            {"continuous_batching_packed_prefill_tokens", double(packed_prefill_tokens_)},
            {"continuous_batching_moe_cached_row_serial", 0.0},
            {"continuous_batching_cuda_graph_captures",
             double(graph_ ? graph_->captures() : 0)},
            {"continuous_batching_cuda_graph_replays", double(graph_ ? graph_->replays() : 0)},
            {"continuous_batching_request_kv_bytes", double(bytes)},
            {"continuous_batching_request_contexts", double(contexts)}};
}

void BatchOps::suspend_decode() {
    if (graph_)
        graph_->clear();
}

void BatchOps::resume_decode(int64_t) {}

tyloi::engine::PrefillChunk BatchOps::prefill_chunk(
    const std::shared_ptr<Request>& request, tyloi::engine::PrefillChunk chunk) {
    CudaScope guard(execution_.config.layer_placement.primary_device(), execution_.executor().contexts());
    initialize_prefill(request, chunk);
    const auto row_bytes = std::size_t(model_.metadata.hidden_size) * model_.metadata.hc_mult * sizeof(float) * 8;
    return budget_.batch(tyloi::Device{tyloi::DeviceType::cuda, execution_.config.layer_placement.primary_device()},
        row_bytes, chunk, request->cache->prefill_growth_bytes(chunk.count));
}

static std::exception_ptr flash_next_prefill_failure(std::exception_ptr failure) {
    try {
        std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (dynamic_cast<const MfqResourceExhausted*>(&error) ||
            message.starts_with("CUDA process VRAM limit exceeded:") ||
            message.starts_with("CUDA VRAM limit exceeded:") ||
            message.starts_with("CUDA pinned allocation failed:"))
            return std::make_exception_ptr(tyloi::engine::InferenceInputError(
                tyloi::engine::InferenceInputErrorCode::ContextLength,
                "insufficient GPU or pinned RAM for prefill workspace; reduce concurrent requests or the packed prefill token budget"));
    } catch (...) {
    }
    return failure;
}

void BatchOps::initialize_prefill(const std::shared_ptr<Request>& request,
                                  tyloi::engine::PrefillChunk chunk) {
    if (!request->cache) {
        TYLOINFER_RUNTIME_CHECK(chunk.offset == 0, "Flash-Next batch lost request state");
        std::erase_if(requests_, [](const auto& weak) { return weak.expired(); });
        const bool active = std::any_of(requests_.begin(), requests_.end(), [](const auto& weak) {
            const auto row = weak.lock();
            return row && row->cache;
        });
        if (!active) {
            // The preceding exclusive request's global KV/recurrent state is
            // unused while independent batch rows own their own context.
            qwen4_release_decode_graphs(model_, nullptr);
            model_.active->reset(1);
            execution_.weights.retire_decode();
            default_context(execution_.config.layer_placement.primary_device())->reclaim_idle_memory();
        }
        requests_.push_back(request);
        request->cache = std::make_unique<Qwen4BatchRequestState>(model_);
        request->prefill_ids =
            ::tyloi::cuda::tensor(
                request->prompt,
                ::tyloi::TensorOptions()
                    .dtype(::tyloi::kInt64)
                    .device(::tyloi::Device(
                        ::tyloi::kCUDA, execution_.config.layer_placement.primary_device())))
                .reshape({1, -1});
        request->sampler.emplace(request->sampling);
        if (request->sampler->has_penalties())
            request->counts = SamplingOps::token_counts(request->prefill_ids, model_.metadata.vocab_size);
    }
    TYLOINFER_RUNTIME_CHECK(request->cache->position() == chunk.offset,
                            "Flash-Next batch prefill position mismatch");
}

BatchOps::Sample BatchOps::finish_prefill(const std::shared_ptr<Request>& request, DenseTensor next) {
    if (!next.defined())
        return {};
    const auto token = next.item<int64_t>();
    double milliseconds = request->packed_prefill_ms;
    for (const auto& measured : request->prefill_timers)
        milliseconds += measured->elapsed_ms();
    request->prefill_timers.clear();
    request->prefill_ids = {};
    request->packed_prefill_ms = 0;
    return {std::move(next), token, {request->prompt.size(), milliseconds, 0.0, milliseconds}};
}

BatchOps::Sample BatchOps::prefill(const std::shared_ptr<Request>& request,
                                   tyloi::engine::PrefillChunk chunk) {
    try {
        CudaScope guard(execution_.config.layer_placement.primary_device(), execution_.executor().contexts());
        initialize_prefill(request, chunk);
        auto timer = std::make_unique<PrefillCudaTimer>();
        auto hidden =
            qwen4_batch_prefill(model_, {request->cache.get()},
                                {request->prefill_ids.narrow(1, chunk.offset, chunk.count)
                                     .contiguous()})
                .front();
        const auto finished =
            chunk.offset + chunk.count == static_cast<int64_t>(request->prompt.size());
        DenseTensor next;
        if (finished)
            next = tyloi::engine::sample_logits(
                *request->sampler,
                dynamic_cast<const DenseTensor&>(*model_.logits(
                    hidden.index({::tyloi::indexing::Slice(), -1, ::tyloi::indexing::Slice()}))).reshape({1, -1}),
                request->counts, request->token_constraint);
        TYLOINFER_CUDA_CHECK(
            cudaEventRecord(timer->finished_event(), mfq_get_current_cuda_stream()));
        request->prefill_timers.push_back(std::move(timer));
        return finish_prefill(request, std::move(next));
    } catch (...) {
        std::rethrow_exception(flash_next_prefill_failure(std::current_exception()));
    }
}

std::size_t BatchOps::prefill_batch_limit() const {
    return ::tyloi::cuda::options::batch_packed_prefill() ? config_.max_sequences : 1;
}

int64_t BatchOps::prefill_batch_token_budget() const {
    return ::tyloi::cuda::options::batch_packed_prefill_token_budget();
}

std::vector<tyloi::engine::PrefillBatchResult<BatchOps::Sample>> BatchOps::prefill_batch(
    const std::vector<std::pair<std::shared_ptr<Request>, tyloi::engine::PrefillChunk>>& items) {
    CudaScope guard(execution_.config.layer_placement.primary_device(), execution_.executor().contexts());
    qwen4_release_decode_graphs(model_, nullptr);
    model_.adapter_begin_forward(false);
    execution_.weights.retire_decode();
    std::vector<tyloi::engine::PrefillBatchResult<Sample>> result(items.size());
    std::vector<std::size_t> ready;
    for (std::size_t row = 0; row < items.size(); ++row) {
        try {
            initialize_prefill(items[row].first, items[row].second);
            ready.push_back(row);
        } catch (...) {
            result[row].failure = flash_next_prefill_failure(std::current_exception());
        }
    }
    const auto execute = [&](auto&& self, const std::vector<std::size_t>& rows) -> void {
        if (rows.empty())
            return;
        std::size_t growth = std::size_t(256) << 20;
        for (const auto row : rows)
            growth += items[row].first->cache->prefill_growth_bytes(items[row].second.count);
        try {
            if (model_.reserve_kv_workspace)
                model_.reserve_kv_workspace(growth);
        } catch (const tyloi::engine::InferenceInputError& error) {
            if (error.code != tyloi::engine::InferenceInputErrorCode::ContextLength)
                throw;
            if (rows.size() == 1) {
                result[rows.front()].failure = std::current_exception();
                return;
            }
            const auto middle = rows.begin() + rows.size() / 2;
            self(self, std::vector<std::size_t>(rows.begin(), middle));
            self(self, std::vector<std::size_t>(middle, rows.end()));
            return;
        }
        try {
            if (rows.size() == 1) {
                const auto row = rows.front();
                result[row].sample = prefill(items[row].first, items[row].second);
                return;
            }
            std::vector<Qwen4BatchRequestState*> caches;
            std::vector<DenseTensor> chunks;
            int64_t tokens = 0;
            for (const auto row : rows) {
                const auto& [request, chunk] = items[row];
                caches.push_back(request->cache.get());
                chunks.push_back(
                    request->prefill_ids.narrow(1, chunk.offset, chunk.count).contiguous());
                tokens += chunk.count;
            }
            PrefillCudaTimer timer;
            auto hidden = qwen4_batch_prefill(model_, caches, chunks);
            std::vector<DenseTensor> last;
            std::vector<std::size_t> finished;
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const auto& [request, chunk] = items[rows[i]];
                if (chunk.offset + chunk.count == static_cast<int64_t>(request->prompt.size())) {
                    last.push_back(hidden[i]);
                    finished.push_back(rows[i]);
                }
            }
            std::vector<DenseTensor> next(items.size());
            if (!last.empty()) {
                auto logits = dynamic_cast<const DenseTensor&>(*model_.logits(::tyloi::cuda::cat(last, 0)))
                                  .reshape({int64_t(last.size()), -1});
                for (std::size_t i = 0; i < finished.size(); ++i) {
                    const auto row = finished[i];
                    const auto& request = items[row].first;
                    next[row] = tyloi::engine::sample_logits(*request->sampler, logits.narrow(0, i, 1),
                                              request->counts, request->token_constraint);
                }
            }
            TYLOINFER_CUDA_CHECK(
                cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
            const auto milliseconds = timer.elapsed_ms();
            for (const auto row : rows) {
                items[row].first->packed_prefill_ms +=
                    milliseconds * items[row].second.count / tokens;
                result[row].sample = finish_prefill(items[row].first, std::move(next[row]));
            }
            ++packed_prefill_batches_;
            packed_prefill_requests_ += rows.size();
            packed_prefill_tokens_ += tokens;
        } catch (...) {
            // Kernels may have advanced attention state. Fail this group;
            // splitting/retrying is allowed only before numerical execution.
            const auto error = flash_next_prefill_failure(std::current_exception());
            for (const auto row : rows)
                result[row].failure = error;
        }
    };
    execute(execute, ready);
    return result;
}

void BatchOps::activate(const std::shared_ptr<Request>& request, const Sample& sampled) {
    accept(request, sampled);
}

void BatchOps::discard_prefill(const std::shared_ptr<Request>& request) {
    if (graph_)
        graph_->clear();
    execution_.weights.finish_exchange();
    request->cache.reset();
    request->prefill_ids = {};
    request->prefill_timers.clear();
    request->packed_prefill_ms = 0;
}

void BatchOps::retire(const std::vector<std::shared_ptr<Request>>& retired, int64_t) {
    for (const auto& request : retired)
        discard_prefill(request);
    const bool active = std::any_of(requests_.begin(), requests_.end(), [](const auto& weak) {
        const auto request = weak.lock();
        return request && request->cache;
    });
    if (!active && !model_.graph_registry) {
        // Batch widths have distinct transfer arenas and graph scratch. Return
        // them at idle while preserving resident experts and adaptation heat.
        execution_.weights.retire_decode();
        default_context(execution_.config.layer_placement.primary_device())->reclaim_idle_memory();
    }
}

BatchOps::Decoded BatchOps::decode(State& state) {
    CudaScope guard(execution_.config.layer_placement.primary_device(), execution_.executor().contexts());
    std::vector<Qwen4BatchRequestState*> caches;
    std::vector<int64_t> tokens;
    for (const auto& request : state.active) {
        request->decode_row = -1;
        if (!request->eligible)
            continue;
        request->decode_row = static_cast<int64_t>(tokens.size());
        TYLOINFER_RUNTIME_CHECK(
            request->cache && request->cache->position() == request->cache_length,
            "Flash-Next batch decode position mismatch");
        caches.push_back(request->cache.get());
        tokens.push_back(request->pending_token);
    }
    auto host_ids = ::tyloi::cuda::tensor(
                        tokens,
                        ::tyloi::TensorOptions().dtype(::tyloi::kInt64))
                        .reshape({-1, 1});
    auto ids = host_ids.to(::tyloi::Device(
        ::tyloi::kCUDA, execution_.config.layer_placement.primary_device()));
    CacheScope moe_cache_scope(
        execution_.executor().mode().ffn.batch_serial, tokens.size() > 8);
    const bool greedy =
        config_.greedy && std::all_of(
                              state.active.begin(), state.active.end(), [](const auto& request) {
                                  return !request->eligible ||
                                         (request->sampler->greedy() &&
                                          !request->sampler->has_penalties() &&
                                          !request->token_constraint);
                              });
    std::optional<DenseTensor> captured;
    try {
        if (graph_)
            captured = graph_->run(caches, ids, host_ids, greedy);
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (!dynamic_cast<const MfqResourceExhausted*>(&error) &&
            !message.starts_with("CUDA process VRAM limit exceeded:") &&
            !message.starts_with("CUDA VRAM limit exceeded:") &&
            !message.starts_with("CUDA pinned allocation failed:"))
            throw;
        throw tyloi::engine::InferenceInputError(
            tyloi::engine::InferenceInputErrorCode::ContextLength,
            "insufficient GPU or pinned RAM for the batch decode workspace; reduce concurrent requests");
    }
    auto logits = captured ? std::move(*captured)
                           : qwen4_batch_decode(model_, caches, std::move(ids), host_ids);
    DenseTensor greedy_tokens;
    if (greedy) {
        greedy_tokens = captured ? logits.to(::tyloi::kCPU).contiguous()
                                 : sample_greedy_cuda(logits.contiguous())
                                       .to(::tyloi::kCPU)
                                       .contiguous();
        ++greedy_batches_;
    }
    ++decode_batches_;
    decode_tokens_ += tokens.size();
    return {std::move(logits), std::move(greedy_tokens)};
}

BatchOps::Sample BatchOps::sample(const std::shared_ptr<Request>& request,
                                  const Decoded& decoded) {
    TYLOINFER_RUNTIME_CHECK(request->decode_row >= 0, "Flash-Next sampled an inactive row");
    if (decoded.greedy_tokens.defined())
        return {{}, decoded.greedy_tokens.data_ptr<int64_t>()[request->decode_row], {}};
    auto next = tyloi::engine::sample_logits(
        *request->sampler, decoded.logits.narrow(0, request->decode_row, 1), request->counts,
        request->token_constraint);
    const auto token = next.item<int64_t>();
    return {std::move(next), token, {}};
}

void BatchOps::accept(const std::shared_ptr<Request>& request, const Sample& sampled) {
    if (request->counts.defined())
        sample_token_counts_add_cuda(request->counts, sampled.next.contiguous());
}

void BatchOps::recover(State& state) {
    for (const auto& request : state.active)
        discard_prefill(request);
    for (const auto& request : state.prefilling)
        discard_prefill(request);
}

} // namespace tyloi::cuda::qwen4_exp

namespace tyloi::cuda {

std::unique_ptr<tyloi::engine::Batch> make_batch(
    qwen4_exp::Model& model, CudaExecutionContext& execution, tyloi::engine::Budget& budget,
    const tyloi::engine::ContinuousBatchConfig& config, std::int64_t chunk_size) {
    return std::make_unique<tyloi::engine::ContinuousBatch<qwen4_exp::BatchOps>>(
        chunk_size, config.prefill_token_budget, model, execution, budget, config);
}

} // namespace tyloi::cuda
