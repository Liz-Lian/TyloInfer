#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/profile.h"
#endif

#include <algorithm>
#include <exception>

#ifdef TYLOINFER_NATIVE_CUDA
DecodeGraphBranchScope::DecodeGraphBranchScope()
    : setting(tyloi::cuda::Executor::current().mode().ffn.serial), previous(setting) {
    auto mode = tyloi::cuda::Executor::current_ffn();
    mode.serial = true;
    scope_ = tyloi::cuda::Executor::current().enter(mode);
    setting = true;
}
DecodeGraphBranchScope::~DecodeGraphBranchScope() { setting = previous; }
#endif

namespace {

#ifdef TYLOINFER_NATIVE_CUDA
const ::tyloi::DenseTensor& dense(const ::tyloi::Tensor& tensor) {
    const auto* result = tensor.representation<::tyloi::DenseTensor>();
    TYLOINFER_RUNTIME_CHECK(result, "decode graph requires a dense tensor");
    return *result;
}

::tyloi::DenseTensor& dense(::tyloi::Tensor& tensor) {
    auto* result = tensor.representation<::tyloi::DenseTensor>();
    TYLOINFER_RUNTIME_CHECK(result, "decode graph requires a dense tensor");
    return *result;
}

bool same_tensor_layout(const ::tyloi::DenseTensor& expected,
                        const ::tyloi::DenseTensor& candidate) {
    if (expected.defined() != candidate.defined()) return false;
    if (!expected.defined()) return true;
    const auto& left = expected.storage();
    const auto& right = candidate.storage();
    if (left.get() != right.get() || left.owner_before(right) || right.owner_before(left) ||
        expected.data_ptr() != candidate.data_ptr() ||
        expected.device() != candidate.device() ||
        expected.scalar_type() != candidate.scalar_type() ||
        expected.dim() != candidate.dim()) return false;
    for (std::int64_t dimension = 0; dimension < expected.dim(); ++dimension)
        if (expected.size(dimension) != candidate.size(dimension) ||
            expected.stride(dimension) != candidate.stride(dimension)) return false;
    return true;
}
#else
bool same_tensor_layout(const ::tyloi::DenseTensor& expected,
                        const ::tyloi::DenseTensor& candidate) {
    if (expected.defined() != candidate.defined()) return false;
    if (!expected.defined()) return true;
    if (expected.data_ptr() != candidate.data_ptr() ||
        expected.device() != candidate.device() ||
        expected.scalar_type() != candidate.scalar_type() ||
        expected.dim() != candidate.dim())
        return false;
    for (std::int64_t dimension = 0; dimension < expected.dim(); ++dimension) {
        if (expected.size(dimension) != candidate.size(dimension) ||
            expected.stride(dimension) != candidate.stride(dimension))
            return false;
    }
    return true;
}
#endif

#ifndef TYLOINFER_NATIVE_CUDA
bool same_storage(const std::vector<::tyloi::DenseTensor>& expected,
                  const std::vector<::tyloi::DenseTensor>& candidate) {
    if (expected.size() != candidate.size()) return false;
    for (std::size_t index = 0; index < expected.size(); ++index)
        if (!same_tensor_layout(expected[index], candidate[index])) return false;
    return true;
}
#endif

} // namespace

#ifdef TYLOINFER_NATIVE_CUDA
decode_graph_detail::Layout decode_graph_detail::snapshot_layout(
        const ::tyloi::Tensor& tensor) {
    Layout result;
    result.defined = tensor.defined();
    if (!result.defined) return result;
    result.dtype = tensor.scalar_type();
    result.device = tensor.device();
    for (std::int64_t dimension = 0; dimension < tensor.dim(); ++dimension)
        result.shape.push_back(tensor.size(dimension));
    for (const auto* buffer : tensor.buffers()) {
        if (buffer) result.buffers.emplace_back(*buffer);
        else result.buffers.emplace_back(std::nullopt);
    }
    return result;
}

bool decode_graph_detail::same_layout(const Layout& expected,
                                      const ::tyloi::Tensor& candidate) {
    if (expected.defined != candidate.defined()) return false;
    if (!expected.defined) return true;
    if (expected.dtype != candidate.scalar_type() || expected.device != candidate.device() ||
        expected.shape.size() != static_cast<std::size_t>(candidate.dim())) return false;
    for (std::size_t dimension = 0; dimension < expected.shape.size(); ++dimension)
        if (expected.shape[dimension] != candidate.size(dimension)) return false;
    const auto right = candidate.buffers();
    if (expected.buffers.size() != right.size()) return false;
    for (std::size_t index = 0; index < expected.buffers.size(); ++index) {
        if (!expected.buffers[index] || !right[index]) {
            if (static_cast<bool>(expected.buffers[index]) != (right[index] != nullptr)) return false;
        } else if (!same_tensor_layout(*expected.buffers[index], *right[index])) return false;
    }
    return true;
}

namespace {
bool same_storage(const std::vector<std::optional<decode_graph_detail::Layout>>& expected,
                  const std::vector<::tyloi::Tensor>& candidate) {
    if (expected.size() != candidate.size()) return false;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!expected[index] || !candidate[index]) {
            if (static_cast<bool>(expected[index]) != static_cast<bool>(candidate[index])) return false;
        } else if (!decode_graph_detail::same_layout(*expected[index], candidate[index])) return false;
    }
    return true;
}
}
#endif

std::vector<MfqCudaStream> make_cuda_graph_compute_streams(
        const MfqCudaStream& primary_stream,
        const tyloi::cuda::Collectives& collectives) {
    if (collectives.streams().empty()) return {primary_stream};
    std::vector<MfqCudaStream> streams;
    streams.reserve(collectives.streams().size());
    for (const auto& participant : collectives.streams()) {
        const int device = participant.device_index();
        streams.push_back(
            device == primary_stream.device_index()
                ? primary_stream
                : mfq_get_stream_from_pool(false, device));
    }
    return streams;
}

std::vector<MfqCudaStream> cuda_graph_participant_streams(
        const std::vector<MfqCudaStream>& compute_streams,
        const tyloi::cuda::Collectives& collectives) {
    auto participants = compute_streams;
    participants.insert(
        participants.end(),
        collectives.streams().begin(), collectives.streams().end());
    return participants;
}

std::vector<std::unique_ptr<MfqCudaStreamGuard>>
activate_cuda_graph_compute_streams(
        const std::vector<MfqCudaStream>& compute_streams) {
    std::vector<std::unique_ptr<MfqCudaStreamGuard>> guards;
    guards.reserve(compute_streams.size());
    try {
        for (const auto& stream : compute_streams)
            guards.push_back(std::make_unique<MfqCudaStreamGuard>(stream));
    } catch (...) {
        while (!guards.empty()) guards.pop_back();
        throw;
    }
    return guards;
}

#ifdef TYLOINFER_NATIVE_CUDA
namespace {
class ExecutionGraph final : public tyloi::Executor::Graph {
    struct Scope final : tyloi::Executor::Scope {
        MfqCudaGuard device;
        std::vector<std::unique_ptr<MfqCudaStreamGuard>> streams;
        std::unique_ptr<tyloi::Executor::Scope> executor;
        Scope(tyloi::cuda::Executor& owner, const std::vector<MfqCudaStream>& compute, int primary)
            : device(primary), streams(activate_cuda_graph_compute_streams(compute)) {
            try {
                auto mode = tyloi::cuda::Executor::current_ffn();
                mode.serial = true;
                executor = owner.enter(mode);
            } catch (...) {
                while (!streams.empty()) streams.pop_back();
                throw;
            }
        }
        ~Scope() override {
            executor.reset();
            while (!streams.empty()) streams.pop_back();
        }
    };
    struct Saved {
        tyloi::Tensor* target;
        ::tyloi::Tensor value;
        decode_graph_detail::Layout layout;
    };
public:
    ExecutionGraph(tyloi::cuda::Executor& owner, int primary)
        : owner_(owner), primary_(primary), stream_(mfq_get_stream_from_pool(false, primary)),
          compute_(make_cuda_graph_compute_streams(stream_, owner.collectives())),
          participants_(cuda_graph_participant_streams(compute_, owner.collectives())) {}
    ~ExecutionGraph() override { try { invalidate(); } catch (...) {} }
    std::unique_ptr<tyloi::Executor::Scope> enter() override {
        return std::make_unique<Scope>(owner_, compute_, primary_);
    }
    bool matches(std::span<const ::tyloi::Tensor> owners) const override {
        std::vector<::tyloi::Tensor> all(owners.begin(), owners.end());
        auto scratch = owner_.storage();
        all.insert(all.end(), scratch.begin(), scratch.end());
        return graph_.valid() && same_storage(layout_, all);
    }
    void prepare(std::span<tyloi::Tensor* const> recurrent) override {
        saved_.reserve(recurrent.size());
        for (auto* state : recurrent)
            saved_.push_back({state, tyloi::ops::clone(*state), decode_graph_detail::snapshot_layout(*state)});
        graph_.prepare_memory(participants_);
    }
    void restore() override {
        for (const auto& state : saved_) {
            TYLOINFER_RUNTIME_CHECK(decode_graph_detail::same_layout(state.layout, *state.target),
                "decode warmup changed recurrent storage addresses");
            tyloi::ops::copy_into(*state.target, state.value);
            TYLOINFER_RUNTIME_CHECK(decode_graph_detail::same_layout(state.layout, *state.target),
                "decode warmup changed recurrent storage addresses");
        }
        wait();
    }
    void begin() override { saved_.clear(); graph_.capture_begin(); }
    void end(std::span<const ::tyloi::Tensor> owners) override {
        graph_.capture_end();
        owners_.assign(owners.begin(), owners.end());
        auto scratch = owner_.storage();
        owners_.insert(owners_.end(), scratch.begin(), scratch.end());
        for (const auto& tensor : owners_)
            layout_.push_back(tensor ? std::optional(decode_graph_detail::snapshot_layout(tensor)) : std::nullopt);
        report_cuda_memory(owner_.options().report_memory, "runtime_graph_capture");
    }
    void replay() override { graph_.replay(); }
    void wait() override {
        std::exception_ptr failure;
        for (const auto& stream : participants_) {
            try {
                MfqCudaGuard guard(stream.device_index());
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            } catch (...) {
                if (!failure) failure = std::current_exception();
            }
        }
        if (failure) std::rethrow_exception(failure);
    }
    void invalidate() override {
        graph_.reset();
        wait();
        layout_.clear(); owners_.clear(); saved_.clear();
    }
private:
    tyloi::cuda::Executor& owner_;
    int primary_;
    MfqCudaStream stream_;
    std::vector<MfqCudaStream> compute_, participants_;
    MfqCudaGraph graph_;
    std::vector<Saved> saved_;
    std::vector<::tyloi::Tensor> owners_;
    std::vector<std::optional<decode_graph_detail::Layout>> layout_;
};
}

std::unique_ptr<tyloi::Executor::Graph> tyloi::cuda::Executor::graph() {
    MfqCudaGuard guard(primary_);
    if (!default_context(primary_)->supports_async_allocations()) return nullptr;
    return std::make_unique<ExecutionGraph>(*this, primary_);
}
#endif

DecodeGraphCache::DecodeGraphCache(int64_t context_capacity)
    : stream(mfq_get_stream_from_pool(false)),
      generated_capacity(std::max<int64_t>(context_capacity, 2048)) {}

void DecodeGraphCache::ensure_compute_streams(
        const tyloi::cuda::Collectives& collectives) {
    if (!compute_streams.empty()) return;
    compute_streams = make_cuda_graph_compute_streams(stream, collectives);
}

std::vector<MfqCudaStream> DecodeGraphCache::graph_participant_streams(
        const tyloi::cuda::Collectives& collectives) const {
    return cuda_graph_participant_streams(compute_streams, collectives);
}

bool DecodeGraphCache::matches(
        int64_t candidate_len, const MfqSamplingParams & sampling,
        bool candidate_greedy,
#ifdef TYLOINFER_NATIVE_CUDA
        const std::vector<::tyloi::Tensor>& candidate_storage) const {
#else
        const std::vector<::tyloi::DenseTensor>& candidate_storage) const {
#endif
    return valid && planned_len == candidate_len && greedy == candidate_greedy &&
#ifdef TYLOINFER_NATIVE_CUDA
           same_storage(storage_layout, candidate_storage) &&
#else
           same_storage(storage, candidate_storage) &&
#endif
           (candidate_greedy ||
            (temperature == sampling.temperature && top_k == sampling.top_k &&
             top_p == sampling.top_p)) &&
           presence_penalty == sampling.presence_penalty &&
           frequency_penalty == sampling.frequency_penalty &&
           repetition_penalty == sampling.repetition_penalty;
}

void DecodeGraphCache::ensure_storage(int64_t vocab_size) {
    auto i64 = ::tyloi::TensorOptions().dtype(::tyloi::kInt64).device(::tyloi::kCUDA);
#ifdef TYLOINFER_NATIVE_CUDA
    if (!static_input || !static_input.defined()) static_input = std::make_shared<::tyloi::DenseTensor>(::tyloi::cuda::empty({1, 1}, i64));
    if (!static_pos || !static_pos.defined()) static_pos = std::make_shared<::tyloi::DenseTensor>(::tyloi::cuda::empty({1}, i64));
    if (!static_len || !static_len.defined()) static_len = std::make_shared<::tyloi::DenseTensor>(::tyloi::cuda::empty({1}, i64));
    if (!static_step || !static_step.defined()) static_step = std::make_shared<::tyloi::DenseTensor>(::tyloi::cuda::empty({1}, i64));
    if (!generated || !generated.defined()) generated = std::make_shared<::tyloi::DenseTensor>(::tyloi::cuda::empty({generated_capacity}, i64));
    if (!random || !random.defined()) random = std::make_shared<::tyloi::DenseTensor>(::tyloi::cuda::empty(
        {1}, ::tyloi::TensorOptions().dtype(::tyloi::kFloat32).device(::tyloi::kCUDA)));
    if (!counts || !counts.defined()) counts = std::make_shared<::tyloi::DenseTensor>(::tyloi::cuda::empty(
        {vocab_size}, ::tyloi::TensorOptions().dtype(::tyloi::kInt32).device(::tyloi::kCUDA)));
#else
    if (!static_input.defined()) static_input = ::tyloi::cuda::empty({1, 1}, i64);
    if (!static_pos.defined()) static_pos = ::tyloi::cuda::empty({1}, i64);
    if (!static_len.defined()) static_len = ::tyloi::cuda::empty({1}, i64);
    if (!static_step.defined()) static_step = ::tyloi::cuda::empty({1}, i64);
    if (!generated.defined()) generated = ::tyloi::cuda::empty({generated_capacity}, i64);
    if (!random.defined()) {
        random = ::tyloi::cuda::empty(
            {1}, ::tyloi::TensorOptions().dtype(::tyloi::kFloat32).device(::tyloi::kCUDA));
    }
    if (!counts.defined()) {
        counts = ::tyloi::cuda::empty(
            {vocab_size}, ::tyloi::TensorOptions().dtype(::tyloi::kInt32).device(::tyloi::kCUDA));
    }
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
void DecodeGraphCache::initialize_inputs(
        const ::tyloi::Tensor& pending, int64_t position,
        ::tyloi::Tensor& random_host) {
    cudaStream_t graph_stream = stream.stream();
    std::int64_t length = position + 1;
    std::int64_t step = 1;
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(dense(static_input).data_ptr<int64_t>(),
        dense(pending).data_ptr<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(dense(generated).data_ptr<int64_t>(),
        dense(pending).data_ptr<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(dense(static_pos).data_ptr<int64_t>(),
        &position, sizeof(int64_t), cudaMemcpyHostToDevice, graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(dense(static_len).data_ptr<int64_t>(),
        &length, sizeof(int64_t), cudaMemcpyHostToDevice, graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(dense(static_step).data_ptr<int64_t>(),
        &step, sizeof(int64_t), cudaMemcpyHostToDevice, graph_stream));
    *dense(random_host).data_ptr<float>() = 0.5f;
    upload_random(random_host);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream));
}

void DecodeGraphCache::upload_random(const ::tyloi::Tensor& random_host) {
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(dense(random).data_ptr<float>(),
        dense(random_host).data_ptr<float>(), sizeof(float), cudaMemcpyHostToDevice, stream.stream()));
}

void DecodeGraphCache::capture_next(::tyloi::Tensor next) {
    static_next = std::move(next);
    decode_graph_commit_cuda(dense(static_next), dense(generated), dense(static_step),
                             dense(static_input), dense(static_pos), dense(static_len));
}
#else
void DecodeGraphCache::initialize_inputs(
        const ::tyloi::DenseTensor& pending, int64_t position,
        const ::tyloi::DenseTensor& random_host) {
    cudaStream_t graph_stream = stream.stream();
    std::int64_t length = position + 1;
    std::int64_t step = 1;
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_input.data_ptr<int64_t>(),
        pending.data_ptr<int64_t>(),
        sizeof(int64_t),
        cudaMemcpyDeviceToDevice,
        graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(generated.data_ptr<int64_t>(),
        pending.data_ptr<int64_t>(),
        sizeof(int64_t),
        cudaMemcpyDeviceToDevice,
        graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_pos.data_ptr<int64_t>(),
        &position,
        sizeof(int64_t),
        cudaMemcpyHostToDevice,
        graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_len.data_ptr<int64_t>(),
        &length,
        sizeof(int64_t),
        cudaMemcpyHostToDevice,
        graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_step.data_ptr<int64_t>(),
        &step,
        sizeof(int64_t),
        cudaMemcpyHostToDevice,
        graph_stream));
    *random_host.data_ptr<float>() = 0.5f;
    upload_random(random_host);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream));
}

void DecodeGraphCache::upload_random(const ::tyloi::DenseTensor& random_host) {
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(random.data_ptr<float>(),
        random_host.data_ptr<float>(),
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream()));
}
#endif

void DecodeGraphCache::invalidate() {
    if (graph) graph->reset();
    graph.reset();
    storage.clear();
#ifdef TYLOINFER_NATIVE_CUDA
    storage_layout.clear();
    static_next = {};
#else
    static_next = ::tyloi::DenseTensor();
#endif
    valid = false;
}

void DecodeGraphCache::set_key(
        int64_t candidate_len, const MfqSamplingParams & sampling,
        bool candidate_greedy,
#ifdef TYLOINFER_NATIVE_CUDA
        const std::vector<::tyloi::Tensor>& candidate_storage) {
#else
        const std::vector<::tyloi::DenseTensor>& candidate_storage) {
#endif
    planned_len = candidate_len;
    greedy = candidate_greedy;
    temperature = sampling.temperature;
    top_k = sampling.top_k;
    top_p = sampling.top_p;
    presence_penalty = sampling.presence_penalty;
    frequency_penalty = sampling.frequency_penalty;
    repetition_penalty = sampling.repetition_penalty;
#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<std::optional<decode_graph_detail::Layout>> layouts;
    layouts.reserve(candidate_storage.size());
    for (const auto& tensor : candidate_storage) {
        if (tensor) layouts.emplace_back(decode_graph_detail::snapshot_layout(tensor));
        else layouts.emplace_back(std::nullopt);
    }
    storage_layout = std::move(layouts);
#endif
    storage = candidate_storage;
    valid = true;
}

int64_t decode_graph_bucket(int64_t planned_len, int64_t context_capacity) {
    const int64_t quantum = planned_len <= 4096 ? 512 :
        (planned_len <= 16384 ? 1024 : 2048);
    const int64_t rounded = ((planned_len + quantum - 1) / quantum) * quantum;
    return std::min<int64_t>(rounded, context_capacity);
}

int64_t decode_graph_attention_parts(
        int64_t planned_len, int64_t max_parts) {
    const int64_t parts = planned_len >= 192
        ? (planned_len + 127) / 128 : 1;
    return std::min(parts, max_parts);
}
