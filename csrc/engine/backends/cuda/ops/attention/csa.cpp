#include "engine/ops/csa.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/ops/attention/csa_core.h"
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::ops {
namespace {
std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("DeepSeek-V4.1 allocation planning overflow");
    return a * b;
}
std::size_t add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("DeepSeek-V4.1 allocation planning overflow");
    return a + b;
}
DenseTensor view(const Tensor& value) {
    if (value.defined() && !value.representation<DenseTensor>())
        throw std::invalid_argument("DeepSeek-V4.1 attention tensor must be dense");
    return value.representation<DenseTensor>() ? value.dense() : DenseTensor{};
}
struct SharedBridge {
    WorkShared value;
    Tensor compressed{DenseTensor{}}, index{DenseTensor{}}, topk{DenseTensor{}}, candidates{DenseTensor{}};
    explicit SharedBridge(const Csa::Shared& shared) {
        value.compressed_kv=view(shared.compressed_kv); value.index_k=view(shared.index_k);
        value.topk=view(shared.topk); value.candidate_blocks=view(shared.candidate_blocks);
        value.compressed_length=shared.compressed_length; value.ratio=shared.ratio;
        compressed.dense()=value.compressed_kv; index.dense()=value.index_k;
        topk.dense()=value.topk; candidates.dense()=value.candidate_blocks;
    }
    void publish(Csa::Shared& shared) {
        compressed.dense()=value.compressed_kv; index.dense()=value.index_k;
        topk.dense()=value.topk; candidates.dense()=value.candidate_blocks;
        shared.compressed_kv=compressed; shared.index_k=index;
        shared.topk=topk; shared.candidate_blocks=candidates;
        shared.compressed_length=value.compressed_length; shared.ratio=value.ratio;
    }
};
}
Tensor csa_candidates(const Tensor& value, std::int64_t block_size,
                      std::int64_t pool_length) {
    const auto& blocks = value.dense();
    const auto count = blocks.size(-1);
    auto block64 = blocks.to(::tyloi::kInt64);
    auto within =
        ::tyloi::cuda::arange(block_size, blocks.options().dtype(::tyloi::kInt64))
            .reshape({1, 1, 1, block_size});
    auto expanded = (block64.unsqueeze(-1) * static_cast<double>(block_size) + within)
                        .reshape({blocks.size(0), blocks.size(1), count * block_size});
    auto valid = (expanded >= 0.0) & (expanded < static_cast<double>(pool_length));
    return ::tyloi::cuda::where(valid, expanded, -1.0)
        .to(::tyloi::kInt64)
        .contiguous();
}

Csa::State::State(const State& other)
    : local_kv(other.local_kv), compressed_kv(other.compressed_kv), index_k(other.index_k),
      partial_kv(other.partial_kv), partial_score(other.partial_score),
      speculative_local_slots(other.speculative_local_slots), speculative_slot_ids(other.speculative_slot_ids),
      speculative_partial_kv(other.speculative_partial_kv), speculative_partial_score(other.speculative_partial_score),
      speculative_projected_kv(other.speculative_projected_kv), speculative_projected_score(other.speculative_projected_score),
      position(other.position), compressed_length(other.compressed_length), partial_length(other.partial_length),
      speculative_position(other.speculative_position), speculative_compressed_length(other.speculative_compressed_length),
      speculative_partial_length(other.speculative_partial_length), prefilled_global_end(other.prefilled_global_end),
      speculative_pending(other.speculative_pending), replay_local_window(other.replay_local_window) {}
Csa::State& Csa::State::operator=(const State& other) {
    if (this != &other) {
        State copy(other);
        *this = std::move(copy);
    }
    return *this;
}
Csa::Csa(CsaWeights weights, CsaConfig config) : impl_(std::make_unique<Impl>(std::move(weights), config)) {}
Csa::~Csa() = default;
Csa::Csa(Csa&&) noexcept = default;
Csa& Csa::operator=(Csa&&) noexcept = default;
void Csa::reset(State& state, std::int64_t batch) const { impl_->reset(state,batch); }
void Csa::begin_speculative(State& state, std::int64_t tokens) const {
    const MfqCudaGuard guard(impl_->config.device);
    impl_->begin_speculative(state,tokens);
}
void Csa::commit_speculative(State& state) const { impl_->commit_speculative(state); }
void Csa::rollback_speculative(State& state, std::int64_t position) const {
    const MfqCudaGuard guard(impl_->config.device);
    impl_->rollback_speculative(state,position);
}
std::size_t Csa::reset_growth(const State& state, std::int64_t batch) const {
    if (batch <= 0) throw std::invalid_argument("DeepSeek-V4.1 reset batch must be positive");
    const auto& c=impl_->config;
    const Device device(kCUDA,c.device);
    std::size_t bytes=0;
    const auto plan=[&](const Tensor& tensor,std::int64_t length,std::int64_t width,ScalarType type,std::size_t element) {
        const bool same=tensor.defined() && tensor.representation<DenseTensor>() && tensor.dense().is_contiguous() &&
            tensor.device()==device && tensor.scalar_type()==type && tensor.dim()==3 && tensor.size(0)==batch &&
            tensor.size(1)==length && tensor.size(2)==width;
        if(!same)bytes=add(bytes,multiply(multiply(multiply(std::size_t(batch),std::size_t(length)),std::size_t(width)),element));
    };
    plan(state.local_kv,c.sliding_window,c.head_dim,kFloat16,2);
    if(c.kv_source) {
        const auto capacity=std::max<std::int64_t>(1,c.maximum/c.ratio);
        plan(state.compressed_kv,capacity,c.head_dim,kFloat16,2);
        plan(state.index_k,capacity,c.index_head_dim,kFloat16,2);
        if(c.ratio>1) {
            plan(state.partial_kv,c.ratio,c.head_dim,kFloat32,4);
            plan(state.partial_score,c.ratio,c.head_dim,kFloat32,4);
        }
    }
    return bytes;
}
std::size_t Csa::suffix_growth(const State& state, std::int64_t batch, std::int64_t tokens) const {
    const auto& c=impl_->config;
    if(batch<=0 || tokens<=0 || tokens>c.sliding_window || state.speculative_pending || !state.local_kv.defined())
        throw std::invalid_argument("DeepSeek-V4.1 invalid speculative checkpoint");
    const auto rows=multiply(std::size_t(batch),std::size_t(tokens));
    auto bytes=add(multiply(multiply(std::size_t(tokens),sizeof(std::int64_t)),3),
        multiply(multiply(rows,std::size_t(c.head_dim)),4));
    if(state.partial_kv.defined())bytes=add(bytes,multiply(multiply(multiply(std::size_t(batch),std::size_t(c.ratio)),std::size_t(c.head_dim)),8));
    return bytes;
}
std::size_t Csa::growth(const State&, std::int64_t batch, std::int64_t tokens, bool speculative) const {
    if(batch<=0 || tokens<=0)throw std::invalid_argument("DeepSeek-V4.1 invalid forward allocation window");
    const auto& c=impl_->config;
    return speculative && c.kv_source && c.ratio>1 ?
        multiply(multiply(multiply(std::size_t(batch),std::size_t(tokens)),std::size_t(c.head_dim)),8) : 0;
}
std::vector<Tensor> Csa::storage(const State& state) const {
    std::vector<Tensor> result;
    const auto append=[&](const Tensor& value) { if(value.defined())result.emplace_back(value.dense()); };
    append(state.local_kv); append(state.compressed_kv); append(state.index_k); append(state.partial_kv); append(state.partial_score);
    append(state.speculative_local_slots); append(state.speculative_slot_ids); append(state.speculative_partial_kv);
    append(state.speculative_partial_score); append(state.speculative_projected_kv); append(state.speculative_projected_score);
    for(const auto& value:state.failed)append(value);
    return result;
}
void Csa::publish_compressed(State& state, Shared& shared, const Tensor& input, std::int64_t position) const {
    const MfqCudaGuard guard(impl_->config.device);
    SharedBridge bridge(shared);
    const auto base=state.failed.size();
    state.failed.reserve(base+37);
    state.failed.emplace_back(input.dense());
    state.failed.push_back(bridge.compressed);state.failed.push_back(bridge.index);
    state.failed.push_back(bridge.topk);state.failed.push_back(bridge.candidates);
    const auto scratch_start=state.failed.size();
    for (int i=0;i<32;++i) state.failed.emplace_back(DenseTensor{});
    bridge.value.scratch=state.failed.data()+scratch_start;
    if(state.speculative_pending) {
        state.speculative_projected_kv=Tensor{DenseTensor{}};
        state.speculative_projected_score=Tensor{DenseTensor{}};
    }
    try { impl_->publish_compressed_source(state,bridge.value,input.dense(),position); }
    catch (...) { bridge.publish(shared); throw; }
    bridge.publish(shared);
    state.failed.resize(base);
}
Tensor Csa::forward(State& state, Shared& shared, const Tensor& input, const Tensor& positions,
        std::int64_t position, const Tensor& sequence_lengths, bool truncated_local) const {
    const MfqCudaGuard guard(impl_->config.device);
    Tensor result{DenseTensor{}};
    SharedBridge bridge(shared);
    const auto base=state.failed.size();
    state.failed.reserve(base+40);
    state.failed.emplace_back(input.dense());state.failed.emplace_back(positions.dense());
    if(sequence_lengths.defined())state.failed.emplace_back(sequence_lengths.dense());
    state.failed.push_back(result);state.failed.push_back(bridge.compressed);state.failed.push_back(bridge.index);
    state.failed.push_back(bridge.topk);state.failed.push_back(bridge.candidates);
    const auto scratch_start=state.failed.size();
    for (int i=0;i<32;++i) state.failed.emplace_back(DenseTensor{});
    bridge.value.scratch=state.failed.data()+scratch_start;
    if(state.speculative_pending) {
        state.speculative_projected_kv=Tensor{DenseTensor{}};
        state.speculative_projected_score=Tensor{DenseTensor{}};
    }
    try {
        const auto lengths=sequence_lengths.defined() ? MfqOptional<DenseTensor>(sequence_lengths.dense()) : mfq_nullopt;
        result.dense()=impl_->attention_forward(state,bridge.value,input.dense(),positions.dense(),position,lengths,truncated_local);
    } catch (...) { bridge.publish(shared); throw; }
    bridge.publish(shared);
    state.failed.resize(base);
    return result;
}
} // namespace tyloi::ops
#endif
