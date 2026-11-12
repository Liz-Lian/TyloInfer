#include "engine/models/minicpmo45/state.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/session/state.h"
#include "engine/models/minicpmo45/model.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::minicpmo45 {

State::State(const Model& model)
    : layers(model.layers.size())
#ifdef TYLOINFER_NATIVE_CUDA
    , model_(&model)
#endif
    , maximum_(model.metadata.max_position_embeddings)
{
    for (const auto& layer : model.layers)
        layout_.push_back({layer.device, layer.cpu, layer.attention.kv_heads,
            layer.attention.attention_head_dim, layer.attention.max_position_embeddings});
}

void State::preallocate() {
    for (size_t index = 0; index < layers.size(); ++index) {
        const auto& layout = layout_[index];
        if (layout.cpu || layers[index].defined()) continue;
        MfqCudaGuard guard(layout.device);
        layers[index] = tyloi::state::allocate_kv(
            1, layout.heads, layout.capacity, layout.head_dim, false,
            ::tyloi::Device(::tyloi::kCUDA, layout.device), ::tyloi::kFloat16);
    }
}

void State::reset(int64_t next_batch) {
    if (next_batch <= 0) throw std::invalid_argument("MiniCPM TTS batch must be positive");
    cache_pos = decode_position_delta = 0;
    batch = next_batch;
    for (size_t index = 0; index < layers.size(); ++index) {
        auto& cache = layers[index];
        if (cache.defined() && cache.batch != batch) {
            MfqCudaGuard guard(layout_[index].device);
            cache = {};
        }
    }
}

std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    for (const auto& cache : layers) {
        if (cache.physical_key.defined()) result.push_back(cache.physical_key);
        if (cache.physical_value.defined()) result.push_back(cache.physical_value);
    }
    return result;
}

TextSessionStateKind State::text_session_state_kind() const {
    return layers.empty() ? TextSessionStateKind::Unsupported : TextSessionStateKind::FullAttention;
}

bool State::supports_text_session_state() const { return !layers.empty(); }
bool State::supports_paged_text_session_state() const { return !layers.empty(); }

TextSessionState State::capture_text_session_state(const std::vector<int64_t>& tokens) const {
    if (layers.empty() || batch != 1 || cache_pos <= 0 || size_t(cache_pos) != tokens.size())
        throw std::runtime_error("MiniCPM TTS session token count does not match the cache");
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_pos;
    state.decode_position_delta = decode_position_delta;
    state.payload = std::vector<KvSnapshot>{};
    auto& saved = std::get<std::vector<KvSnapshot>>(state.payload);
    for (size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        saved.push_back(capture_kv(layers[index], cache_pos, state.bytes));
    }
    return state;
}

State::Restore State::prepare(const TextSessionState& state) const {
    const auto* saved = std::get_if<std::vector<KvSnapshot>>(&state.payload);
    if (layers.empty() || !saved || saved->size() != layers.size() || state.cache_pos <= 0 ||
        state.cache_pos > maximum_ || size_t(state.cache_pos) != state.tokens.size())
        throw CudaSessionStateError("MiniCPM TTS session state is incompatible");
    for (size_t index = 0; index < saved->size(); ++index) {
        const auto& layout = layout_[index];
        const auto& cache = saved->at(index);
        validate_kv(cache);
        const auto device = ::tyloi::Device(layout.cpu ? ::tyloi::kCPU : ::tyloi::kCUDA,
                                       layout.cpu ? 0 : layout.device);
        if (cache.ring || cache.k.device() != device || cache.v.device() != device ||
            cache.k.scalar_type() != ::tyloi::kFloat16 ||
            cache.v.scalar_type() != ::tyloi::kFloat16 ||
            cache.k.size(1) != layout.heads || cache.v.size(1) != layout.heads ||
            cache.k.size(3) != layout.head_dim || cache.v.size(3) != layout.head_dim ||
            cache.k.size(2) != state.cache_pos || cache.capacity < state.cache_pos)
            throw CudaSessionStateError("MiniCPM TTS session cache geometry changed");
    }
    return {std::vector<tyloi::state::KvView>(layers.size()), state.cache_pos,
            state.decode_position_delta};
}

void State::stage(Restore& restore, const TextSessionState& state) const {
    const auto& saved = std::get<std::vector<KvSnapshot>>(state.payload);
    for (size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
#ifdef TYLOINFER_NATIVE_CUDA
        stage_kv(restore.layers[index], saved[index], layout_[index].head_dim, ::tyloi::kFloat16, {});
#else
        restore_kv(restore.layers[index], saved[index], layout_[index].head_dim, ::tyloi::kFloat16, {});
#endif
    }
}

void State::publish(Restore&& restore) noexcept {
    layers.swap(restore.layers);
    cache_pos = restore.cache_pos;
    decode_position_delta = restore.decode_position_delta;
    batch = 1;
}

#ifdef TYLOINFER_NATIVE_CUDA
namespace {
void add_growth(std::vector<std::pair<Device,std::size_t>>& values,Device device,std::size_t bytes) {
    if (!device.is_cuda() || !bytes) return;
    auto it=std::find_if(values.begin(),values.end(),[&](const auto& item){return item.first==device;});
    if (it==values.end()) values.emplace_back(device,bytes);
    else { if (it->second>std::numeric_limits<std::size_t>::max()-bytes) throw std::overflow_error("MiniCPM state growth overflow"); it->second+=bytes; }
}
}
std::vector<std::pair<Device,std::size_t>> State::reset_growth(std::int64_t batch) const {
    if (batch<=0) throw std::invalid_argument("MiniCPM batch must be positive");
    return {};
}
std::vector<std::pair<Device,std::size_t>> State::growth(std::int64_t next_batch,std::int64_t tokens,
        const tyloi::models::ForwardOptions& options,std::int64_t confirmed) const {
    if (next_batch<=0 || tokens<=0 || confirmed!=0 || cache_pos<0 ||
        tokens>maximum_ || cache_pos>maximum_-tokens || (cache_pos>0 && batch!=next_batch))
        throw std::invalid_argument("invalid MiniCPM forward geometry");
    std::vector<std::pair<Device,std::size_t>> result;
    for (std::size_t i=0;i<layers.size();++i) {
        const auto& l=layout_[i];
        if (layers[i].defined() && (cache_pos!=0 || layers[i].batch==next_batch)) continue;
        const auto capacity=false ? l.capacity : (options.kv_capacity>0 ? std::max(cache_pos+tokens,options.kv_capacity) : l.capacity);
        const Device device(l.cpu?kCPU:kCUDA,l.cpu?0:l.device);
        add_growth(result,device,tyloi::state::kv_bytes(next_batch,l.heads,capacity,l.head_dim,false,kFloat16,{}));
    }
    return result;
}
std::unique_ptr<tyloi::models::Restore> State::prepare_restore(const TextSessionState& snapshot) {
    struct Candidate final : tyloi::models::Restore {
        State& owner; TextSessionState snapshot; State::Restore candidate;
        Candidate(State& state,const TextSessionState& value) : owner(state),snapshot(value),candidate(state.prepare(snapshot)) {}
        std::vector<std::pair<Device,std::size_t>> growth() const override {
            std::vector<std::pair<Device,std::size_t>> result;
            for (const auto& kv:std::get<std::vector<KvSnapshot>>(snapshot.payload)) {
                add_growth(result,kv.k.device(),restore_prefix_bytes({},kv.k,2,kv.capacity));
                add_growth(result,kv.v.device(),restore_prefix_bytes({},kv.v,2,kv.capacity));
            }
            return result;
        }
        void stage() override { owner.stage(candidate,snapshot); }
        void publish() noexcept override { owner.publish(std::move(candidate)); }
    };
    return std::make_unique<Candidate>(*this,snapshot);
}
#endif

} // namespace tyloi::cuda::minicpmo45
