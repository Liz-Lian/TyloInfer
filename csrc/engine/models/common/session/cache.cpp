#include "engine/models/common/session/cache.h"
#include "engine/state/kv_cache.h"
#include <array>
#include <cstring>

#include "cuda_runtime_config.h"
#include "backends/cuda/engine/executor/execution.h"
#include "engine/models/deepseek_v41/session_codec.h"
#include "engine/models/qwen4_exp/session_codec.h"
#include "engine/models/common/session/state.h"
#include "engine/state/session/prefix_cache.h"
#include "engine/state/session/paged.h"
#include "engine/state/session/cache.h"
#include "source/model_source.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <fstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace mfq::cuda::internal {

bool cuda_has_separate_memory() {
    cudaDeviceProp properties{};
    TYLOINFER_CUDA_CHECK(cudaGetDeviceProperties(&properties, mfq_current_cuda_device()));
    return !properties.integrated && std::strstr(properties.name, "GB10") == nullptr;
}

static std::string cuda_prefix_cache_compatibility_key(
    const mfq::ModelSource &source, int64_t max_position_embeddings) {
    std::ostringstream key;
    key << "mfq-cuda-prefix-v1\n"
        << "codec=" << (source.architecture()=="qwen4_exp" ? "cuda-qwen4-session-v1" :
            source.architecture()=="deepseek_v41" ? "cuda-deepseek-v41-session-v1" : "cuda-full-attention-kv-v1") << '\n'
        << "context=" << max_position_embeddings << '\n'
        << "architecture=" << source.architecture() << '\n';
    const auto quantization=mfq::cuda::cuda_kv_quantization_tag();
    if(!quantization.empty())key<<"kv="<<quantization<<'\n';
    for (std::size_t index = 0; index < source.source_paths().size(); ++index) {
        const auto &path = source.source_paths()[index];
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source for prefix cache: " + error.message());
        }
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error("cannot identify model source timestamp: " + error.message());
        }
        const auto modified_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(modified.time_since_epoch())
                .count();
        key << "source=" << index << ':' << path.filename().string() << ':' << size << ':'
            << modified_ns << '\n';
    }
    for (const auto &tensor : source.tensors()) {
        key << "tensor=" << tensor.name << ':' << tensor.dtype << ':' << tensor.nbytes << '\n';
    }
    return key.str();
}

std::shared_ptr<mfq::cache::PagedPrefixCache> make_cuda_paged_prefix_cache(
    const mfq::ModelSource &source, int64_t max_position_embeddings,
    bool supports_paged_text_session_state, const mfq::engine::PrefixCacheConfig &config) {
    const auto format = source.metadata().find("source.format");
    // ponytail: HF source fingerprints exclude config sidecars for now.
    if (!config.enabled ||
        (format != source.metadata().end() && format->second == "hf-safetensors") ||
        !supports_paged_text_session_state) {
        return {};
    }
    mfq::cache::PrefixBlockStoreConfig cache;
    cache.cache_dir = config.directory;
    if (!source.source_paths().empty()) cache.model_path = source.source_paths().front();
    cache.compatibility_key = cuda_prefix_cache_compatibility_key(source, max_position_embeddings);
    cache.block_size_tokens = static_cast<size_t>(config.block_tokens);
    // A longer block may lack the local/rolling state of a shorter exact
    // checkpoint. Retain those boundaries under the existing byte-budget LRU.
    cache.retain_tail_versions = 0;
    cache.max_disk_bytes = config.disk_bytes;
    const bool separate = cuda_has_separate_memory();
    cache.max_hot_bytes = separate ? config.ram_bytes : config.hot_bytes;
    cache.spill_hot_on_eviction = separate;
    if (separate) cache.available_ram_bytes = []() -> std::uint64_t {
#ifdef _WIN32
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        if (GlobalMemoryStatusEx(&state)) return state.ullAvailPhys;
#elif defined(__linux__)
        std::ifstream input("/proc/meminfo");
        std::string name, unit;
        std::uint64_t kib = 0;
        while (input >> name >> kib >> unit) if (name == "MemAvailable:") return kib * 1024;
#endif
        return UINT64_MAX;
    };
    cache.max_pending_writes = config.pending_writes;
    cache.max_pending_bytes = config.pending_bytes;
    return std::make_shared<mfq::cache::PagedPrefixCache>(std::move(cache));
}

TextSessionState CudaSessionOps::decode(
    const std::vector<CudaPagedPayload>& payloads,
    const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    if (payloads.front() && payloads.front()->size() > 6 && (*payloads.front())[6] == '2')
        return mfq::models::qwen4_exp::decode_session(payloads, tokens, block_size);
    if (payloads.front() && payloads.front()->size() > 6 && (*payloads.front())[6] == '3')
        return mfq::models::deepseek_v41::decode_session(payloads, tokens, block_size);
    return decode_cuda_paged_session(payloads, tokens, block_size);
}

CudaPagedPayload CudaSessionOps::encode(
    const Snapshot& state, std::size_t block_size, std::size_t index) {
    if (state.kind() == TextSessionStateKind::Qwen4)
        return mfq::models::qwen4_exp::encode_block(state, block_size, index);
    if (state.kind() == TextSessionStateKind::DeepseekV41)
        return mfq::models::deepseek_v41::encode_block(state, block_size, index);
    return encode_cuda_paged_block(state, block_size, index);
}

bool CudaSessionOps::paged_mtp(const Snapshot& state) {
    return state.kind() == TextSessionStateKind::Qwen4 ||
        state.kind() == TextSessionStateKind::DeepseekV41;
}

bool CudaSessionOps::partial_blocks(const Snapshot& state) {
    return paged_mtp(state);
}

bool CudaSessionOps::paged_media(const Snapshot& state) {
    return state.kind() == TextSessionStateKind::DeepseekV41;
}

bool CudaSessionOps::checkpoint_needs_update(
    const CudaPagedPayload& payload, const Snapshot& state) {
    if (state.kind() != TextSessionStateKind::DeepseekV41) return false;
    return mfq::models::deepseek_v41::checkpoint_needs_update(payload, state);
}

std::size_t CudaSessionOps::decodable_blocks(
    const std::vector<CudaPagedPayload>& payloads) {
    return cuda_paged_decodable_blocks(payloads);
}

std::size_t CudaSessionOps::token_count(const CudaPagedPayload& payload) {
    return cuda_paged_payload_tokens(payload);
}

void CudaSessionOps::release_host_cache() {
    mfq_release_host_allocator_cache();
}

TextSessionCache::TextSessionCache(const mfq::engine::SessionCacheConfig &session_config,
    const mfq::engine::PrefixCacheConfig &prefix_config,
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache, bool supported, int disabled_reason,
    bool native_hot_snapshots)
    : impl_(std::make_unique<Impl>(
          session_config, prefix_config, std::move(paged_cache), supported, disabled_reason,
          native_hot_snapshots)) {}

TextSessionCache::~TextSessionCache() = default;

bool TextSessionCache::persistent_prefix_enabled() const noexcept {
    return impl_->persistent_prefix_enabled();
}

void TextSessionCache::store(const std::string &session_id, TextSessionState state) {
    impl_->reserve_ram(state.bytes);
    impl_->store(session_id, std::move(state));
}

size_t TextSessionCache::fork_session(
    const std::string &source_session, const std::string &target_session) {
    return impl_->fork_session(source_session, target_session);
}

size_t TextSessionCache::close_session(const std::string &session_id) {
    return impl_->close_session(session_id);
}

std::vector<std::pair<std::string, double>> TextSessionCache::metrics() const {
    return impl_->metrics();
}

std::map<int, std::size_t> TextSessionCache::device_hot_bytes() const {
    std::map<std::pair<int, std::uintptr_t>, std::size_t> allocations;
    const auto tensor = [&](const mfq_tensor_backend::Tensor& value) {
        if (!value.defined() || !value.is_cuda()) return;
        auto& bytes = allocations[{value.get_device(), reinterpret_cast<std::uintptr_t>(value.data_ptr())}];
        bytes = std::max(bytes, std::size_t(value.numel()) * value.element_size());
    };
    const auto payload_tensors = [&](const auto& value) {
        if constexpr (requires { value.tensors(tensor); }) {
            value.tensors(tensor);
        } else if constexpr (!std::is_same_v<std::decay_t<decltype(value)>, std::monostate>) {
            for (const auto& block : value) block.tensors(tensor);
        }
    };
    impl_->visit_snapshots([&](const TextSessionState& state) {
        std::visit(payload_tensors, state.payload);
        if (state.mtp) {
            tensor(state.mtp->last_target_hidden);
            std::visit(payload_tensors, state.mtp->payload);
        }
    });
    std::map<int, std::size_t> result;
    for (const auto& [address, bytes] : allocations) result[address.first] += bytes;
    return result;
}

size_t TextSessionCache::clear_live_sessions() noexcept { return impl_->clear_live_sessions(); }

size_t TextSessionCache::clear() { return impl_->clear(); }

uint64_t TextSessionCache::trim_hot(uint64_t target_bytes) { return impl_->trim_hot(target_bytes); }
uint64_t TextSessionCache::set_hot_limit(uint64_t max_bytes) { return impl_->set_hot_limit(max_bytes); }
uint64_t TextSessionCache::set_ram_limit(uint64_t max_bytes) { return impl_->set_ram_limit(max_bytes); }
void TextSessionCache::reserve_ram(uint64_t bytes) { impl_->reserve_ram(bytes); }
uint64_t TextSessionCache::set_disk_limit(uint64_t max_bytes) { return impl_->set_disk_limit(max_bytes); }
uint64_t TextSessionCache::refresh_disk_index() { return impl_->refresh_disk_index(); }

} // namespace mfq::cuda::internal

size_t cuda_paged_payload_tokens(const CudaPagedPayload& payload) {
    if(!payload)throw CudaSessionStateError("null CUDA prefix block");
    CudaPagedReader reader(*payload);std::array<uint8_t,8> magic{};reader.raw(magic.data(),magic.size(),"magic");
    constexpr std::array<uint8_t,8> qwen4_magic{'M','F','Q','C','U','D','2',0};
    constexpr std::array<uint8_t,8> full_magic{'M','F','Q','C','U','D','1',0};
    constexpr std::array<uint8_t,8> deepseek_magic{'M','F','Q','C','U','D','3',0};
    if((magic!=qwen4_magic && magic!=full_magic && magic!=deepseek_magic) || reader.scalar<uint32_t>("version")!=1)
        throw CudaSessionStateError("invalid CUDA prefix block header");
    (void)reader.scalar<uint32_t>("start");return reader.scalar<uint32_t>("count");
}
size_t cuda_paged_decodable_blocks(const std::vector<CudaPagedPayload>& payloads) {
    for(size_t count=payloads.size();count>0;--count) {
        const auto& payload=payloads[count-1];
        (void)cuda_paged_payload_tokens(payload);
        if((*payload)[6]=='1')return count;
        if((*payload)[6]=='3') {
            if(mfq::models::deepseek_v41::exact(payload))return count;
            continue;
        }
        if(mfq::models::qwen4_exp::exact(payload))return count;
    }
    return 0;
}



std::vector<CudaPagedPayload> encode_cuda_paged_session(const TextSessionState &state,
                                                        size_t block_size, size_t first_block,
                                                        size_t maximum_blocks) {
    static constexpr std::array<uint8_t, 8> magic{'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    const auto *layers = std::get_if<std::vector<KvSnapshot>>(&state.payload);
    // Paged prefixes are text-only; media positions are not block-sliceable.
    if (state.kind() != TextSessionStateKind::FullAttention || state.decode_position_delta != 0 ||
        !state.input_key.empty() || state.cache_pos <= 0 || layers == nullptr || layers->empty() ||
        state.tokens.size() != static_cast<size_t>(state.cache_pos)) {
        throw std::runtime_error("CUDA session state is not block-sliceable");
    }
    const size_t full_blocks = state.tokens.size() / block_size;
    if (first_block > full_blocks) {
        throw std::runtime_error("invalid CUDA cache first block");
    }
    const size_t end_block =
        maximum_blocks > full_blocks - first_block ? full_blocks : first_block + maximum_blocks;
    std::vector<CudaPagedPayload> result;
    result.reserve(end_block - first_block);
    for (size_t block_index = first_block; block_index < end_block; ++block_index) {
        const int64_t start = static_cast<int64_t>(block_index * block_size);
        CudaPagedWriter writer;
        writer.raw(magic.data(), magic.size());
        writer.scalar<uint32_t>(1);
        writer.scalar<uint32_t>(static_cast<uint32_t>(start));
        writer.scalar<uint32_t>(static_cast<uint32_t>(block_size));
        writer.scalar<uint32_t>(static_cast<uint32_t>(layers->size()));
        for (const auto &layer : *layers) {
            if (layer.ring || layer.capacity <= 0 || !layer.k.defined() || !layer.v.defined() ||
                !session_kv_geometry_matches(layer.k,layer.v) ||
                layer.k.size(2) < start + static_cast<int64_t>(block_size)) {
                throw std::runtime_error("CUDA session layer cannot be block-sliced");
            }
            writer.scalar<int64_t>(layer.capacity);
            writer.tensor(layer.k.narrow(2, start, static_cast<int64_t>(block_size)));
            writer.tensor(layer.v.narrow(2, start, static_cast<int64_t>(block_size)));
        }
        result.push_back(std::move(writer).finish());
    }
    return result;
}

CudaPagedPayload encode_cuda_paged_block(const TextSessionState &state, size_t block_size,
                                         size_t block_index) {
    auto result = encode_cuda_paged_session(state, block_size, block_index, 1);
    if (result.size() != 1) {
        throw std::runtime_error("invalid CUDA cache block index");
    }
    return std::move(result.front());
}

CudaDecodedPagedBlock decode_cuda_paged_block(const std::vector<uint8_t> &payload) {
    static constexpr std::array<uint8_t, 8> magic{'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    CudaPagedReader reader(payload);
    std::array<uint8_t, 8> found_magic{};
    reader.raw(found_magic.data(), found_magic.size(), "magic");
    if (found_magic != magic || reader.scalar<uint32_t>("version") != 1) {
        throw CudaSessionStateError("unsupported CUDA paged cache payload");
    }
    CudaDecodedPagedBlock block;
    block.start = reader.scalar<uint32_t>("start");
    block.count = reader.scalar<uint32_t>("count");
    const auto layer_count = reader.scalar<uint32_t>("layer count");
    if (block.count == 0 || layer_count == 0 || layer_count > 4096) {
        throw CudaSessionStateError("invalid CUDA paged cache geometry");
    }
    block.layers.reserve(layer_count);
    for (uint32_t index = 0; index < layer_count; ++index) {
        CudaDecodedPagedLayer layer;
        layer.capacity = reader.scalar<int64_t>("capacity");
        auto key = reader.tensor();
        auto value = reader.tensor();
        if (key.second != value.second || !session_kv_geometry_matches(key.first,value.first) ||
            key.first.size(2) != static_cast<int64_t>(block.count)) {
            throw CudaSessionStateError("invalid CUDA paged cache layer");
        }
        layer.device = key.second;
        layer.k = std::move(key.first);
        layer.v = std::move(value.first);
        block.layers.push_back(std::move(layer));
    }
    reader.expect_end();
    return block;
}

TextSessionState decode_cuda_paged_session(const std::vector<CudaPagedPayload> &payloads,
                                           const std::vector<int64_t> &tokens, size_t block_size) {
    if (payloads.empty() || tokens.size() != payloads.size() * block_size) {
        throw CudaSessionStateError("CUDA paged cache chain length mismatch");
    }
    std::vector<CudaDecodedPagedBlock> blocks;
    blocks.reserve(payloads.size());
    size_t expected_start = 0;
    size_t layer_count = 0;
    for (const auto &payload : payloads) {
        if (!payload) {
            throw CudaSessionStateError("CUDA paged cache block payload is null");
        }
        auto block = decode_cuda_paged_block(*payload);
        if (block.start != expected_start || block.count != block_size ||
            (layer_count != 0 && block.layers.size() != layer_count)) {
            throw CudaSessionStateError("incompatible CUDA paged cache chain");
        }
        expected_start += block.count;
        layer_count = block.layers.size();
        blocks.push_back(std::move(block));
    }
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = static_cast<int64_t>(tokens.size());
    state.payload = std::vector<KvSnapshot>{};
    auto &layers = std::get<std::vector<KvSnapshot>>(state.payload);
    layers.reserve(layer_count);
    for (size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        const auto &final = blocks.back().layers[layer_index];
        std::vector<mfq_tensor_backend::Tensor> keys;
        std::vector<mfq_tensor_backend::Tensor> values;
        keys.reserve(blocks.size());
        values.reserve(blocks.size());
        for (const auto &block : blocks) {
            const auto &layer = block.layers[layer_index];
            if (layer.capacity != final.capacity || layer.device != final.device ||
                layer.k.scalar_type() != final.k.scalar_type() ||
                layer.k.size(0) != final.k.size(0) || layer.k.size(1) != final.k.size(1) ||
                layer.k.size(3) != final.k.size(3) || layer.v.size(3)!=final.v.size(3)) {
                throw CudaSessionStateError("inconsistent CUDA paged cache topology");
            }
            keys.push_back(layer.k);
            values.push_back(layer.v);
        }
        auto key = mfq_tensor_backend::cat(keys, 2).contiguous();
        auto value = mfq_tensor_backend::cat(values, 2).contiguous();
        MfqCudaGuard guard(final.device);
        const auto options = key.options().device(
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, final.device));
        key = key.to(options, true, false).contiguous();
        value = value.to(options, true, false).contiguous();
        state.bytes += static_cast<size_t>(key.numel() * key.element_size() +
                                           value.numel() * value.element_size());
        layers.push_back(KvSnapshot{
            std::move(key),
            std::move(value),
            final.capacity,
            false,
        });
    }
    return state;
}

size_t session_tensor_bytes(const mfq_tensor_backend::Tensor &tensor) {
    if (!tensor.defined())
        return 0;
    return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

bool session_tensor_layout_matches(const mfq_tensor_backend::Tensor &target,
                                   const mfq_tensor_backend::Tensor &source) {
    return target.defined() && source.defined() && target.sizes() == source.sizes() &&
           target.scalar_type() == source.scalar_type() && target.device() == source.device();
}

void restore_session_tensor(mfq_tensor_backend::Tensor &target,
                            const mfq_tensor_backend::Tensor &source) {
    if (!source.defined()) {
        target = mfq_tensor_backend::Tensor();
        return;
    }
    if (!session_tensor_layout_matches(target, source)) {
        target = mfq_tensor_backend::empty(source.sizes(), source.options());
    }
    target.copy_(source);
}

void restore_session_prefix_tensor(mfq_tensor_backend::Tensor &target,
                                   const mfq_tensor_backend::Tensor &source, int64_t dimension,
                                   int64_t capacity) {
    if (!source.defined() || dimension < 0 || dimension >= source.dim() || capacity <= 0 ||
        source.size(dimension) > capacity) {
        throw CudaSessionStateError("session prefix tensor layout is invalid");
    }
    auto target_shape = source.sizes().vec();
    target_shape.at(static_cast<size_t>(dimension)) = capacity;
    const bool compatible =
        target.defined() && target.sizes() == mfq_tensor_backend::IntArrayRef(target_shape) &&
        target.scalar_type() == source.scalar_type() && target.device() == source.device();
    if (!compatible) {
        target = mfq_tensor_backend::empty(target_shape, source.options());
    }
    if (source.size(dimension) > 0) {
        target.narrow(dimension, 0, source.size(dimension)).copy_(source);
    }
}

KvSnapshot capture_kv(const KVCache &cache, int64_t cache_pos, size_t &bytes) {
    if (!cache.k.defined() || !cache.v.defined() || cache.k.dim() != 4 ||
        !session_kv_geometry_matches(cache.k,cache.v) || cache.k.size(0) != 1) {
        throw std::runtime_error("full-attention KV cache is unavailable");
    }
    KvSnapshot state;
    state.capacity = cache.k.size(2);
    state.ring = cache.ring;
    const int64_t saved_tokens =
        state.ring ? state.capacity : std::min<int64_t>(cache_pos, state.capacity);
    state.k = cache.k.narrow(2, 0, saved_tokens).clone();
    state.v = cache.v.narrow(2, 0, saved_tokens).clone();
    bytes += session_tensor_bytes(state.k);
    bytes += session_tensor_bytes(state.v);
    return state;
}

void validate_kv(const KvSnapshot& state) {
    if (!state.k.defined() || !state.v.defined() || state.capacity <= 0 || state.k.dim() != 4 ||
        !session_kv_geometry_matches(state.k,state.v) || state.k.size(0) != 1 ||
        state.k.size(2) > state.capacity) {
        throw CudaSessionStateError("full-attention session KV layout is invalid");
    }
}

void restore_kv(KVCache &cache, const KvSnapshot &state, int64_t head_dim,
                mfq_tensor_backend::ScalarType dtype, mfq::cuda::KvQuantization quantization) {
    validate_kv(state);
    restore_session_prefix_tensor(cache.k, state.k, 2, state.capacity);
    restore_session_prefix_tensor(cache.v, state.v, 2, state.capacity);
    cache.ring = state.ring;
    cache.quantization = quantization;
    cache.paged_batch = state.k.size(0);
    cache.paged_heads = state.k.size(1);
    cache.paged_head_dim = head_dim;
    cache.paged_dtype = dtype;
    cache.k_chunk_ptrs = {};
    cache.v_chunk_ptrs = {};
    cache.page_table = {};
    cache.page_size = 0;
    cache.pages_per_chunk = 0;
}
