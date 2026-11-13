#include "engine/models/session/state.h"
#include "engine/models/session/snapshot.h"
#include "engine/ops.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

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
            const auto& k=layer.k;
            const auto& v=layer.v;
            if (layer.ring || layer.capacity <= 0 || !k.defined() || !v.defined() ||
                k.is_encoded() || v.is_encoded() || !session_kv_geometry_matches(k,v) ||
                k.size(2) < start + static_cast<int64_t>(block_size)) {
                throw std::runtime_error("CUDA session layer cannot be block-sliced");
            }
            writer.scalar<int64_t>(layer.capacity);
            writer.tensor(tyloi::ops::slice(k,2,start,start+static_cast<int64_t>(block_size)));
            writer.tensor(tyloi::ops::slice(v,2,start,start+static_cast<int64_t>(block_size)));
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
        std::vector<::tyloi::Tensor> keys;
        std::vector<::tyloi::Tensor> values;
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
        auto key=tyloi::ops::contiguous(tyloi::ops::cat(keys,2));
        auto value=tyloi::ops::contiguous(tyloi::ops::cat(values,2));
        const tyloi::Device device(tyloi::kCUDA,final.device);
        key=tyloi::ops::contiguous(tyloi::ops::copy(key,device));
        value=tyloi::ops::contiguous(tyloi::ops::copy(value,device));
        state.bytes += static_cast<size_t>(key.numel()*key.element_size()+value.numel()*value.element_size());
        layers.push_back(KvSnapshot{std::move(key),std::move(value),final.capacity,false});
    }
    return state;
}

size_t session_tensor_bytes(const ::tyloi::Tensor &tensor) {
    if (!tensor.defined())
        return 0;
    return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

bool session_tensor_layout_matches(const ::tyloi::Tensor &target,
                                   const ::tyloi::Tensor &source) {
    if (!target.defined() || !source.defined() || target.dim() != source.dim() ||
        target.scalar_type() != source.scalar_type() || target.device() != source.device()) return false;
    for (int64_t i = 0; i < target.dim(); ++i)
        if (target.size(i) != source.size(i)) return false;
    return true;
}

namespace {
std::vector<int64_t> tensor_shape(const ::tyloi::Tensor& tensor) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<std::size_t>(tensor.dim()));
    for(int64_t axis=0;axis<tensor.dim();++axis) shape.push_back(tensor.size(axis));
    return shape;
}
std::vector<int64_t> restore_prefix_shape(const ::tyloi::Tensor& source, int64_t dimension, int64_t capacity) {
    if (!source.defined() || source.is_encoded())
        throw CudaSessionStateError("session prefix tensor is not dense");
    if (!source.defined() || dimension < 0 || dimension >= source.dim() || capacity <= 0 ||
        source.size(dimension) > capacity)
        throw CudaSessionStateError("session prefix tensor layout is invalid");
    auto shape = tensor_shape(source);
    shape.at(static_cast<size_t>(dimension)) = capacity;
    return shape;
}

bool restore_prefix_compatible(const ::tyloi::Tensor& target, const ::tyloi::Tensor& source,
                               const std::vector<int64_t>& shape) {
    return target.defined() && !target.is_encoded() && target.dim()==static_cast<int64_t>(shape.size()) &&
        tensor_shape(target)==shape && target.scalar_type()==source.scalar_type() && target.device()==source.device();
}

std::size_t restore_shape_bytes(const std::vector<int64_t>& shape, std::size_t element_size) {
    auto bytes = element_size;
    for (const auto dimension : shape) {
        if (dimension < 0 || (dimension && bytes > std::numeric_limits<std::size_t>::max() / std::size_t(dimension)))
            throw CudaSessionStateError("session tensor size exceeds addressable memory");
        bytes *= std::size_t(dimension);
    }
    return bytes;
}
}

std::size_t restore_tensor_bytes(const ::tyloi::Tensor& target, const ::tyloi::Tensor& source) {
    if (!source.defined()) return 0;
    if (source.is_encoded())
        throw CudaSessionStateError("session tensor is not dense");
    if (target.defined() && !target.is_encoded() && session_tensor_layout_matches(target, source)) return 0;
    return restore_shape_bytes(tensor_shape(source), source.element_size());
}

std::size_t restore_prefix_bytes(const ::tyloi::Tensor& target, const ::tyloi::Tensor& source,
                                 int64_t dimension, int64_t capacity) {
    const auto shape = restore_prefix_shape(source, dimension, capacity);
    return restore_prefix_compatible(target, source, shape) ? 0 : restore_shape_bytes(shape, source.element_size());
}

::tyloi::Tensor restore_session_tensor(::tyloi::Tensor target,
                                       const ::tyloi::Tensor &source) {
    if (!source.defined()) {
        return {};
    }
    if (source.is_encoded())
        throw CudaSessionStateError("session tensor is not dense");
    if (!target.defined() || target.is_encoded() || !session_tensor_layout_matches(target, source)) {
        target = tyloi::ops::empty(tensor_shape(source), source.options());
    }
    tyloi::ops::copy_into(target, source);
    return target;
}

::tyloi::Tensor restore_session_prefix_tensor(::tyloi::Tensor target,
                                               const ::tyloi::Tensor &tensor, int64_t dimension,
                                               int64_t capacity) {
    const auto& source = tensor;
    auto target_shape = restore_prefix_shape(source, dimension, capacity);
    if (!restore_prefix_compatible(target, source, target_shape)) {
        target = tyloi::ops::empty(target_shape, source.options());
    }
    if (source.size(dimension) > 0) {
        auto prefix = tyloi::ops::slice(target, dimension, 0, source.size(dimension));
        tyloi::ops::copy_into(prefix, source);
    }
    return target;
}

KvSnapshot capture_kv(const tyloi::state::KvView &cache, int64_t cache_pos, size_t &bytes) {
    if (!cache.physical_key.defined() || !cache.physical_value.defined() ||
        cache.physical_key.is_encoded() || cache.physical_value.is_encoded() ||
        !cache.physical_key.defined() || !cache.physical_value.defined() || cache.physical_key.dim() != 4 ||
        !session_kv_geometry_matches(cache.physical_key,cache.physical_value) || cache.physical_key.size(0) != 1) {
        throw std::runtime_error("full-attention KV cache is unavailable");
    }
    KvSnapshot state;
    state.capacity = cache.physical_key.size(2);
    state.ring = cache.ring;
    const int64_t saved_tokens =
        state.ring ? state.capacity : std::min<int64_t>(cache_pos, state.capacity);
    state.k = tyloi::ops::clone(tyloi::ops::slice(cache.physical_key, 2, 0, saved_tokens));
    state.v = tyloi::ops::clone(tyloi::ops::slice(cache.physical_value, 2, 0, saved_tokens));
    bytes += session_tensor_bytes(state.k);
    bytes += session_tensor_bytes(state.v);
    return state;
}

void validate_kv(const KvSnapshot& state) {
    if (!state.k.defined() || !state.v.defined() || state.k.is_encoded() || state.v.is_encoded() ||
        state.capacity <= 0 || state.k.dim() != 4 ||
        !session_kv_geometry_matches(state.k,state.v) || state.k.size(0) != 1 ||
        state.k.size(2) > state.capacity) {
        throw CudaSessionStateError("full-attention session KV layout is invalid");
    }
}

void restore_kv(tyloi::state::KvView &cache, const KvSnapshot &state, int64_t head_dim,
                ::tyloi::ScalarType dtype, tyloi::state::KvQuantization quantization) {
    validate_kv(state);
    cache.physical_key = restore_session_prefix_tensor(cache.physical_key, state.k, 2, state.capacity);
    cache.physical_value = restore_session_prefix_tensor(cache.physical_value, state.v, 2, state.capacity);
    cache.visible_key = {};
    cache.visible_value = {};
    cache.ring = state.ring;
    cache.quantization = quantization;
    cache.batch = state.k.size(0);
    cache.heads = state.k.size(1);
    cache.head_dim = head_dim;
    cache.dtype = dtype;
    cache.key_chunk_ptrs = {};
    cache.value_chunk_ptrs = {};
    cache.page_table = {};
    cache.page_size = 0;
    cache.pages_per_chunk = 0;
}

void stage_kv(tyloi::state::KvView& cache, const KvSnapshot& state, int64_t head_dim,
              ::tyloi::ScalarType dtype, tyloi::state::KvQuantization quantization) {
    validate_kv(state);
    auto shape = [](const ::tyloi::Tensor& tensor, int64_t capacity) {
        std::vector<int64_t> result;
        for (int64_t axis = 0; axis < tensor.dim(); ++axis)
            result.push_back(axis == 2 ? capacity : tensor.size(axis));
        return result;
    };
    cache.physical_key = tyloi::ops::empty(shape(state.k, state.capacity), state.k.options());
    cache.physical_value = tyloi::ops::empty(shape(state.v, state.capacity), state.v.options());
    restore_kv(cache, state, head_dim, dtype, quantization);
}
