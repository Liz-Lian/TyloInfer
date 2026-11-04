#pragma once

#include "backend.h"
#include "engine/state/recurrent.h"
#include "engine/models/common/session/paged.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

struct TextSessionState;
enum class TextSessionStateKind : int;
struct KVCache;
namespace tyloi::state { struct KvQuantization; }
namespace tyloi::cuda { using KvQuantization = tyloi::state::KvQuantization; }

struct KvSnapshot {
    ::tyloi::DenseTensor k;
    ::tyloi::DenseTensor v;
    int64_t capacity = 0;
    bool ring = false;

    template <class Visit>
    void tensors(const Visit& visit) const {
        visit(k);
        visit(v);
    }
};

inline bool session_kv_geometry_matches(const ::tyloi::DenseTensor& k,
        const ::tyloi::DenseTensor& v) {
    if(!k.defined()||!v.defined()||k.dim()!=4||v.dim()!=4||k.scalar_type()!=v.scalar_type())return false;
    for(int axis=0;axis<3;++axis)if(k.size(axis)!=v.size(axis))return false;
    return k.scalar_type()==::tyloi::kInt32 || k.size(3)==v.size(3);
}

enum class HybridBlockSessionStateKind {
    FullAttention,
    Recurrent,
    // The block keeps no cache state, so a checkpoint only records position.
    Stateless,
};

struct HybridBlockSessionState {
    HybridBlockSessionStateKind kind =
        HybridBlockSessionStateKind::FullAttention;
    KvSnapshot full_attention;
    tyloi::cuda::Recurrent::State mixer;

    template <class Visit>
    void tensors(const Visit& visit) const {
        full_attention.tensors(visit);
        visit(mixer.convolution);
        visit(mixer.recurrent);
    }
};

namespace tyloi::cuda {
template <typename Model> struct CudaCausalOps;
template <typename Model> struct FullAttentionSessionCodec {
    using CausalModel = typename Model::template CausalModel<CudaCausalOps<Model>>;
    static TextSessionStateKind kind(const CausalModel &model);
    static bool supports_paged(const CausalModel &model);
    static TextSessionState capture(const CausalModel &model, const std::vector<int64_t> &tokens);
    static void restore(CausalModel &model, const TextSessionState &state);
};
} // namespace tyloi::cuda

struct CudaDecodedPagedLayer {
    int64_t capacity = 0;
    int device = -1;
    ::tyloi::DenseTensor k;
    ::tyloi::DenseTensor v;
};

struct CudaDecodedPagedBlock {
    uint32_t start = 0;
    uint32_t count = 0;
    std::vector<CudaDecodedPagedLayer> layers;
};

std::vector<CudaPagedPayload> encode_cuda_paged_session(
    const TextSessionState& state,
    size_t block_size,
    size_t first_block,
    size_t maximum_blocks = std::numeric_limits<size_t>::max());

CudaPagedPayload encode_cuda_paged_block(
    const TextSessionState& state,
    size_t block_size,
    size_t block_index);

CudaDecodedPagedBlock decode_cuda_paged_block(
    const std::vector<uint8_t>& payload);

TextSessionState decode_cuda_paged_session(
    const std::vector<CudaPagedPayload>& payloads,
    const std::vector<int64_t>& tokens,
    size_t block_size);

size_t session_tensor_bytes(const ::tyloi::DenseTensor& tensor);

bool session_tensor_layout_matches(
    const ::tyloi::DenseTensor& target,
    const ::tyloi::DenseTensor& source);

void restore_session_tensor(
    ::tyloi::DenseTensor& target,
    const ::tyloi::DenseTensor& source);

void restore_session_prefix_tensor(
    ::tyloi::DenseTensor& target,
    const ::tyloi::DenseTensor& source,
    int64_t dimension,
    int64_t capacity);

KvSnapshot capture_kv(
    const KVCache& cache,
    int64_t cache_pos,
    size_t& bytes);

void validate_kv(const KvSnapshot& state);

void restore_kv(
    KVCache& cache,
    const KvSnapshot& state,
    int64_t head_dim,
    ::tyloi::ScalarType dtype,
    tyloi::cuda::KvQuantization quantization);
