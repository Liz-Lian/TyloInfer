#pragma once

#include "backend.h"
#include "engine/state/recurrent.h"
#include "engine/models/common/session/paged.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

struct Dsv4PoolState;
struct KVCache;
namespace mfq::cuda { struct KvQuantization; }
namespace mfq::models::deepseek_v41 { struct EngramHashSnapshot; }

struct KvSnapshot {
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
    int64_t capacity = 0;
    bool ring = false;
};

inline bool session_kv_geometry_matches(const mfq_tensor_backend::Tensor& k,
        const mfq_tensor_backend::Tensor& v) {
    if(!k.defined()||!v.defined()||k.dim()!=4||v.dim()!=4||k.scalar_type()!=v.scalar_type())return false;
    for(int axis=0;axis<3;++axis)if(k.size(axis)!=v.size(axis))return false;
    return k.scalar_type()==mfq_tensor_backend::kInt32 || k.size(3)==v.size(3);
}

enum class TextSessionStateKind : int {
    Unsupported,
    FullAttention,
    HybridAttention,
    DeepseekV4,
    GlmDsa,
    Qwen4,
    DeepseekV41,
};

struct Dsv4PoolSessionState {
    int64_t ratio = 0;
    int64_t head_dim = 0;
    int64_t cache_quant_mode = 0;
    int64_t capacity = 0;
    bool overlap = false;
    mfq_tensor_backend::Tensor state_kv;
    mfq_tensor_backend::Tensor state_gate;
    mfq_tensor_backend::Tensor previous_kv;
    mfq_tensor_backend::Tensor previous_gate;
    mfq_tensor_backend::Tensor pool;
};

struct Dsv4BlockSessionState {
    mfq_tensor_backend::Tensor local_cache;
    Dsv4PoolSessionState compressor;
    Dsv4PoolSessionState indexer_compressor;
};

struct DeepseekV41BlockSessionState {
    mfq_tensor_backend::Tensor local_kv, compressed_kv, index_k, partial_kv, partial_score;
    int64_t compressed_length = 0, partial_length = 0;
};

struct GlmDsaBlockSessionState {
    mfq_tensor_backend::Tensor kv_cache;
    mfq_tensor_backend::Tensor index_cache;
    int64_t kv_capacity = 0;
    int64_t index_capacity = 0;
    bool full_indexer = false;
};

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
    mfq::cuda::Recurrent::State mixer;
};

struct Qwen4BlockSessionState {
    mfq_tensor_backend::Tensor keys, values, index;
    mfq_tensor_backend::Tensor convolution, recurrent, ple_convolution;
    int64_t ple_batch = 0;
    std::vector<int64_t> ple_tokens;
};

struct MtpSessionState {
    std::vector<KvSnapshot> blocks;
    mfq_tensor_backend::Tensor last_target_hidden;
    std::vector<Qwen4BlockSessionState> qwen4_blocks;
    std::vector<mfq_tensor_backend::Tensor> qwen4_positions;
    std::vector<int64_t> qwen4_lengths;
    std::vector<mfq_tensor_backend::Tensor> deepseek_v41_rings;
    int64_t cache_pos = 0;
    size_t bytes = 0;
};

using TextSessionPayload = std::variant<
    std::monostate,
    std::vector<KvSnapshot>,
    std::vector<HybridBlockSessionState>,
    std::vector<Dsv4BlockSessionState>,
    std::vector<GlmDsaBlockSessionState>,
    std::vector<Qwen4BlockSessionState>,
    std::vector<DeepseekV41BlockSessionState>>;

struct TextSessionState {
    std::vector<int64_t> tokens;
    std::string input_key;
    TextSessionPayload payload;
    mfq_tensor_backend::Tensor qwen4_positions;
    std::shared_ptr<const mfq::models::deepseek_v41::EngramHashSnapshot> deepseek_v41_engram;
    std::optional<MtpSessionState> mtp;
    int64_t cache_pos = 0;
    int64_t decode_position_delta = 0;
    size_t bytes = 0;

    TextSessionStateKind kind() const noexcept {
        return static_cast<TextSessionStateKind>(payload.index());
    }
};

namespace mfq::cuda {
template <typename Model> struct CudaCausalOps;
template <typename Model> struct FullAttentionSessionCodec {
    using CausalModel = typename Model::template CausalModel<CudaCausalOps<Model>>;
    static TextSessionStateKind kind(const CausalModel &model);
    static bool supports_paged(const CausalModel &model);
    static TextSessionState capture(const CausalModel &model, const std::vector<int64_t> &tokens);
    static void restore(CausalModel &model, const TextSessionState &state);
};
} // namespace mfq::cuda

struct CudaDecodedPagedLayer {
    int64_t capacity = 0;
    int device = -1;
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
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

size_t session_tensor_bytes(const mfq_tensor_backend::Tensor& tensor);

size_t cuda_paged_decodable_blocks(const std::vector<CudaPagedPayload>&);
size_t cuda_paged_payload_tokens(const CudaPagedPayload&);

bool session_tensor_layout_matches(
    const mfq_tensor_backend::Tensor& target,
    const mfq_tensor_backend::Tensor& source);

void restore_session_tensor(
    mfq_tensor_backend::Tensor& target,
    const mfq_tensor_backend::Tensor& source);

void restore_session_prefix_tensor(
    mfq_tensor_backend::Tensor& target,
    const mfq_tensor_backend::Tensor& source,
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
    mfq_tensor_backend::ScalarType dtype,
    mfq::cuda::KvQuantization quantization);

Dsv4PoolSessionState capture_dsv4_pool_session_state(
    const Dsv4PoolState& source,
    int64_t cache_pos,
    size_t& bytes);

void restore_dsv4_pool_session_state(
    Dsv4PoolState& target,
    const Dsv4PoolSessionState& state);
