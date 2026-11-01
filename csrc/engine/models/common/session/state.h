#pragma once

#include "backend.h"
#include "engine/state/recurrent.h"
#include "engine/models/common/session/paged.h"
#include "engine/models/deepseek_v4/snapshot.h"
#include "engine/models/deepseek_v41/snapshot.h"
#include "engine/models/glm_dsa/snapshot.h"
#include "engine/models/qwen4_exp/snapshot.h"

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

struct KVCache;
namespace mfq::cuda { struct KvQuantization; }
namespace mfq::models::deepseek_v41 { struct EngramHashSnapshot; }

struct KvSnapshot {
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
    int64_t capacity = 0;
    bool ring = false;

    template <class Visit>
    void tensors(const Visit& visit) const {
        visit(k);
        visit(v);
    }
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

    template <class Visit>
    void tensors(const Visit& visit) const {
        full_attention.tensors(visit);
        visit(mixer.convolution);
        visit(mixer.recurrent);
    }
};

struct MtpSessionState {
    std::vector<KvSnapshot> blocks;
    mfq_tensor_backend::Tensor last_target_hidden;
    std::vector<mfq::models::qwen4_exp::BlockSnapshot> qwen4_blocks;
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
    std::vector<mfq::models::deepseek_v4::BlockSnapshot>,
    std::vector<mfq::models::glm_dsa::BlockSnapshot>,
    std::vector<mfq::models::qwen4_exp::BlockSnapshot>,
    std::vector<mfq::models::deepseek_v41::BlockSnapshot>>;

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
