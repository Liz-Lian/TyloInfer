#include "engine/models/session_codec.h"
#include "engine/models/common/session/cache.h"
#include "engine/models/session.h"
#include "engine/models/deepseek_v41/session_codec.h"
#include "engine/models/qwen4_exp/session_codec.h"
#include "engine/models/source.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace tyloi::cuda::internal {

std::string cuda_prefix_cache_compatibility_key(
    const tyloi::ModelSource &source, int64_t max_position_embeddings) {
    std::ostringstream key;
    key << "mfq-cuda-prefix-v1\n"
        << "codec=" << (source.architecture()=="qwen4_exp" ? "cuda-qwen4-session-v1" :
            source.architecture()=="deepseek_v41" ? "cuda-deepseek-v41-session-v1" : "cuda-full-attention-kv-v1") << '\n'
        << "context=" << max_position_embeddings << '\n'
        << "architecture=" << source.architecture() << '\n';
    const auto quantization=tyloi::cuda::cuda_kv_quantization_tag();
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

TextSessionState CudaSessionOps::decode(
    const std::vector<CudaPagedPayload>& payloads,
    const std::vector<std::int64_t>& tokens,
    std::size_t block_size) {
    if (payloads.front() && payloads.front()->size() > 6 && (*payloads.front())[6] == '2')
        return tyloi::models::qwen4_exp::decode_session(payloads, tokens, block_size);
    if (payloads.front() && payloads.front()->size() > 6 && (*payloads.front())[6] == '3')
        return tyloi::models::deepseek_v41::decode_session(payloads, tokens, block_size);
    return decode_cuda_paged_session(payloads, tokens, block_size);
}

CudaPagedPayload CudaSessionOps::encode(
    const Snapshot& state, std::size_t block_size, std::size_t index) {
    if (state.kind() == TextSessionStateKind::Qwen4)
        return tyloi::models::qwen4_exp::encode_block(state, block_size, index);
    if (state.kind() == TextSessionStateKind::DeepseekV41)
        return tyloi::models::deepseek_v41::encode_block(state, block_size, index);
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
    return tyloi::models::deepseek_v41::checkpoint_needs_update(payload, state);
}

std::size_t CudaSessionOps::decodable_blocks(
    const std::vector<CudaPagedPayload>& payloads) {
    return cuda_paged_decodable_blocks(payloads);
}

std::size_t CudaSessionOps::token_count(const CudaPagedPayload& payload) {
    return cuda_paged_payload_tokens(payload);
}

} // namespace tyloi::cuda::internal

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
            if(tyloi::models::deepseek_v41::exact(payload))return count;
            continue;
        }
        if(tyloi::models::qwen4_exp::exact(payload))return count;
    }
    return 0;
}
