#include "mfq_cuda_linear_attention_ops.h"
#include "gdn_update.cuh"
#include "mfq_cuda_context.h"
#include "reduce.cuh"
#include <type_traits>

namespace {
namespace tb = mfq_tensor_backend;

struct ReplayInput {
    const float* data;
    int64_t batch, head, token;
};

ReplayInput replay_input(const tb::Tensor& value) {
    return {value.data_ptr<float>(), value.stride(0), value.stride(1), value.stride(2)};
}

// Each warp owns disjoint state columns. The convolution work reads a separate
// checkpoint, so its history can be committed in parallel with the recurrence.
// No query/readout or compacted prefix copies are needed for state-only replay.
template<int D, bool TRANSPOSED, bool TILED_HEADS>
__global__ void gdn_replay_state_kernel(
    ReplayInput key, ReplayInput value, ReplayInput gate, ReplayInput beta,
    const float* initial_state, float* next_state,
    const float* __restrict__ initial_conv, float* __restrict__ next_conv,
    const float* __restrict__ qkv, int64_t qkv_batch_stride, int64_t qkv_token_stride,
    int key_heads, int value_heads, int tokens, int history, int channels,
    int state_blocks, int64_t convolution_elements) {
    constexpr int COLUMNS = 2, ROWS = D / 32, WARPS = 4;
    if (blockIdx.x >= state_blocks) {
        const int64_t index = int64_t(blockIdx.x - state_blocks) * blockDim.x + threadIdx.x;
        if (index < convolution_elements) {
            const int channel = index % channels;
            const int row = (index / channels) % history;
            const int batch = index / (int64_t(history) * channels);
            const int source_row = row + tokens;
            next_conv[index] = source_row < history ?
                initial_conv[(int64_t(batch) * history + source_row) * channels + channel] :
                qkv[int64_t(batch) * qkv_batch_stride +
                    int64_t(source_row - history) * qkv_token_stride + channel];
        }
        return;
    }
    constexpr int COLUMN_BLOCKS = (D + WARPS * COLUMNS - 1) / (WARPS * COLUMNS);
    const int bh = blockIdx.x / COLUMN_BLOCKS;
    const int column_block = blockIdx.x % COLUMN_BLOCKS;
    const int hv = bh % value_heads, batch = bh / value_heads;
    const int hq = TILED_HEADS ? hv % key_heads : hv / (value_heads / key_heads);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int first_column = (column_block * WARPS + warp) * COLUMNS;
    const int64_t state_offset = int64_t(bh) * D * D;
    const auto* keys = key.data + batch * key.batch + hq * key.head;
    const auto* values = value.data + batch * value.batch + hv * value.head;
    const auto* gates = gate.data + batch * gate.batch + hv * gate.head;
    const auto* betas = beta.data + batch * beta.batch + hv * beta.head;
    float state[COLUMNS][ROWS];
    #pragma unroll
    for (int c = 0; c < COLUMNS; ++c) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = r * 32 + lane, column = first_column + c;
            const int index = TRANSPOSED ? column * D + row : row * D + column;
            state[c][r] = initial_state[state_offset + index];
        }
    }
    for (int token = 0; token < tokens; ++token) {
        float keys_in_registers[ROWS], kv[COLUMNS] = {};
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            keys_in_registers[r] = keys[token * key.token + r * 32 + lane];
            #pragma unroll
            for (int c = 0; c < COLUMNS; ++c)
                kv[c] += state[c][r] * keys_in_registers[r];
        }
        const float decay = expf(gates[token * gate.token]);
        const float strength = betas[token * beta.token];
        float delta[COLUMNS];
        #pragma unroll
        for (int c = 0; c < COLUMNS; ++c) {
            kv[c] = warp_sum(kv[c]);
            delta[c] = (values[token * value.token + first_column + c] - decay * kv[c]) * strength;
        }
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            #pragma unroll
            for (int c = 0; c < COLUMNS; ++c)
                state[c][r] = mfq_gdn_update(state[c][r],decay,keys_in_registers[r],delta[c]);
        }
    }
    #pragma unroll
    for (int c = 0; c < COLUMNS; ++c) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = r * 32 + lane, column = first_column + c;
            const int index = TRANSPOSED ? column * D + row : row * D + column;
            next_state[state_offset + index] = state[c][r];
        }
    }
}
} // namespace

bool try_gdn_replay_state_cuda(
    const tb::Tensor& key, const tb::Tensor& value,
    const tb::Tensor& gate, const tb::Tensor& beta, const tb::Tensor& qkv,
    const tb::Tensor& initial_convolution, const tb::Tensor& initial_recurrent,
    tb::Tensor& convolution, tb::Tensor& recurrent,
    int64_t retained_tokens, bool transposed, bool tiled_heads) {
    const auto real = [](const tb::Tensor& tensor) {
        return tensor.defined() && tensor.is_cuda() && tensor.scalar_type() == tb::kFloat32;
    };
    if (!real(key) || !real(value) || !real(gate) || !real(beta) || !real(qkv) ||
        !real(initial_convolution) || !real(initial_recurrent) || !real(convolution) || !real(recurrent) ||
        key.dim() != 4 || value.dim() != 4 || gate.dim() != 3 || beta.dim() != 3 || qkv.dim() != 3 ||
        initial_convolution.dim() != 3 || initial_recurrent.dim() != 4 ||
        !initial_convolution.is_contiguous() || !initial_recurrent.is_contiguous() ||
        !convolution.is_contiguous() || !recurrent.is_contiguous() ||
        convolution.sizes() != initial_convolution.sizes() || recurrent.sizes() != initial_recurrent.sizes() ||
        initial_convolution.data_ptr() == convolution.data_ptr()) return false;
    // Each CTA owns complete, disjoint value columns and loads its initial
    // column before writing it. Recurrence can therefore commit in place;
    // convolution history still needs a separate immutable input buffer.
    const auto B = key.size(0), Hq = key.size(1), T = key.size(2), D = key.size(3), Hv = value.size(1);
    if (B <= 0 || Hq <= 0 || Hv <= 0 || Hv % Hq || retained_tokens <= 0 || retained_tokens > T ||
        (D != 32 && D != 64 && D != 128) || key.stride(3) != 1 || value.stride(3) != 1 ||
        value.sizes().vec() != std::vector<int64_t>{B,Hv,T,D} ||
        gate.sizes().vec() != std::vector<int64_t>{B,Hv,T} || beta.sizes() != gate.sizes() ||
        qkv.size(0) != B || qkv.size(1) < retained_tokens || qkv.stride(2) != 1 ||
        initial_convolution.size(0) != B || initial_convolution.size(1) <= 0 ||
        initial_convolution.size(2) != qkv.size(2) ||
        initial_recurrent.sizes().vec() != std::vector<int64_t>{B,Hv,D,D}) return false;
    const int device = key.get_device();
    const tb::Tensor* operands[] = {&value,&gate,&beta,&qkv,&initial_convolution,&initial_recurrent,&convolution,&recurrent};
    for (const auto* tensor : operands)
        if (tensor->get_device() != device) return false;
    const auto launch = [&](auto width, auto layout, auto heads) {
        constexpr int d = decltype(width)::value;
        constexpr int blocks_per_head = d / 8;
        const int state_blocks = B * Hv * blocks_per_head;
        const int64_t conv_elements = convolution.numel();
        const auto grid = unsigned(state_blocks + (conv_elements + 127) / 128);
        gdn_replay_state_kernel<d,decltype(layout)::value,decltype(heads)::value>
            <<<grid,128,0,mfq_current_cuda_stream()>>>(
                replay_input(key),replay_input(value),replay_input(gate),replay_input(beta),
                initial_recurrent.data_ptr<float>(),recurrent.data_ptr<float>(),
                initial_convolution.data_ptr<float>(),convolution.data_ptr<float>(),qkv.data_ptr<float>(),
                qkv.stride(0),qkv.stride(1),Hq,Hv,retained_tokens,convolution.size(1),qkv.size(2),
                state_blocks,conv_elements);
    };
    const auto with_width = [&](auto width) {
        if (transposed) {
            if (tiled_heads) launch(width,std::true_type{},std::true_type{});
            else launch(width,std::true_type{},std::false_type{});
        } else {
            if (tiled_heads) launch(width,std::false_type{},std::true_type{});
            else launch(width,std::false_type{},std::false_type{});
        }
    };
    if (D == 32) with_width(std::integral_constant<int,32>{});
    else if (D == 64) with_width(std::integral_constant<int,64>{});
    else with_width(std::integral_constant<int,128>{});
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    return true;
}
