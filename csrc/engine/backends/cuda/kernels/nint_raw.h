#pragma once

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cstdint>

/** Describe a validated fixed-capacity NINT expert pool. */
struct NintSingleRouteWeight {
    std::int64_t pointers[9]{};
    std::int32_t geometry[3]{};
    std::int32_t local_experts = 0;
    std::int64_t row_metadata = 0;
};

namespace tyloi::cuda::nint_raw {

/** Describe one shared activation quantization launch and its output buffers. */
struct GroupQuantize {
    struct Group {
        std::int8_t *codes, *interleaved;
        float *scale;
        int *sum;
        int gs, groups, begin, columns;
    } group[4]{};
    const void *input = nullptr;
    int count = 0, rows = 0, width = 0;
    bool fp32 = false;
};

/** Quantize shared activation groups. */
void group_quantize(const GroupQuantize & args, int total, cudaStream_t stream);
/** Record the shared quantization launch during capture when tracing is enabled. */
void trace_group_quantize(const GroupQuantize & args, cudaStream_t stream);

/** Describe the small-M activation quantization buffers and geometry. */
struct Quantize {
    const void *input, *gate;
    std::int8_t *codes, *interleaved;
    float *scale;
    std::int32_t *sums;
    int rows, width, padded_width, groups, group_size, activation_mode, columns;
    bool input_float, gate_float, cached_sums, interleave;
};

/** Quantize one activation workspace using the selected layout. */
void quantize(const Quantize & args, cudaStream_t stream);

/** Describe the buffers and geometry of one routed NINT launch. */
struct Routed {
    const std::uint8_t *bitstream, *row_q_bits, *subgroup_scale, *subgroup_minimum;
    const std::int64_t *row_q_bit_offsets;
    const float *neuron_scale, *neuron_minimum, *input_scale;
    const std::int8_t *quantized_input;
    const void *input;
    const std::int32_t *route_ids, *expert_local, *ids_dst, *expert_bounds, *tile_bounds, *tile_experts;
    void *output;
    int tokens, routes, experts, output_rows, groups, padded_width, input_width;
    int group_size, q_expert_stride, route_tile_m, pool_phase, epilogue_mode;
    bool route_map_ready, masked_experts, routed_input;
};

/** Dispatch the original routed projection kernels on the supplied stream. */
void routed(const Routed & args, cudaStream_t stream);

/** Describe validated mixed-pool route buffers and dispatch policy. */
struct Mixed {
    const std::int64_t *pointers;
    const std::int32_t *params, *expert_pool, *expert_local, *ids;
    void *output;
    int pairs, routes, experts, outputs;
    std::int64_t input_width;
    int group_hint, bits_hint;
    const char *grouped, *lanes, *blocks, *rows, *hint, *warps;
    bool routed_input, multi_sum, late_scale;
};

/** Dispatch mixed-pool kernels using the supplied policy. */
void mixed(const Mixed & args, cudaStream_t stream);

/** Dispatch a single-pool hinted route using validated buffers. */
void single(NintSingleRouteWeight weight, const std::int32_t *expert_local,
            const std::int32_t *ids, void *output, int pairs, int routes,
            int experts, int outputs, bool routed_input, int bits_hint,
            bool packed_metadata, bool multi_sum, bool late_scale,
            cudaStream_t stream);

/** Describe packed weight rows for dense NINT decoding. */
struct Decode {
    const std::uint8_t *bitstream, *row_q_bits, *subgroup_scale, *subgroup_minimum;
    const std::int64_t *row_q_bit_offsets;
    const float *neuron_scale, *neuron_minimum;
    void *output;
    std::uint64_t bytes;
    int rows, groups, group_size;
    std::int64_t width;
    bool output_float, dense_reference;
};

/** Decode packed NINT weights with the original vector or scalar kernel. */
void decode(const Decode & args, cudaStream_t stream);

inline constexpr int prefill_tile_n = 64, prefill_tile_k = 64;

/** Run the original split-K online FP16 projection. */
void online(const Decode & weight, const void *input, void *output,
            float *partial, int rows, int split, cudaStream_t stream);

/** Describe a validated aligned Q8 projection and its dispatch choice. */
struct Q8 {
    Decode weight;
    const std::int8_t *input;
    const float *input_scale;
    const std::int32_t *sums;
    void *output;
    int rows, padded_width, q8_decode;
    bool grouped, ordered;
};

/** Dispatch aligned Q8 projection kernels. */
void q8(const Q8 & args, cudaStream_t stream);

/** Describe small-M projection buffers, scratch and the prepared policy. */
struct Projection {
    Q8 q8;
    const std::int8_t *interleaved;
    const char *dense_group_setting, *fixed_setting;
    int real_width, uniform_bits, device;
    bool aligned_q8, whole_input, cached_sums;
};

/** Dispatch the selected small-M kernel; return a device-query failure code. */
int project(const Projection & args, cudaStream_t stream);

/** Decode packed NINT8-zero rows to FP16 or FP32 on the supplied stream. */
void zero_decode(const std::uint8_t * quantized, const void * scale,
                 void * output, int rows, int groups, std::int64_t width,
                 bool output_float, cudaStream_t stream);

/** Quantize activations and project packed NINT8-zero weights. */
void zero_gemv(const std::uint8_t *quantized, const void *scale,
               const void *input, std::int8_t *workspace, float *input_scale,
               void *output, int rows, int outputs, int groups, int width,
               bool input_float, cudaStream_t stream);

/** Quantize and reconstruct NINT8-one activation groups. */
void one_quantize(const void *input, std::int8_t *quantized, void *scale,
                  void *sum, void *reconstructed, int rows, int width,
                  int groups, cudaStream_t stream);

/** Report stream selection and GEMM status independently. */
struct GemmStatus { cublasStatus_t stream, gemm; };

/** Run the original NINT dense GEMM with the supplied handle and stream. */
GemmStatus gemm_nt(cublasHandle_t handle, cudaStream_t stream,
                   const void *input, const void *weight, void *output,
                   int rows, int outputs, int width, bool float_accum,
                   bool float_output);

} // namespace tyloi::cuda::nint_raw
