#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mfq::nvq {

struct Data {
    int format = 0;
    int sign_mode = 0;
    int sub_bits = 0;
    int gs = 0;
    int axis = 0;
    int neuron_len = 0;
    int out = 0;
    int ng = 0;
    int nvec = 0;
    int nsign = 0;
    std::vector<int64_t> shape;
    std::vector<uint8_t> indices_packed;
    std::vector<uint8_t> aux_packed;
    std::vector<uint8_t> sub_scale_packed;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<int8_t> codebook;
};

constexpr int vector_size(int format) {
    return (format == 3 || format == 10 || format == 12 || format == 15) ? 4 : 8;
}

constexpr int index_bits(int format) {
    return format == 1 ? 11 :
        (format == 8 ? 9 :
         (format == 7 ? 7 :
          (format == 9 ? 6 :
           (format == 12 ? 9 :
            ((format == 13 || format == 15) ? 10 :
             (format == 14 ? 12 : 8))))));
}

Data unpack(const std::vector<uint8_t>& blob, const std::string& dtype);

} // namespace mfq::nvq

namespace mfq::nepq {

struct Data {
    int profile = -1;
    int format = 0;
    int state_bits = 0;
    int index_bits = 0;
    int aux_bits = 0;
    int table_bytes = 0;
    int runtime_table_bytes = 0;
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int bank_count = 0;
    int rotation_block = 0;
    uint64_t rotation_seed = 0;
    int ng = 0;
    int nvec = 0;
    int nsuper = 0;
    std::vector<uint8_t> indices_packed;
    std::vector<uint8_t> aux_packed;
    std::vector<uint8_t> state_packed;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<int8_t> table_pool;
    std::vector<int8_t> grouped_table_pool;
    std::vector<uint8_t> bank_ids;
    std::vector<int8_t> rotation_signs;
    bool residual = false;
    bool residual_second = false;
    int residual_position_bits = 0;
    int residual_record_bits = 0;
    int residual_block_vectors = 0;
    int residual_blocks_per_row = 0;
    std::vector<uint16_t> residual_codebook_h;
    std::vector<int16_t> residual_first;
    std::vector<int16_t> residual_second_dense;
};

Data unpack(
    const std::vector<uint8_t>& blob,
    const std::string& dtype,
    const std::vector<uint8_t>& runtime_payload);

} // namespace mfq::nepq
