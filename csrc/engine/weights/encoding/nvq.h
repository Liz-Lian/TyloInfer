#pragma once

#include "engine/weights/encoding/encoded.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi {

inline constexpr int64_t kNvq2JscXlGroupExecKernelFormat = 16;
inline constexpr int64_t kNvq3JscLGroupExecKernelFormat = 17;

// NVQ: vector-quantized indices plus auxiliary codes and a codebook.
class NvqTensor final : public EncodedTensor {
    std::shared_ptr<EncodedTensor> cloned() const override {
        return std::make_shared<NvqTensor>(*this);
    }

public:
    DenseTensor indices_packed;
    DenseTensor aux_packed;
    DenseTensor sub_scale_packed;
    DenseTensor neuron_scale;
    DenseTensor codebook;
    // Optional decode-only int8 digits: NVQ1-L 8*v+delta, S 32*v+5*delta.
    // Canonical books retain the dense/CPU and reference decode contract.
    DenseTensor integer_codebook;
    // Complete NVQ1 GS24 records (S 4B, L 5B), aliasing indices_packed.
    DenseTensor decode_records;
    // One byte-preserving group stream replaces indices/aux/states.
    bool dense_groups = false;
    uint32_t bank_map = 0;
    int64_t format = 0;
    int64_t kernel_format = 0;
    int64_t sign_mode = 0;
    int64_t sub_bits = 0;
    int64_t gs = 0;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t neuron_len = 0;
    std::vector<int64_t> shape;


    std::vector<const DenseTensor*> buffers() const override {
        return {&indices_packed, &aux_packed, &sub_scale_packed, &neuron_scale, &codebook};
    }

protected:
    const DenseTensor& primary() const override { return indices_packed; }
    std::int64_t logical_rows() const override { return out; }
    std::int64_t logical_columns() const override { return neuron_len; }
};

} // namespace tyloi
