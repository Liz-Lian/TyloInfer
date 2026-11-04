#pragma once

#include "engine/weights/encoding/encoded.h"

#include <cstdint>
#include <vector>

namespace tyloi {

// MX: block-scaled microscaling payload plus its scale stream.
class MxTensor : public EncodedTensor {
public:
    MxTensor() = default;
    MxTensor(DenseTensor values, DenseTensor scales, std::int64_t out, std::int64_t neuron_len)
        : values(std::move(values)), scales(std::move(scales)), out(out), neuron_len(neuron_len) {}

    DenseTensor values;
    DenseTensor scales;
    int64_t out = 0;
    int64_t neuron_len = 0;

    std::vector<const DenseTensor*> buffers() const override { return {&values, &scales}; }
    std::int64_t logical_rows() const override { return out; }

protected:
    const DenseTensor& primary() const override { return values; }
    std::int64_t logical_columns() const override { return neuron_len; }
};

class Mxfp4Tensor final : public MxTensor {
public:
    using MxTensor::MxTensor;
    std::shared_ptr<EncodedTensor> cloned() const override {
        return std::make_shared<Mxfp4Tensor>(*this);
    }
};
class Mxfp8Tensor final : public MxTensor {
public:
    using MxTensor::MxTensor;
    std::shared_ptr<EncodedTensor> cloned() const override {
        return std::make_shared<Mxfp8Tensor>(*this);
    }
};

} // namespace tyloi
