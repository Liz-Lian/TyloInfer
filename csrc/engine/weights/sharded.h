#pragma once

#include "engine/tensor.h"
#include "engine/weights/tensor_parallel.h"

namespace tyloi {

class ShardedTensor final : public TensorImpl {
public:
    struct Shard {
        int device = 0;
        int64_t input_begin = 0;
        int64_t input_end = 0;
        int64_t output_begin = 0;
        int64_t output_end = 0;
        Tensor weight;
    };

    ShardedTensor(int64_t out, int64_t in, ScalarType dtype,
                  TensorParallelAxis axis, std::vector<Shard> shards)
        : axis(axis), shards(std::move(shards)),
          view_(make_contiguous_view(nullptr, std::array<int64_t, 2>{out, in}, dtype,
                                     Device(kCUDA, this->shards.empty() ? 0 : this->shards.front().device))) {
        if (out <= 0 || in <= 0 || this->shards.empty() ||
            (axis != TensorParallelAxis::Input && axis != TensorParallelAxis::Output))
            throw std::invalid_argument("invalid sharded tensor geometry");
        int64_t end = 0;
        for (const auto& shard : this->shards) {
            if (!shard.weight || shard.weight.dim() != 2 || shard.device < 0 ||
                shard.input_begin < 0 || shard.input_end <= shard.input_begin || shard.input_end > in ||
                shard.output_begin < 0 || shard.output_end <= shard.output_begin || shard.output_end > out ||
                shard.weight.size(0) != shard.output_end - shard.output_begin ||
                shard.weight.size(1) < shard.input_end - shard.input_begin)
                throw std::invalid_argument("invalid sharded tensor shard");
            if (axis == TensorParallelAxis::Input) {
                if (shard.input_begin != end || shard.output_begin != 0 || shard.output_end != out)
                    throw std::invalid_argument("invalid input-axis shards");
                end = shard.input_end;
            } else {
                if (shard.output_begin != end || shard.input_begin != 0 || shard.input_end != in)
                    throw std::invalid_argument("invalid output-axis shards");
                end = shard.output_end;
            }
        }
        if (end != (axis == TensorParallelAxis::Input ? in : out))
            throw std::invalid_argument("shards do not cover tensor extent");
    }

    bool defined() const noexcept override {
        for (const auto& shard : shards)
            if (!shard.weight || !shard.weight.defined()) return false;
        return true;
    }
    int64_t dim() const noexcept override { return 2; }
    int64_t numel() const override { return view_.numel(); }
    std::size_t nbytes() const override { return view_.nbytes(); }
    std::size_t backing_nbytes() const override { return backing_bytes(buffers()); }
    int64_t size(int64_t dimension) const override { return view_.size(dimension); }
    ScalarType scalar_type() const noexcept override { return view_.scalar_type; }
    const Device& device() const noexcept override { return view_.device; }
    bool is_contiguous() const noexcept override { return false; }
    bool is_encoded() const noexcept override { return true; }
    std::vector<const DenseTensor*> buffers() const override {
        std::vector<const DenseTensor*> result;
        for (const auto& shard : shards) {
            if (!shard.weight) continue;
            auto part = shard.weight.buffers();
            result.insert(result.end(), part.begin(), part.end());
        }
        return result;
    }
    void record_stream(std::uintptr_t stream) const override {
        for (const auto& shard : shards)
            if (!shard.weight || shard.device != shards.front().device)
                throw std::invalid_argument("cannot record one stream for multiple devices");
        for (const auto& shard : shards) shard.weight.record_stream(stream);
    }
    std::shared_ptr<TensorImpl> relocated(Device) const override {
        auto moved = shards;
        for (auto& shard : moved)
            shard.weight = shard.weight.relocated(Device(kCUDA, shard.device));
        return std::make_shared<ShardedTensor>(size(0), size(1), scalar_type(), axis, std::move(moved));
    }

    TensorParallelAxis axis;
    std::vector<Shard> shards;

private:
    TensorView view_;
};

} // namespace tyloi
