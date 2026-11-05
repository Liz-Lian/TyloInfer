#pragma once

#include "mfq_cuda_gdn_replay_ops.h"
#include "runtime/recurrent_copy_batch.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace mfq::cuda {

// Reuse recorded rank-one updates across layers. The accepted prefix is a
// launch argument, so a new acceptance length never needs a new CUDA Graph.
class RecordedGdnReplayBatch {
    using Tensor = mfq_tensor_backend::Tensor;
public:
    struct Layer {
        Tensor qkv, updates, initial_convolution, convolution, recurrent;
        int key_heads = 0;
        bool transposed = false, tiled_heads = false;
    };

private:
    struct Span { const void* pointer; std::uint64_t bytes; };
    std::vector<Layer> layers_;
    std::vector<Tensor> scratch_;
    Tensor host_, descriptors_;
    int maximum_heads_ = 0, maximum_channels_ = 0, tokens_ = 0;
    cudaStream_t recorded_stream_ = nullptr;
    bool stream_recorded_ = false;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    std::vector<DmaCopy> copies_;
#endif

    static bool overlaps(Span a, Span b) {
        const auto x = reinterpret_cast<std::uintptr_t>(a.pointer);
        const auto y = reinterpret_cast<std::uintptr_t>(b.pointer);
        return x <= y ? y - x < a.bytes : x - y < b.bytes;
    }

    static bool compatible(const std::vector<Layer>& layers) {
        namespace tb = mfq_tensor_backend;
        if (layers.empty() || layers.size() > std::size_t(std::numeric_limits<int>::max())) return false;
        const auto& first = layers.front().qkv;
        if (!first.defined() || !first.is_cuda() || first.dim() != 3) return false;
        const auto real = [&](const Tensor& value) {
            return value.defined() && value.is_cuda() && value.device() == first.device() &&
                value.scalar_type() == tb::kFloat32 && value.is_contiguous();
        };
        std::vector<Span> writes, reads;
        for (const auto& layer : layers) {
            const auto& qkv = layer.qkv;
            if (!qkv.defined() || !qkv.is_cuda() || qkv.device() != first.device() ||
                qkv.dim() != 3 || qkv.size(0) != 1 || qkv.size(1) < 1 || qkv.size(1) > 8 ||
                qkv.size(1) != first.size(1) || qkv.size(2) < 1 ||
                qkv.size(2) > std::numeric_limits<int>::max() || qkv.stride(2) != 1 ||
                qkv.stride(1) < qkv.size(2) ||
                (qkv.scalar_type() != tb::kFloat16 && qkv.scalar_type() != tb::kFloat32) ||
                !real(layer.updates) || !real(layer.initial_convolution) ||
                !real(layer.convolution) || !real(layer.recurrent)) return false;
            const auto& updates = layer.updates;
            const auto& state = layer.recurrent;
            if (updates.dim() != 3 || updates.size(0) != qkv.size(1) || updates.size(1) < 1 ||
                updates.size(1) > std::numeric_limits<int>::max() || updates.size(2) != 257 ||
                layer.key_heads <= 0 || updates.size(1) % layer.key_heads != 0 ||
                qkv.size(2) != (2 * int64_t(layer.key_heads) + updates.size(1)) * 128 ||
                state.sizes().vec() != std::vector<int64_t>({1, updates.size(1), 128, 128}) ||
                layer.convolution.sizes().vec() != std::vector<int64_t>({1, 3, qkv.size(2)}) ||
                layer.initial_convolution.sizes() != layer.convolution.sizes()) return false;
            const auto bytes = qkv.scalar_type() == tb::kFloat32 ? 4u : 2u;
            const auto safe = std::uint64_t(std::numeric_limits<int64_t>::max()) / bytes;
            if (std::uint64_t(qkv.size(2)) > safe || (qkv.size(1) > 1 &&
                std::uint64_t(qkv.stride(1)) > (safe - qkv.size(2)) / (qkv.size(1) - 1))) return false;
            reads.push_back({qkv.data_ptr(), std::uint64_t(
                ((qkv.size(1) - 1) * qkv.stride(1) + qkv.size(2)) * bytes)});
            reads.push_back({updates.data_ptr(), std::uint64_t(updates.nbytes())});
            const Span conv{layer.convolution.data_ptr(), std::uint64_t(layer.convolution.nbytes())};
            const Span initial{layer.initial_convolution.data_ptr(), std::uint64_t(layer.initial_convolution.nbytes())};
            writes.push_back(conv);
            writes.push_back({state.data_ptr(), std::uint64_t(state.nbytes())});
            // Exact initial/output convolution aliases are safe. A source
            // must not overlap a different layer's output or any recurrence.
            for (std::size_t i = 0; i < layers.size(); ++i) {
                const auto& other = layers[i];
                if (other.convolution.defined() && other.convolution.is_cuda() &&
                    overlaps(initial, {other.convolution.data_ptr(), std::uint64_t(other.convolution.nbytes())}) &&
                    (&other != &layer || initial.pointer != conv.pointer)) return false;
                if (other.recurrent.defined() && other.recurrent.is_cuda() &&
                    overlaps(initial, {other.recurrent.data_ptr(), std::uint64_t(other.recurrent.nbytes())})) return false;
            }
        }
        for (std::size_t i = 0; i < writes.size(); ++i) {
            for (std::size_t j = 0; j < i; ++j) if (overlaps(writes[i], writes[j])) return false;
            for (const auto& read : reads) if (overlaps(writes[i], read)) return false;
        }
        return true;
    }

public:
    // Workspace may be shared by graph cells on the same ordered stream.
    // Each batch retains its allocations if a later shape replaces a slot.
    static std::unique_ptr<RecordedGdnReplayBatch> create(
            std::vector<Layer> layers, std::vector<Tensor>& workspace) {
        if (!compatible(layers)) return {};
        auto batch = std::unique_ptr<RecordedGdnReplayBatch>(new RecordedGdnReplayBatch);
        batch->layers_ = std::move(layers);
        batch->tokens_ = int(batch->layers_.front().qkv.size(1));
        const MfqCudaGuard guard(batch->layers_.front().qkv.device());
        std::vector<GdnReplayLayer> metadata;
        std::vector<Span> protected_storage;
        for (const auto& layer : batch->layers_) {
            protected_storage.push_back({layer.qkv.data_ptr(), std::uint64_t(
                ((layer.qkv.size(1) - 1) * layer.qkv.stride(1) + layer.qkv.size(2)) * layer.qkv.element_size())});
            for (const auto* tensor : {&layer.updates, &layer.initial_convolution,
                                      &layer.convolution, &layer.recurrent})
                protected_storage.push_back({tensor->data_ptr(), std::uint64_t(tensor->nbytes())});
        }
        metadata.reserve(batch->layers_.size());
        workspace.resize(std::max(workspace.size(), batch->layers_.size()));
        for (std::size_t i = 0; i < batch->layers_.size(); ++i) {
            const auto& layer = batch->layers_[i];
            auto& scratch = workspace[i];
            bool reusable = scratch.defined() && scratch.device() == layer.convolution.device() &&
                scratch.scalar_type() == layer.convolution.scalar_type() &&
                scratch.sizes() == layer.convolution.sizes() && scratch.is_contiguous();
            if (reusable) {
                const Span span{scratch.data_ptr(), std::uint64_t(scratch.nbytes())};
                for (const auto& owner : protected_storage) if (overlaps(span, owner)) reusable = false;
                for (const auto& owner : batch->scratch_)
                    if (overlaps(span, {owner.data_ptr(), std::uint64_t(owner.nbytes())})) reusable = false;
            }
            if (!reusable)
                scratch = mfq_tensor_backend::empty_like(layer.convolution);
            batch->scratch_.push_back(scratch);
            GdnReplayLayer descriptor{};
            descriptor.qkv = layer.qkv.data_ptr();
            descriptor.updates = layer.updates.data_ptr<float>();
            descriptor.convolution = layer.convolution.data_ptr<float>();
            descriptor.state = layer.recurrent.data_ptr<float>();
            descriptor.next_convolution = scratch.data_ptr<float>();
            descriptor.qkv_stride = layer.qkv.stride(1);
            descriptor.key_heads = layer.key_heads;
            descriptor.value_heads = int(layer.updates.size(1));
            descriptor.channels = int(layer.qkv.size(2));
            descriptor.tokens = batch->tokens_;
            descriptor.transposed = layer.transposed;
            descriptor.tiled_heads = layer.tiled_heads;
            descriptor.qkv_float = layer.qkv.scalar_type() == mfq_tensor_backend::kFloat32;
            batch->maximum_heads_ = std::max(batch->maximum_heads_, descriptor.value_heads);
            batch->maximum_channels_ = std::max(batch->maximum_channels_, descriptor.channels);
            metadata.push_back(descriptor);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            if (layer.convolution.data_ptr() != layer.initial_convolution.data_ptr())
                batch->copies_.push_back({layer.convolution.data_ptr(), layer.initial_convolution.data_ptr(),
                    std::size_t(layer.convolution.nbytes())});
#endif
        }
        const auto& example = batch->layers_.front().qkv;
        batch->host_ = mfq_tensor_backend::empty({int64_t(metadata.size() * sizeof(GdnReplayLayer))},
            example.options().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kUInt8));
        std::memcpy(batch->host_.data_ptr(), metadata.data(), batch->host_.nbytes());
        batch->descriptors_ = batch->host_.to(example.device());
        return batch;
    }

    void run(int64_t retained_tokens) {
        TYLOINFER_RUNTIME_CHECK(retained_tokens > 0 && retained_tokens <= tokens_,
            "recorded GDN replay prefix exceeds the verified window");
        const MfqCudaGuard guard(descriptors_.device());
        const auto stream = mfq_get_current_cuda_stream(descriptors_.get_device());
        // These owners live with the batch. Recording a stream once also
        // covers later submissions on it; no tensor allocation changes here.
        if (!stream_recorded_ || recorded_stream_ != stream.stream()) {
            for (const auto& layer : layers_)
                for (const auto* tensor : {&layer.qkv, &layer.updates, &layer.initial_convolution,
                                          &layer.convolution, &layer.recurrent})
                    mfq_cuda_record_stream(*tensor, stream);
            for (const auto& tensor : scratch_) mfq_cuda_record_stream(tensor, stream);
            mfq_cuda_record_stream(descriptors_, stream);
            recorded_stream_ = stream.stream();
            stream_recorded_ = true;
        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        const auto* copy_batch = std::getenv("TYLOINFER_RECURRENT_COPY_BATCH");
        if (!copy_batch || copy_batch[0] != '0') {
            if (!copies_.empty()) copy_recurrent_states(copies_.data(), copies_.size(), stream.stream());
        } else {
            for (auto& layer : layers_)
                if (layer.convolution.data_ptr() != layer.initial_convolution.data_ptr())
                    layer.convolution.copy_(layer.initial_convolution);
        }
#else
        for (auto& layer : layers_) layer.convolution.copy_(layer.initial_convolution);
#endif
        gdn_replay_commit_layers_cuda(descriptors_, int(layers_.size()), maximum_heads_,
            maximum_channels_, int(retained_tokens), true);
    }
};
} // namespace mfq::cuda
