#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/native/stream.h"

#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace mfq::cuda {

inline bool decode_branches(bool enabled, bool serial, int64_t rows) {
    // Graph allocations stay on the capture stream until branch lifetime tracking supports them.
    return rows == 1 && !serial && enabled;
}

// One serialized execution lane; nested calls borrow different stream sets.
class Branches {
    struct Branch {
        MfqCudaStream stream;
        Event done;
    };
    struct Set {
        std::optional<Event> ready;
        std::vector<Branch> branches;

        bool prepare(std::size_t count, const MfqCudaStream& parent) {
            if (branches.size() >= count) return true;
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(parent.stream(), &capture));
            if (capture != cudaStreamCaptureStatusNone) return false;
            if (!ready) ready.emplace();
            while (branches.size() < count)
                branches.push_back({mfq_get_stream_from_pool(false, parent.device_index()), Event{}});
            return true;
        }
    };
    std::map<int, std::vector<std::unique_ptr<Set>>> devices_;
    std::size_t depth_ = 0;

public:
    Branches() = default;
    Branches(const Branches&) = delete;
    Branches& operator=(const Branches&) = delete;

    bool empty() const noexcept { return devices_.empty(); }

    void drain() {
        std::exception_ptr failure;
        for (const auto& [device, sets] : devices_) {
            for (const auto& set : sets) {
                for (const auto& branch : set->branches) {
                    try {
                        const MfqCudaGuard guard(device);
                        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(branch.stream.stream()));
                    } catch (...) {
                        if (!failure) failure = std::current_exception();
                    }
                }
            }
        }
        if (failure) std::rethrow_exception(failure);
    }

    // The execution owner drains before clearing or destroying these resources.
    void clear() {
        TYLOINFER_RUNTIME_CHECK(depth_ == 0, "cannot retire active CUDA branches");
        devices_.clear();
    }

    template<class Fn>
    bool run(std::size_t count, Fn&& fn, std::vector<mfq_tensor_backend::Tensor>& outputs) {
        if (count < 2) return false;
        const auto parent = mfq_get_current_cuda_stream();
        auto& sets = devices_[parent.device_index()];
        while (sets.size() <= depth_) sets.push_back(std::make_unique<Set>());
        auto& set = *sets[depth_];
        if (!set.prepare(count, parent)) return false;
        struct Scope {
            std::size_t& depth;
            explicit Scope(std::size_t& value) : depth(value) { ++depth; }
            ~Scope() { --depth; }
        } scope(depth_);

        outputs.resize(count);
        set.ready->record(parent.stream());
        std::exception_ptr failure;
        std::size_t started = 0;
        for (std::size_t i = 0; i < count; ++i) {
            auto& branch = set.branches[i];
            try {
                TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(branch.stream.stream(), set.ready->get(), 0));
                ++started;
                const MfqCudaStreamGuard guard(branch.stream);
                outputs[i] = fn(i);
            } catch (...) {
                failure = std::current_exception();
                break;
            }
        }
        // Join even a partially submitted branch before propagating its error.
        for (std::size_t i = 0; i < started; ++i) {
            auto& branch = set.branches[i];
            try {
                branch.done.record(branch.stream.stream());
                TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(parent.stream(), branch.done.get(), 0));
                if (outputs[i].defined()) mfq_cuda_record_stream(outputs[i], parent);
            } catch (...) {
                if (!failure) failure = std::current_exception();
            }
        }
        if (failure) std::rethrow_exception(failure);
        return true;
    }
};

} // namespace mfq::cuda
