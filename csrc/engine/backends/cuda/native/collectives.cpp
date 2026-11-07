#include "collectives.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#ifdef TYLOINFER_HAVE_NCCL
#ifndef TYLOINFER_NCCL_CHECK
#define TYLOINFER_NCCL_CHECK(expr) do { \
    ncclResult_t err__ = (expr); \
    if (err__ != ncclSuccess) { \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(err__)); \
    } \
} while (0)
#endif
#endif

namespace tyloi::cuda {

Collectives::~Collectives() {
    reset();
}

void Collectives::drain() const {
    std::exception_ptr failure;
    for (const auto& stream : streams_) {
        try {
            MfqCudaGuard guard(stream.device_index());
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
    }
    if (failure) std::rethrow_exception(failure);
}

void Collectives::reset() noexcept {
    // Release buffers before destroying their device contexts.
    for (int device : devices_) {
        (void)cudaSetDevice(device);
        (void)cudaDeviceSynchronize();
    }
    reduction_buffers_.clear();
#ifdef TYLOINFER_HAVE_NCCL
    for (auto communicator : communicators_) {
        if (communicator != nullptr) (void)ncclCommDestroy(communicator);
    }
    communicators_.clear();
#endif
    for (size_t index = 0; index < devices_.size(); ++index) {
        (void)cudaSetDevice(devices_[index]);
        if (index < ready_.size() && ready_[index] != nullptr) {
            (void)cudaEventDestroy(ready_[index]);
        }
        if (index < completed_.size() && completed_[index] != nullptr) {
            (void)cudaEventDestroy(completed_[index]);
        }
    }
    devices_.clear();
    streams_.clear();
    ready_.clear();
    completed_.clear();
    collectives_enabled_ = false;
}

void Collectives::enable_peer_access(const std::vector<int>& devices) {
    std::unordered_set<int> unique_devices(
        devices.begin(), devices.end());
    for (const int source : unique_devices) {
        for (const int destination : unique_devices) {
            if (source == destination) continue;
            int can_access = 0;
            TYLOINFER_CUDA_CHECK(cudaDeviceCanAccessPeer(
                &can_access, source, destination));
            if (!can_access) continue;
            TYLOINFER_CUDA_CHECK(cudaSetDevice(source));
            const cudaError_t status = cudaDeviceEnablePeerAccess(
                destination, 0);
            if (status != cudaSuccess &&
                    status != cudaErrorPeerAccessAlreadyEnabled) {
                throw std::runtime_error(
                    "failed to enable model-parallel peer access from CUDA " +
                    std::to_string(source) + " to CUDA " +
                    std::to_string(destination) + ": " +
                    cudaGetErrorString(status));
            }
            if (status == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();
            }
        }
    }
}

void Collectives::configure(
        const std::vector<int>& requested_devices,
        bool allow_duplicate_devices) {
    reset();
    if (requested_devices.size() < 2 || allow_duplicate_devices) return;

    devices_ = requested_devices;
    streams_.reserve(devices_.size());
    ready_.resize(devices_.size(), nullptr);
    completed_.resize(devices_.size(), nullptr);
    reduction_buffers_.resize(devices_.size());
    for (size_t index = 0; index < devices_.size(); ++index) {
        MfqCudaGuard guard(devices_[index]);
        streams_.push_back(mfq_get_stream_from_pool(false, devices_[index]));
        TYLOINFER_CUDA_CHECK(cudaEventCreateWithFlags(
            &ready_[index], cudaEventDisableTiming));
        TYLOINFER_CUDA_CHECK(cudaEventCreateWithFlags(
            &completed_[index], cudaEventDisableTiming));
    }
#ifdef TYLOINFER_HAVE_NCCL
    communicators_.resize(devices_.size(), nullptr);
    TYLOINFER_NCCL_CHECK(ncclCommInitAll(
        communicators_.data(),
        static_cast<int>(devices_.size()),
        devices_.data()));
    // Warm both directions before graph capture triggers NCCL's lazy setup.
    std::vector<::tyloi::DenseTensor> p2p_warmup_buffers;
    p2p_warmup_buffers.reserve(devices_.size());
    for (const int device : devices_) {
        MfqCudaGuard guard(device);
        p2p_warmup_buffers.push_back(::tyloi::cuda::empty(
            {1}, ::tyloi::TensorOptions()
                .device(::tyloi::Device(
                    ::tyloi::kCUDA, device))
                .dtype(::tyloi::kUInt8)));
    }
    for (size_t peer = 1; peer < devices_.size(); ++peer) {
        TYLOINFER_NCCL_CHECK(ncclGroupStart());
        TYLOINFER_NCCL_CHECK(ncclSend(
            p2p_warmup_buffers[0].data_ptr(), 1, ncclUint8,
            static_cast<int>(peer), communicators_[0], streams_[0].stream()));
        TYLOINFER_NCCL_CHECK(ncclRecv(
            p2p_warmup_buffers[peer].data_ptr(), 1, ncclUint8,
            0, communicators_[peer], streams_[peer].stream()));
        TYLOINFER_NCCL_CHECK(ncclSend(
            p2p_warmup_buffers[peer].data_ptr(), 1, ncclUint8,
            0, communicators_[peer], streams_[peer].stream()));
        TYLOINFER_NCCL_CHECK(ncclRecv(
            p2p_warmup_buffers[0].data_ptr(), 1, ncclUint8,
            static_cast<int>(peer), communicators_[0], streams_[0].stream()));
        TYLOINFER_NCCL_CHECK(ncclGroupEnd());
    }
    for (size_t index = 0; index < devices_.size(); ++index) {
        MfqCudaGuard guard(devices_[index]);
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(streams_[index].stream()));
    }
    collectives_enabled_ = true;
#endif
}

::tyloi::DenseTensor Collectives::copy(::tyloi::DenseTensor value, int device,
                                                  ::tyloi::DenseTensor reusable) {
    if (value.is_cuda() && value.get_device() == device) {
        return ::tyloi::cuda::contiguous(value);
    }
    const auto reusable_matches = [&]() {
        return reusable.defined() && reusable.is_cuda() &&
            reusable.get_device() == device &&
            reusable.sizes() == value.sizes() &&
            reusable.scalar_type() == value.scalar_type() &&
            reusable.is_contiguous();
    };
    const auto destination_tensor = [&]() {
        if (reusable_matches()) return reusable;
        MfqCudaGuard destination_guard(device);
        return ::tyloi::cuda::empty(
            value.sizes(),
            value.options().device(
                ::tyloi::Device(
                    ::tyloi::kCUDA, device)));
    };
    if (value.numel() == 0) {
        return destination_tensor();
    }
#if defined(TYLOINFER_NATIVE_CUDA) && defined(TYLOINFER_HAVE_NCCL)
    auto& runtime = *this;
    if (value.is_cuda() && runtime.collectives_enabled_) {
        const int source_device = value.get_device();
        const auto source_rank_it = std::find(
            runtime.devices_.begin(), runtime.devices_.end(), source_device);
        const auto destination_rank_it = std::find(
            runtime.devices_.begin(), runtime.devices_.end(), device);
        if (source_rank_it != runtime.devices_.end() &&
                destination_rank_it != runtime.devices_.end()) {
            const auto source_stream =
                mfq_get_current_cuda_stream(source_device);
            cudaStreamCaptureStatus capture_status =
                cudaStreamCaptureStatusNone;
            {
                MfqCudaGuard source_guard(source_device);
                TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(
                    source_stream.stream(), &capture_status));
            }
            if (capture_status != cudaStreamCaptureStatusNone) {
                auto source = ::tyloi::cuda::contiguous(value);
                auto destination = destination_tensor();
                const auto source_rank = static_cast<size_t>(
                    source_rank_it - runtime.devices_.begin());
                const auto destination_rank = static_cast<size_t>(
                    destination_rank_it - runtime.devices_.begin());
                {
                    MfqCudaGuard source_guard(source_device);
                    TYLOINFER_CUDA_CHECK(cudaEventRecord(
                        runtime.ready_[source_rank],
                        source_stream.stream()));
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
                        runtime.streams_[source_rank].stream(),
                        runtime.ready_[source_rank], 0));
                }
                {
                    MfqCudaGuard destination_guard(device);
                    // Cross-device event waits make the receiving NCCL stream
                    // participate in the same capture before ncclRecv runs.
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
                        runtime.streams_[destination_rank].stream(),
                        runtime.ready_[source_rank], 0));
                }
                TYLOINFER_NCCL_CHECK(ncclGroupStart());
                TYLOINFER_NCCL_CHECK(ncclSend(
                    source.data_ptr(), source.nbytes(), ncclUint8,
                    static_cast<int>(destination_rank),
                    runtime.communicators_[source_rank],
                    runtime.streams_[source_rank].stream()));
                TYLOINFER_NCCL_CHECK(ncclRecv(
                    destination.data_ptr(), destination.nbytes(), ncclUint8,
                    static_cast<int>(source_rank),
                    runtime.communicators_[destination_rank],
                    runtime.streams_[destination_rank].stream()));
                TYLOINFER_NCCL_CHECK(ncclGroupEnd());
                {
                    MfqCudaGuard source_guard(source_device);
                    TYLOINFER_CUDA_CHECK(cudaEventRecord(
                        runtime.completed_[source_rank],
                        runtime.streams_[source_rank].stream()));
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
                        source_stream.stream(),
                        runtime.completed_[source_rank], 0));
                }
                {
                    MfqCudaGuard destination_guard(device);
                    const auto destination_stream =
                        mfq_get_current_cuda_stream(device);
                    TYLOINFER_CUDA_CHECK(cudaEventRecord(
                        runtime.completed_[destination_rank],
                        runtime.streams_[destination_rank].stream()));
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
                        destination_stream.stream(),
                        runtime.completed_[destination_rank], 0));
                }
                return destination;
            }
        }
    }
#endif
    MfqCudaGuard guard(device);
    if (reusable_matches()) {
        ::tyloi::cuda::copy_into(reusable, value);
        return reusable;
    }
    return ::tyloi::cuda::contiguous(
        ::tyloi::cuda::copy(value, ::tyloi::Device(::tyloi::kCUDA, device)));
}

::tyloi::DenseTensor Collectives::sum(
        std::vector<::tyloi::DenseTensor> outputs, int primary,
        bool reduce_to_primary, bool allow_fp16) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "cannot reduce an empty model-parallel output");
    }
#ifdef TYLOINFER_HAVE_NCCL
    auto& runtime = *this;
    if (runtime.collectives_enabled_ &&
            outputs.size() == runtime.devices_.size()) {
        const auto shape = outputs.front().sizes().vec();
        const auto output_dtype = outputs.front().scalar_type();
        const int64_t elements = outputs.front().numel();
        // A two-input FP16 sum has the same final FP16 rounding as the former
        // FP32 reduction, while avoiding both conversion passes.
        const bool fp16_reduce =
            allow_fp16 &&
            reduce_to_primary &&
            outputs.size() == 2 &&
            output_dtype == ::tyloi::kFloat16;
        const auto primary_rank_it = std::find(
            runtime.devices_.begin(), runtime.devices_.end(), primary);
        if (primary_rank_it == runtime.devices_.end()) {
            throw std::runtime_error(
                "model-parallel primary device is absent from NCCL ranks");
        }
        const auto primary_rank = static_cast<size_t>(
            primary_rank_it - runtime.devices_.begin());
        for (size_t index = 0; index < outputs.size(); ++index) {
            const int device = runtime.devices_[index];
            if (!outputs[index].defined() || !outputs[index].is_cuda() ||
                    outputs[index].get_device() != device ||
                    outputs[index].sizes().vec() != shape ||
                    outputs[index].scalar_type() != output_dtype) {
                throw std::runtime_error(
                    "NCCL model-parallel reduction received mismatched shards");
            }
            MfqCudaGuard guard(device);
            if (fp16_reduce) {
                outputs[index] = ::tyloi::cuda::contiguous(outputs[index]);
            }
            const auto producer =
                mfq_get_current_cuda_stream(device);
            TYLOINFER_CUDA_CHECK(cudaEventRecord(
                runtime.ready_[index], producer.stream()));
            const auto communication = runtime.streams_[index];
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
                communication.stream(), runtime.ready_[index], 0));
            if (fp16_reduce) {
                mfq_cuda_record_stream(outputs[index], communication);
            } else {
                MfqCudaStreamGuard stream_guard(communication);
                auto & buffer = runtime.reduction_buffers_[index];
                if (!buffer.defined() || buffer.sizes().vec() != shape ||
                        buffer.get_device() != device ||
                        buffer.scalar_type() != ::tyloi::kFloat32) {
                    buffer = ::tyloi::cuda::empty(
                        shape,
                        outputs[index].options().dtype(::tyloi::kFloat32));
                }
                ::tyloi::cuda::copy_into(buffer, outputs[index]);
                mfq_cuda_record_stream(outputs[index], communication);
            }
        }

        TYLOINFER_NCCL_CHECK(ncclGroupStart());
        for (size_t index = 0; index < outputs.size(); ++index) {
            auto & buffer = runtime.reduction_buffers_[index];
            if (reduce_to_primary) {
                void * reduction_data = fp16_reduce
                    ? outputs[index].data_ptr()
                    : buffer.data_ptr<float>();
                TYLOINFER_NCCL_CHECK(ncclReduce(
                    reduction_data,
                    reduction_data,
                    static_cast<size_t>(elements),
                    fp16_reduce ? ncclFloat16 : ncclFloat32,
                    ncclSum,
                    static_cast<int>(primary_rank),
                    runtime.communicators_[index],
                    runtime.streams_[index].stream()));
            } else {
                TYLOINFER_NCCL_CHECK(ncclAllReduce(
                    buffer.data_ptr<float>(),
                    buffer.data_ptr<float>(),
                    static_cast<size_t>(elements),
                    ncclFloat32,
                    ncclSum,
                    runtime.communicators_[index],
                    runtime.streams_[index].stream()));
            }
        }
        TYLOINFER_NCCL_CHECK(ncclGroupEnd());

        ::tyloi::DenseTensor result;
        for (size_t index = 0; index < outputs.size(); ++index) {
            MfqCudaGuard guard(runtime.devices_[index]);
            const auto communication = runtime.streams_[index];
            if (runtime.devices_[index] == primary) {
                if (fp16_reduce) {
                    result = outputs[index];
                } else {
                    MfqCudaStreamGuard stream_guard(communication);
                    result = ::tyloi::cuda::contiguous(
                        ::tyloi::cuda::cast(runtime.reduction_buffers_[index], output_dtype));
                }
            }
            TYLOINFER_CUDA_CHECK(cudaEventRecord(
                runtime.completed_[index], communication.stream()));
        }
        MfqCudaGuard primary_guard(primary);
        const auto parent = mfq_get_current_cuda_stream(primary);
        TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(
            parent.stream(), runtime.completed_[primary_rank], 0));
        mfq_cuda_record_stream(result, parent);
        return result;
    }
#endif
    MfqCudaGuard primary_guard(primary);
    const auto output_dtype =
        outputs.front().scalar_type();
    ::tyloi::DenseTensor reduced;
    for (auto & output : outputs) {
        auto partial = ::tyloi::cuda::cast(copy(output, primary), ::tyloi::kFloat32);
        if (!reduced.defined()) {
            reduced = std::move(partial);
        } else {
            reduced.add_(partial);
        }
    }
    return ::tyloi::cuda::contiguous(::tyloi::cuda::cast(reduced, output_dtype));
}
} // namespace tyloi::cuda
