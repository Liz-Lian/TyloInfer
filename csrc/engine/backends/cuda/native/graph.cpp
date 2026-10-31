#include "env.h"
#include "graph.h"
#include "device.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mfq::cuda {

Graph::~Graph() noexcept {
    reset();
}

Graph::Graph(Graph&& other) noexcept
    : device_(other.device_),
      stream_(std::move(other.stream_)),
      context_(std::move(other.context_)),
      pool_streams_(std::move(other.pool_streams_)),
      pool_contexts_(std::move(other.pool_contexts_)),
      graph_(std::exchange(other.graph_, nullptr)),
      executable_(std::exchange(other.executable_, nullptr)),
      shared_pool_(std::exchange(other.shared_pool_, false)) {}

Graph& Graph::operator=(Graph&& other) noexcept {
    if (this != &other) {
        reset();
        device_ = other.device_;
        stream_ = std::move(other.stream_);
        context_ = std::move(other.context_);
        pool_streams_ = std::move(other.pool_streams_);
        pool_contexts_ = std::move(other.pool_contexts_);
        graph_ = std::exchange(other.graph_, nullptr);
        executable_ = std::exchange(other.executable_, nullptr);
        shared_pool_ = std::exchange(other.shared_pool_, false);
    }
    return *this;
}

void Graph::prepare_memory() {
    prepare_memory({});
}

void Graph::prepare_memory(
        const std::vector<StreamHandle>& participant_streams) {
    reset();
    auto stream = current_stream();
    auto context = default_context(stream.device_index());
    stream_ = std::move(stream);
    device_ = stream_.device_index();
    context_ = std::move(context);
    const auto add_pool = [&](const StreamHandle& candidate) {
        if (!candidate) return;
        const auto duplicate = std::find_if(
            pool_streams_.begin(), pool_streams_.end(),
            [&](const StreamHandle& existing) {
                return existing.device_index() == candidate.device_index() &&
                    existing.stream() == candidate.stream();
            });
        if (duplicate != pool_streams_.end()) return;
        auto participant_context =
            default_context(candidate.device_index());
        if (!participant_context->supports_async_allocations()) {
            throw Error(
                "CUDA graph participant does not support async allocations");
        }
        participant_context->begin_graph_pool(candidate.stream());
        participant_context->begin_graph_warmup(candidate.stream());
        pool_streams_.push_back(candidate);
        pool_contexts_.push_back(std::move(participant_context));
    };
    try {
        add_pool(stream_);
        for (const auto& participant : participant_streams) {
            add_pool(participant);
        }
    } catch (...) {
        reset();
        throw;
    }
}

void Graph::capture_begin() {
    if (!stream_) {
        prepare_memory();
    }
    const auto active = current_stream();
    if (active.device_index() != device_ ||
            active.stream() != stream_.stream()) {
        throw Error(
            "CUDA graph capture stream differs from its prepared memory stream");
    }
    DeviceGuard guard(device_);
    try {
        for (std::size_t index = 0; index < pool_streams_.size(); ++index) {
            pool_contexts_[index]->begin_graph_capture(
                pool_streams_[index].stream());
        }
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(
            stream_.stream(), cudaStreamCaptureModeThreadLocal));
    } catch (...) {
        for (std::size_t index = 0; index < pool_streams_.size(); ++index) {
            pool_contexts_[index]->end_graph_capture(
                pool_streams_[index].stream());
        }
        throw;
    }
}

void Graph::capture_begin_shared() {
    if (stream_ || graph_ || executable_) throw Error("graph already owns a capture");
    auto stream = current_stream();
    auto context = default_context(stream.device_index());
    stream_ = std::move(stream);
    device_ = stream_.device_index();
    context_ = std::move(context);
    shared_pool_ = true;
    context_->begin_graph_capture(stream_.stream());
    try {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(
            stream_.stream(), cudaStreamCaptureModeThreadLocal));
    } catch (...) {
        context_->end_graph_capture(stream_.stream());
        throw;
    }
}

std::size_t Graph::nodes() const {
    if (!graph_) return 0;
    DeviceGuard guard(device_);
    std::size_t count = 0;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &count));
    return count;
}

void Graph::capture_end() {
    if (!stream_) {
        throw Error("CUDA graph capture_end called without capture_begin");
    }
    DeviceGuard guard(device_);
    const auto capture_status =
        cudaStreamEndCapture(stream_.stream(), &graph_);
    if (shared_pool_) context_->end_graph_capture(stream_.stream());
    for (std::size_t index = 0; index < pool_streams_.size(); ++index) {
        pool_contexts_[index]->end_graph_capture(
            pool_streams_[index].stream());
    }
    TYLOINFER_NATIVE_CUDA_CHECK(capture_status);
    if (const char* dot_path = mfq::cuda::env_lookup("TYLOINFER_NATIVE_CUDA_GRAPH_DOT");
        dot_path != nullptr && dot_path[0] != '\0') {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphDebugDotPrint(
            graph_, dot_path, cudaGraphDebugDotFlagsVerbose));
    }
    if (const char* trace = mfq::cuda::env_lookup("TYLOINFER_TRACE_NATIVE_CUDA_GRAPH");
        trace != nullptr && std::atoi(trace) != 0) {
        std::size_t count = 0;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &count));
        std::vector<cudaGraphNode_t> nodes(count);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphGetNodes(graph_, nodes.data(), &count));
        std::unordered_set<void*> allocations;
        std::unordered_set<void*> frees;
        std::unordered_set<void*> reported_kernels;
        const char* resources=mfq::cuda::env_lookup("TYLOINFER_TRACE_NATIVE_CUDA_KERNEL_RESOURCES");
        std::size_t kernels = 0;
        std::size_t copies = 0;
        for (auto node : nodes) {
            cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
            if (type == cudaGraphNodeTypeKernel) {
                ++kernels;
                if(resources && std::atoi(resources)!=0) {
                    cudaKernelNodeParams params{};
                    TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&params));
                    if(reported_kernels.insert(params.func).second) {
                        cudaFuncAttributes attributes{};
                        TYLOINFER_NATIVE_CUDA_CHECK(cudaFuncGetAttributes(&attributes,params.func));
                        std::cerr<<"native_cuda_kernel function="<<params.func
                            <<" grid="<<params.gridDim.x<<','<<params.gridDim.y<<','<<params.gridDim.z
                            <<" block="<<params.blockDim.x<<','<<params.blockDim.y<<','<<params.blockDim.z
                            <<" registers="<<attributes.numRegs<<" local_bytes="<<attributes.localSizeBytes
                            <<" static_shared_bytes="<<attributes.sharedSizeBytes
                            <<" dynamic_shared_bytes="<<params.sharedMemBytes<<'\n';
                    }
                }
            } else if (type == cudaGraphNodeTypeMemcpy) {
                ++copies;
            } else if (type == cudaGraphNodeTypeMemAlloc) {
                cudaMemAllocNodeParams params{};
                TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphMemAllocNodeGetParams(node, &params));
                allocations.insert(reinterpret_cast<void*>(params.dptr));
            } else if (type == cudaGraphNodeTypeMemFree) {
                void* pointer = nullptr;
                TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphMemFreeNodeGetParams(node, &pointer));
                frees.insert(pointer);
            }
        }
        std::size_t allocation_only = 0;
        std::size_t free_only = 0;
        for (auto pointer : allocations) {
            allocation_only += frees.contains(pointer) ? 0 : 1;
        }
        for (auto pointer : frees) {
            free_only += allocations.contains(pointer) ? 0 : 1;
        }
        std::cerr << "native_cuda_graph nodes=" << count
                  << " kernels=" << kernels
                  << " copies=" << copies
                  << " allocations=" << allocations.size()
                  << " frees=" << frees.size()
                  << " allocation_only=" << allocation_only
                  << " free_only=" << free_only << '\n';
    }
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&executable_, graph_, 0));
    // A first replay may block in the driver while it uploads a large graph.
    // Prepare it before replay can wait for CPU-produced mapped flags.
    if (const char* upload = mfq::cuda::env_lookup("TYLOINFER_NATIVE_CUDA_GRAPH_UPLOAD");
        upload == nullptr || std::atoi(upload) != 0) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphUpload(executable_, stream_.stream()));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
        context_->check_memory_limit();
    }
}

void Graph::replay() {
    if (executable_ == nullptr || !stream_) {
        throw Error("CUDA graph replay called before capture");
    }
    DeviceGuard guard(device_);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(executable_, stream_.stream()));
}

void Graph::reset() noexcept {
    internal::on_device_noexcept(device_, [&] {
        if (stream_) {
            cudaStreamCaptureStatus capture_status =
                cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(
                    stream_.stream(), &capture_status) == cudaSuccess &&
                    capture_status != cudaStreamCaptureStatusNone) {
                cudaGraph_t abandoned = nullptr;
                if (cudaStreamEndCapture(
                        stream_.stream(), &abandoned) == cudaSuccess &&
                        abandoned != nullptr) {
                    (void)cudaGraphDestroy(abandoned);
                } else {
                    (void)cudaGetLastError();
                }
            }
            (void)cudaStreamSynchronize(stream_.stream());
        }
        if (executable_ != nullptr) {
            (void)cudaGraphExecDestroy(executable_);
            executable_ = nullptr;
        }
        if (graph_ != nullptr) {
            (void)cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        for (std::size_t index = 0; index < pool_streams_.size(); ++index) {
            pool_contexts_[index]->end_graph_capture(
                pool_streams_[index].stream());
            pool_contexts_[index]->end_graph_pool(
                pool_streams_[index].stream());
        }
        if (shared_pool_ && context_) context_->end_graph_capture(stream_.stream());
    });
    pool_contexts_.clear();
    pool_streams_.clear();
    context_.reset();
    stream_ = {};
    shared_pool_ = false;
}

}  // namespace mfq::cuda
