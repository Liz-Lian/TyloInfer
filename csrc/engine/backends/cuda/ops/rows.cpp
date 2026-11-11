#include "engine/weights/row_stage.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/decode_window.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/kernels/moe/pipeline.h"
#include "engine/backends/cuda/native/memory.h"
#include "engine/backends/cuda/native/stream.h"
#include "backend.h"
#include <chrono>
#include <cstring>
#include <iostream>
#endif
#include <stdexcept>

namespace tyloi::weights {
struct RowStage::Graph {
#ifdef TYLOINFER_NATIVE_CUDA
    cuda::HostBuffer flags{128, true};
    MfqCudaStream upload_stream;
    std::uint32_t* device = nullptr;
    bool charged = false;
    bool mapped = false;
    bool profile = false;
    std::uint64_t calls = 0, route_wait_ns = 0, upload_ns = 0, notify_ns = 0;
    explicit Graph(const Device& target)
        : upload_stream(mfq_get_stream_from_pool(false, target.index)) {
        const auto timing = cuda::options::trace_ple_timings();
        profile = timing && *timing;
        TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), flags.data(), 0));
        std::memset(flags.data(), 0, 128);
        cuda::publish_mapped_flag(host() + 16);
        cuda::charge_tensor_host_bytes(128);
        charged = true;
    }
    ~Graph() {
        (void)cudaStreamSynchronize(upload_stream.stream());
        if (profile)
            std::cerr << "ple_stage_host mapped_rows=" << mapped << " calls=" << calls << " route_wait_ns=" << route_wait_ns
                      << " upload_ns=" << upload_ns << " notify_ns=" << notify_ns << '\n';
        if (charged) cuda::tensor_host_bytes.fetch_sub(128);
    }
    std::uint32_t* host() const { return static_cast<std::uint32_t*>(flags.data()); }
#else
    explicit Graph(const Device&) {}
#endif
    std::vector<std::int64_t> ids, shape;
};

RowStage::RowStage() = default;
RowStage::~RowStage() = default;
void RowStage::init_graph(const Device& device) {
#ifdef TYLOINFER_NATIVE_CUDA
    const MfqCudaGuard guard(device);
    graph_ = std::make_unique<Graph>(device);
    graph_->mapped = mapped_rows();
#else
    (void)device;
#endif
}
void RowStage::prepare_graph() {
#ifdef TYLOINFER_NATIVE_CUDA
    if (!graph_) init_graph(output().device());
#else
    throw std::logic_error("graph row transfer requires native CUDA");
#endif
}
void RowStage::prefetch_rows(const std::vector<std::int64_t>&, const std::vector<std::int64_t>&) {}
void RowStage::set_rows(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) {
    if (!graph_) throw std::logic_error("row stage has no graph resources");
    graph_->ids = ids;
    graph_->shape = shape;
}
void RowStage::decode_graph() {
#ifdef TYLOINFER_NATIVE_CUDA
    auto* window = cuda::DecodeWindow::recording();
    if (!window || !graph_) throw std::logic_error("row stage requires an active graph window");
    auto stage = shared_from_this();
    const auto stream = mfq_current_cuda_stream();
    window->enroll({stage.get(),
        [stage] {
            std::memset(stage->graph_->flags.data(), 0, 128);
            stage->prefetch_rows(stage->graph_->ids, stage->graph_->shape);
        },
        [stage, stream] {
            using Clock = std::chrono::steady_clock;
            auto& graph = *stage->graph_;
            auto point = graph.profile ? Clock::now() : Clock::time_point{};
            const auto sample = [&](std::uint64_t& total) {
                if (!graph.profile) return;
                const auto now = Clock::now();
                total += std::chrono::duration_cast<std::chrono::nanoseconds>(now - point).count();
                point = now;
            };
            cuda::wait_route_publication(graph.host(), stream);
            sample(graph.route_wait_ns);
            const MfqCudaStreamGuard guard(graph.upload_stream);
            stage->upload(graph.ids, graph.shape);
            sample(graph.upload_ns);
            if (stage->mapped_rows()) cuda::publish_mapped_flag(graph.host() + 16);
            else {
                TYLOINFER_CUDA_CHECK(cudaLaunchHostFunc(graph.upload_stream.stream(),
                    [](void* value) { cuda::publish_mapped_flag(static_cast<std::uint32_t*>(value)); },
                    graph.host() + 16));
                cuda::check_stream_progress(graph.upload_stream.stream());
            }
            sample(graph.notify_ns);
            if (graph.profile) ++graph.calls;
        }, {},
        [stage] { cuda::publish_mapped_flag(stage->graph_->host() + 16); },
        [stage] { (void)cudaStreamSynchronize(stage->graph_->upload_stream.stream()); }});
    cuda::signal_mapped_flag(graph_->device, stream);
    cuda::wait_mapped_flag(graph_->device + 16, stream);
    decode();
#else
    throw std::logic_error("graph row transfer requires native CUDA");
#endif
}
} // namespace tyloi::weights
