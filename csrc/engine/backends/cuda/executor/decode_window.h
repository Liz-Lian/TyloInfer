#pragma once
#include "engine/backends/cuda/executor/graph_registry.h"
#include "engine/backends/cuda/kernels/residual_trace.h"
#include "engine/ops/residual.h"
#include <cuda_runtime_api.h>
#include <exception>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace tyloi::weights { class Cache; }
namespace tyloi::cuda {
class Graph;
void wait_route_publication(const uint32_t* flag,cudaStream_t stream,int timeout_ms=60000);
void publish_mapped_flag(uint32_t* flag) noexcept;
void check_stream_progress(cudaStream_t stream);
void check_route_submission(cudaStream_t stream);
// Resident KV uses one model graph. Streamed KV adds boundaries at host I/O;
// expert operators within each segment keep their GPU body and host tasks in
// one submission. Disk reads run after a segment completes, without GPU spins.
class DecodeWindow {
public:
    struct Task {
        const void* identity;
        std::function<void()> reset, serve, finish, cancel;
        std::function<void()> cleanup;
        std::function<void()> release_warm_output;
    };
    struct GpuPhase {
        int from_layer,layer;
        std::string from,to;
        uint64_t samples;
        int64_t total_ns,min_ns,max_ns;
        int from_tile=-1,tile=-1;
    };
    /** When eager, prepare capture memory before initializing profiling storage. */
    explicit DecodeWindow(cudaStream_t stream,bool layer_profile=false,bool independent_sequences=false,bool eager=false);
    bool independent_sequences() const noexcept {return independent_sequences_;}
    ~DecodeWindow();
    DecodeWindow(const DecodeWindow&)=delete;
    DecodeWindow& operator=(const DecodeWindow&)=delete;
    /** Bind a single-request window to the active execution stream. */
    DecodeWindow();
    /** Retain the caller's stream and a ready event for independent requests. */
    static std::unique_ptr<DecodeWindow> batch();
    /** Reject a failed window or a changed execution stream. */
    void validate() const;
    /** Hold the window stream on the current thread for the scope lifetime. */
    class Scope {
    public:
        virtual ~Scope() = default;
    };
    /** Enter the owned stream, optionally waiting for the caller's current stream. */
    std::unique_ptr<Scope> enter(bool wait = true);
    /** Retire captures and scratch while retaining the stream and event owner. */
    void clear() noexcept;
    /** Prepare graph weights or own a fallback pool before snapshot allocation. */
    void prepare();
    /** Drive preparation, two warmups and recording on the entering thread. */
    class Capture {
    public:
        ~Capture();
        Capture(const Capture&) = delete;
        Capture& operator=(const Capture&) = delete;
        /** Begin the next numerical pass; a valid window requires no passes. */
        bool next();
        /** Report whether the current pass requires releasing warm outputs. */
        bool warmup() const;
        /** End the submitted pass before the caller restores recurrent state. */
        void submitted();
        /** Finish the pass after the caller restores state and releases outputs. */
        void restored();
        /** Cancel a failed capture and return the exception to rethrow after restoration. */
        std::exception_ptr fail(std::exception_ptr error);
    private:
        friend class DecodeWindow;
        explicit Capture(DecodeWindow& window);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    /** Create an owner for the explicit capture phases. */
    std::unique_ptr<Capture> capture();
    void enroll(Task task);
    // Finish the current GPU segment before running host I/O. During graph
    // preparation only the supplied placeholder buffers are consumed.
    void host_barrier(std::function<void()> serve);
    // Update the next segment at a completed host barrier, before its launch.
    // Zero disables a kernel while preserving the captured dependency edges.
    void set_kernel_rows(std::size_t segment,cudaGraphNode_t node,unsigned rows);
    template<class T,class Factory> std::shared_ptr<T> workspace(const void* identity,Factory create) {
        auto& owner=workspaces_[identity];if(!owner)owner=create();return std::static_pointer_cast<T>(owner);
    }
    void run();
    void mark(int layer,const char* phase);
    void mark_current(const char* phase) {mark(profile_layer_,phase);}
    /** Prepare fused residual profiling points without recording them. */
    ResidualTraceData prepare_residual_trace(tyloi::ops::ResidualRole role);
    /** Publish the points actually recorded by the fused launch. */
    void commit_residual_trace(const ResidualTraceData& trace) noexcept;
    /** Reject a fused launch that published more profiling points than the clock supports. */
    void validate_residual_trace() const;
    int set_profile_tile(int tile) noexcept {const int previous=profile_tile_;profile_tile_=tile;return previous;}
    void set_profile_geometry(const char* role,int rows) {profile_role_=role;profile_rows_=rows;}
    void set_profile_layer_kind(int layer,const char* kind) {if(layer_profile_)profile_kinds_[layer]=kind;}
    std::vector<GpuPhase> gpu_phases() const;
    std::size_t nodes() const;
    std::size_t segments() const {return barriers_.size()+1;}
    bool valid() const { const auto* graph=graphs_.find(0,1);return graph && graph->valid() && !recording_segments_; }
    static DecodeWindow* recording() noexcept;
private:
    struct Context;
    explicit DecodeWindow(std::unique_ptr<Context> context);
    std::unique_ptr<Context> context_;
    void initialize_profile();
    cudaStream_t stream_;
    GraphRegistry graphs_;
    ::tyloi::weights::Cache* weights_ = nullptr;
    std::unique_ptr<Graph> memory_pool_;
    bool prepared_ = false;
    struct LayerPoint;
    struct ProfileClock;
    std::unique_ptr<ProfileClock> profile_clock_;
    std::vector<std::unique_ptr<LayerPoint>> layer_points_;
    std::vector<std::unique_ptr<LayerPoint>> pending_points_;
    std::vector<std::size_t> free_profile_slots_;
    std::size_t next_profile_slot_ = 0;
    LayerPoint* declare_point(int layer, const char* phase);
    void publish_point(LayerPoint* point) noexcept;
    std::map<std::tuple<int,int,std::string>,LayerPoint*> layer_point_index_;
    int profile_tile_=-1;
    int profile_layer_=-1;
    bool layer_profile_;
    bool independent_sequences_;
    std::string profile_role_="unknown";
    int profile_rows_=1;
    std::map<int,std::string> profile_kinds_;
    void observe_layers();
    std::vector<Task> tasks_;
    std::map<const void*,std::shared_ptr<void>> workspaces_;
    struct Barrier {std::size_t task_end;std::function<void()> serve;};
    std::vector<Barrier> barriers_;
    bool recording_segments_=false;
    bool failed_=false;
    int64_t capture_ns_=0,replay_ns_=0,reset_ns_=0,launch_ns_=0,serve_ns_=0,wait_ns_=0,finish_ns_=0,profile_ns_=0;
    uint64_t replays_=0;
    uint64_t dropped_profiles_=0;
};
}
