#pragma once
#include "gguf_cpu.h"
#include "tyloinfer/host_parallel.h"
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <vector>
struct GgufInputRotation;

// Only original compressed routed experts are accepted here. Storage owners
// keep their RAM images alive until the asynchronous CPU work has completed.
struct GgufCpuExpert {
    std::array<mfq::cpu::gguf::Projection,3> projections;
    std::array<std::shared_ptr<void>,3> owners;
    std::vector<int> slots;
};
struct GgufCpuWork {
    int tokens=0,routes=0,input_width=0,hidden_width=0,output_width=0;
    bool half=false;
    float limit=0;
    const float* input=nullptr;
    const float* up_input=nullptr;
    std::shared_ptr<const GgufInputRotation> down_rotation;
    float* output=nullptr;
    std::uint32_t* ready=nullptr;
    std::vector<GgufCpuExpert> experts;
    std::promise<void> completion;
    std::shared_future<void> future;
    std::atomic<bool> submitted{false};
    double elapsed_ms=0;
    std::chrono::steady_clock::time_point queued_at;
    std::chrono::steady_clock::time_point completed_at;
    double queue_ms=0;
    GgufCpuWork():future(completion.get_future().share()){}
};
class GgufCpuExecutor {
    std::vector<int> cores_;
    mfq::HostParallelPool pool_;
    std::mutex mutex_;
    std::deque<std::shared_ptr<GgufCpuWork>> queue_;
    bool stopping_=false;
    std::atomic<std::uint64_t> activity_{0};
    std::thread worker_;
    struct PreparedExpert {
        int count=0;
        std::array<int,4> token{},first_slot{};
        std::array<const void*,4> x{},u{},h{};
        std::array<float*,4> gate{},up{},out{};
    };
    // execute() is serialized by worker_; completed jobs retain only outputs.
    std::vector<PreparedExpert> prepared_;
    std::vector<std::uint8_t> qinput_,qup_,qhidden_;
    std::vector<float> gate_,up_,hidden_;
    std::size_t workspace_bytes_=0;
    void execute(GgufCpuWork&);
    void run();
public:
    explicit GgufCpuExecutor(std::vector<int> cores);
    ~GgufCpuExecutor();
    // GPU-only layers still belong to an active CPU/GPU decode pipeline.
    // Keep its coordinator responsive; an idle request naturally parks it.
    void keep_awake() noexcept;
    void submit(std::shared_ptr<GgufCpuWork>);
};
