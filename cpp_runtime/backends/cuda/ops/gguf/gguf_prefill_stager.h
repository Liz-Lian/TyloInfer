#pragma once
#include "mfq_cuda_context.h"
#include "mfq_native_tensor.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Adapted from Strata src/prefill/prefill.cpp (MIT): parallel RAM copies
// run ahead of uploads through a generation-safe, DMA-fenced pinned ring.
class GgufPrefillStager {
public:
    // D-5: the pinned ring's depth (STRATA_STAGER_RING, default 16) - how far the host copies can run ahead of the
    // DMAs of the unpinned experts' blobs
    int kRing = 16;
    // pp-opt: the jobs a thread claims at once (STRATA_STAGER_BATCH; set by init's caller, at most kMaxBatch and kRing)
    static constexpr int kMaxBatch = 32;
    int batch = 1;
    // Source owners survive every CPU copy, including cancelled layers.
    struct Job { const uint8_t* src; size_t bytes; std::shared_ptr<void> owner; };
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<cudaEvent_t> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    // layer-major prefill: a job the routing turned out not to need (`skip`) is not read once nobody has claimed it
    std::unique_ptr<std::atomic<char>[]> skipped;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one - the expert pool's issue #29 lesson)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;
    std::size_t capacity=0;

    bool init(size_t blob_bytes, int nthreads) {
        capacity=blob_bytes;

        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = cudaHostAlloc((void**) &buf[i], blob_bytes, cudaHostAllocDefault) == cudaSuccess;
            if (!pinned[i]) {
                cudaGetLastError();
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            mfq::cuda::charge_tensor_host_bytes(blob_bytes);
            if (cudaEventCreateWithFlags(&dma_done[i], cudaEventDisableTiming | cudaEventBlockingSync) != cudaSuccess) return false;
        }
        cudaGetDevice(&device);
        for (int t = 0; t < nthreads; ++t)
            threads.emplace_back([this] {
                work();
            });
        return true;
    }
    ~GgufPrefillStager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) {cudaEventSynchronize(dma_done[i]);cudaEventDestroy(dma_done[i]);}
            if (buf[i]) {if(pinned[i])cudaFreeHost(buf[i]);mfq::cuda::tensor_host_bytes.fetch_sub(capacity);}
        }
    }
    void work() {
        cudaSetDevice(device);
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                // pp-opt: a run of up to `batch` consecutive jobs (batch <= kRing, so the earliest unfinished run
                // never waits for a buffer of its own)
                int k = 1;
                const int j = claim(seen, batch, k);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                for (int i = j; i < j + k; ++i) {
                    // The copies run ahead of the DMAs, so most of these threads spend most of a prompt waiting here: sleep
                    // (atomic wait, and a blocking-sync event below), not a yield spin - 32 spinners took every core
                    // (Linux; see stager_sleep for Windows).
                    if (i >= kRing) {   // job i - kRing's DMA from this buffer is queued
#if defined(__cpp_lib_atomic_wait)
                        if (true)
                            for (int x; (x = issued.load(std::memory_order_acquire)) <= i - kRing;) issued.wait(x);
                        else
#endif
                            while (issued.load(std::memory_order_acquire) <= i - kRing) std::this_thread::yield();
                        // ...and its own copy into the buffer is done: `issued` is a watermark, so a job nobody waits for
                        // (the layer-major prefetch reads experts the routing may skip) can be passed while its thread still
                        // writes the buffer this job is about to fill
                        while (!ready[(size_t) (i - kRing)].load(std::memory_order_acquire)) std::this_thread::yield();
                    }
                    // and done - for a generation's first kRing jobs that is the previous generation's last DMA from
                    // the buffer, which nothing else waits for when a chunk ends without a sync (no MTP) or the DMA
                    // was a ring entry the routing skipped (an event never recorded returns at once)
                    cudaEventSynchronize(dma_done[i % kRing]);
                }
                for (int i = j; i < j + k; ++i) {
                    if(!skipped[(size_t)i].load(std::memory_order_acquire)) {
                        const auto& job=jobs[(size_t)i];
                        std::memcpy(buf[i%kRing],job.src,job.bytes);
                    }
                    ready[(size_t)i].store(1,std::memory_order_release);
                }
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    int claim(uint32_t g, int want, int& k) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            k = std::min(want, n - j);
            if (head.compare_exchange_weak(cur, cur + (uint64_t) k, std::memory_order_acq_rel, std::memory_order_acquire))
                return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js) {
        if (js.empty()) return;
        if(js.size()>65535)throw std::length_error("GGUF host staging job count");
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
            skipped.reset(new std::atomic<char>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) {
            ready[i].store(0, std::memory_order_relaxed);
            skipped[i].store(0, std::memory_order_relaxed);
        }
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Layer-major prefill: job j's bytes are not needed after all (the routing skipped the expert) - a thread that has not
    /// started it yet marks it ready without reading.  Never waited for, so its buffer's content does not matter.
    void skip(int j) { skipped[(size_t) j].store(1, std::memory_order_release); }
    /// Layer-major prefill, before wait(j): the launching thread stages its jobs in increasing order, so a job below j
    /// it has not issued was skipped and never gets a DMA - the watermark moves to j.  A skipped job is never issued,
    /// so without this a routed job after about a ring's depth of skipped ones waits for `issued` forever.
    void pass(int j) {
        if (issued.load(std::memory_order_relaxed) < j) {
            issued.store(j, std::memory_order_release);
#if defined(__cpp_lib_atomic_wait)
            issued.notify_all();
#endif
        }
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) std::this_thread::yield();
        return buf[j % kRing];
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, cudaStream_t copy) {
        cudaEventRecord(dma_done[j % kRing], copy);
        issued.store(j + 1, std::memory_order_release);
#if defined(__cpp_lib_atomic_wait)
        issued.notify_all();
#endif
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
#if defined(__cpp_lib_atomic_wait)
        issued.notify_all();
#endif
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};
