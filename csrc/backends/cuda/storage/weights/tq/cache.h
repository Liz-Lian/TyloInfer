#pragma once

#include "ops/tq/include/tq.h"
#include "engine/weights/lru.h"

#include <array>
#include <list>
#include <mutex>
#include <unordered_map>

namespace mfq { class TqTable; }
namespace mfq::cuda { class Event; }

// Selected weights own their backing after eviction; uploads retain host data
// until DMA completes. Cache mutation is serialized across request streams.
class TqCache {
public:
    TqCache(std::size_t budget, std::size_t host_budget, std::size_t process_limit,
            bool retain_all, std::shared_ptr<TqExec> exec);
    ~TqCache();

    std::uint64_t add(std::shared_ptr<const mfq::TqTable>, std::int64_t rows);
    std::vector<TqWeight> get(std::uint64_t source, const std::int64_t* ids,
                              std::size_t count, int device);
    mfq_tensor_backend::Tensor views(std::uint64_t source, const std::int64_t* ids,
                                      const std::vector<TqWeight>& weights, int device);
    std::size_t budget();
    void resize(std::size_t bytes);
    std::size_t reserve(std::size_t bytes);
    std::vector<std::pair<std::string, double>> metrics();

private:
    struct HostImage;
    struct Stage;
    struct HostTransfer {
        std::shared_ptr<mfq::cuda::Event> ready;
        std::vector<std::shared_ptr<HostImage>> images;
        ~HostTransfer() noexcept;
    };
    struct DeviceKey {
        std::uint64_t image;
        int device;
        bool operator==(const DeviceKey&) const = default;
    };
    struct DeviceKeyHash {
        std::size_t operator()(const DeviceKey& key) const noexcept {
            return std::hash<std::uint64_t>{}(key.image) ^ (std::hash<int>{}(key.device) << 1);
        }
    };
    struct DeviceEntry {
        TqWeight weight;
        std::shared_ptr<mfq::cuda::Event> uploaded;
        cudaStream_t producer = nullptr;
        std::size_t source_bytes = 0;
    };
    struct ViewTable {
        int device;
        cudaStream_t stream;
        mfq_tensor_backend::Tensor views;
        std::vector<std::uintptr_t> images;
        std::vector<std::uint8_t> bytes;
    };
    struct DeviceStages {
        std::array<std::unique_ptr<Stage>, 2> stages;
        unsigned index = 0;
    };
    struct Source {
        std::shared_ptr<const mfq::TqTable> table;
        std::int64_t rows;
    };

    // Streams outlive every cached tensor and its final consumer fence.
    std::shared_ptr<TqExec> tq_;
    mfq::weights::Lru<std::uint64_t, std::shared_ptr<HostImage>> host_;
    mfq::weights::Lru<DeviceKey, DeviceEntry, DeviceKeyHash> device_;
    std::unordered_map<std::uint64_t, std::vector<ViewTable>> view_tables_;
    std::size_t payload_bytes_ = 0, process_host_limit_ = 0;
    std::uint64_t source_reads_ = 0;
    std::vector<Source> sources_;
    std::mutex mutex_;
    std::unordered_map<int, DeviceStages> stages_;
    std::list<HostTransfer> host_transfers_;
    std::uint64_t calls_ = 0, gpu_hits_ = 0, host_hits_ = 0, source_bytes_ = 0, transfer_bytes_ = 0;
    std::uint64_t direct_transfer_bytes_ = 0, pin_attempts_ = 0, pin_failures_ = 0;
    double fetch_ms_ = 0, upload_submit_ms_ = 0;
    bool profile_;

    std::size_t reserve_host_locked(std::size_t bytes);
    void report() const;
    void upload(const mfq_tensor_backend::Tensor&, const std::vector<std::uint8_t>&);
};
