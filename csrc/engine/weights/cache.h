#pragma once

#include "engine/weights/loader.h"
#include "engine/weights/workspace.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::engine { class Budget; }

namespace tyloi::weights {
class Cache;
namespace detail {
struct CacheData;
CacheData& cache(Cache& value);
const CacheData& cache(const Cache& value);
}

// The owner drains consumers and enters the CUDA scope before workspace operations.
class Cache {
public:
    enum class Graph { native, legacy, model_parallel };
    enum class Prefill { regular, streamed, retained };
    enum class Stage { loaded, decode, snapshot };
    struct PrefillBudget {
        std::size_t reserve_bytes = 0;
        std::size_t lendable_bytes = 0;
        std::size_t transfer_bytes = 0;
    };
    struct Activity {
        std::uint64_t streamed_bytes = 0;
        std::uint64_t graph_captures = 0;
    };
    struct KvMetrics {
        std::size_t bytes = 0;
        std::uint64_t hit_bytes = 0;
        double lookup_us = 0;
    };

    class Routes {
    public:
        Routes() noexcept;
        Routes(Routes&&) noexcept;
        ~Routes();
        Routes(const Routes&) = delete;
        Routes& operator=(const Routes&) = delete;
        Routes& operator=(Routes&&) = delete;
        void begin(Cache& cache, int rows);
        void resolve(int rows, bool cancelled);
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Warmup {
    public:
        Warmup(Cache& cache, bool suspend_residency);
        ~Warmup();
        Warmup(const Warmup&) = delete;
        Warmup& operator=(const Warmup&) = delete;
        Warmup(Warmup&&) = delete;
        Warmup& operator=(Warmup&&) = delete;
        void drain();
        void release(Device device, bool decode);
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    class Priority {
    public:
        Priority(Cache& cache, int first, int last);
        ~Priority();
        Priority(const Priority&) = delete;
        Priority& operator=(const Priority&) = delete;
        Priority(Priority&&) = delete;
        Priority& operator=(Priority&&) = delete;
        void finish();
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    Cache();
    ~Cache();
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;

    Loader loader(const source::Source& source, Loader::Options options, Loader::Placement placement);
    std::vector<std::pair<std::string, double>> metrics() const;
    Activity activity() const;
    std::size_t budget() const;
    std::size_t minimum() const;
    bool empty() const;
    bool graph_safe(Graph kind = Graph::native, bool parallel = false) const;
    bool needs_finalize() const;
    void finalize(std::size_t reserve = 0, std::size_t kv_charge = 0);
    void resize(std::size_t bytes);
    std::size_t reserve(std::size_t bytes);
    void drain() const;
    void finish_exchange() const;
    void retire_prefill();
    void prepare_predictor();
    void settle();
    void trace(const char* boundary, bool wait, const source::Source* source);
    void release_workspace(std::size_t required_available =
        std::numeric_limits<std::size_t>::max(), const char* reason = nullptr);
    void clear_workspace() noexcept;
    Prefill prefill_mode(std::size_t tokens, bool resumed) const;
    PrefillBudget prefill_budget(bool allow_borrow) const;
    void workspace(std::size_t bytes);
    void request(bool retire_decode);
    std::optional<std::size_t> prepare_prefill(Device device, std::size_t tokens,
        std::int64_t limit, Prefill mode, bool retire_decode);
    void configure_decode(std::size_t lendable_bytes, int graph_depth);
    void prepare(Stage stage, Device device, const source::Source* source);
    void prepare_kv();
    KvRegion lend_kv(const Tensor& backing, std::int64_t capacity, std::size_t row_bytes,
        bool prefix, std::int64_t ratio = 1);
    KvRegion lend_kv(const Tensor& arena);
    void commit_kv();
    void protect_kv(std::int64_t end);
    KvMetrics kv_metrics() const;
    void bind(engine::Budget& budget, Device device, bool predictor);
    void unbind() noexcept;
    void retire_decode();
    void reset() noexcept;

private:
    friend detail::CacheData& detail::cache(Cache& value);
    friend const detail::CacheData& detail::cache(const Cache& value);
    std::unique_ptr<detail::CacheData> data_;
    std::size_t lend_decode(std::size_t maximum_minus_active);
    void restore_workspace(std::size_t reserve_bytes, const source::Source* source);
    std::size_t prefill_workspace_capacity() const noexcept;
    void append_workspace_metrics(std::vector<std::pair<std::string, double>>& metrics) const;
};
} // namespace tyloi::weights
