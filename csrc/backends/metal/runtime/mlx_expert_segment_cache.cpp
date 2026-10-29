#include "mlx_expert_segment_cache.h"
#include "mfq_container.h"
#include "mlx_resource_telemetry.h"
#include "mlx_streamed_weight_pool.h"
#include "mlx_weight_residency.h"
#include "backends/cpu/native/host_parallel.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

namespace mfq::metal {
namespace {
using namespace mlx::core;
using Clock = std::chrono::steady_clock;
}
struct MlxExpertSegmentCache::Impl {
    struct Layer {
        MlxExpertSegments source;
        std::vector<std::size_t> offsets;
        std::vector<array> tables;
        std::size_t entry_bytes = 0, payload_bytes = 0;
    };
    struct Entry {
        std::size_t offset = 0, bytes = 0;
        std::list<std::uint64_t>::iterator recency, replaceable_recency;
        bool preferred = false;
    };
    MfqContainer model;
    std::vector<Layer> layers;
    std::size_t budget, backing = 0, resident = 0, maximum_entry = 0;
    int workers;
    bool prefill_read_ahead;
    mfq::HostParallelPool io;
    mutable std::mutex mutex;
    std::list<std::uint64_t> lru, replaceable;
    bool prefill_priority_active = false;
    long double total_entry_bytes = 0, maximum_layer_bytes = 0;
    std::unordered_map<std::uint64_t, Entry> entries;
    MlxSsdExpertCacheStats counters;
    std::optional<array> pending;
    MlxResourceTelemetry resources;
    std::unique_ptr<MlxStreamedWeightPool> pool;
    struct Prefetch {
        std::size_t layer = 0;
        std::vector<std::pair<std::uint64_t, Entry>> entries;
        std::size_t calls = 0;
        std::future<double> read;
    };
    std::optional<Prefetch> prefetched;

    void wait_pending() {
        if (pending) { pending->eval(); pending.reset(); }
    }

    Impl(const MfqContainer& source, std::vector<MlxExpertSegments> definitions, std::size_t limit, int threads, bool read_ahead)
        : model(source), budget(limit), workers(threads), prefill_read_ahead(read_ahead) {
        if (!budget || workers < 1 || definitions.empty())
            throw std::invalid_argument("expert segment cache requires a budget and layers");
        const auto page = std::size_t(getpagesize());
        for (auto& definition : definitions) {
            if (definition.experts < 1 || definition.fields.empty())
                throw std::invalid_argument("expert segment layer is empty");
            Layer layer; layer.source = std::move(definition);
            for (const auto& field : layer.source.fields) {
                const auto& record = model.record(field.tensor);
                if (!field.expert_bytes || field.offset > record.nbytes ||
                    field.expert_bytes > (record.nbytes - field.offset) / layer.source.experts ||
                    field.expert_bytes > SIZE_MAX - page + 1)
                    throw std::invalid_argument("expert segment lies outside its tensor");
                const auto rounded = (field.expert_bytes + page - 1) / page * page;
                if (rounded > SIZE_MAX - layer.entry_bytes ||
                    field.expert_bytes > (SIZE_MAX - backing) / layer.source.experts)
                    throw std::overflow_error("expert segment cache extent is too large");
                layer.offsets.push_back(layer.entry_bytes); layer.entry_bytes += rounded;
                layer.payload_bytes += field.expert_bytes;
                backing += field.expert_bytes * layer.source.experts;
                const std::vector<std::uint64_t> empty(layer.source.experts, 0);
                layer.tables.emplace_back(empty.begin(), Shape{layer.source.experts}, uint64);
                MlxWeightResidency::track(layer.tables.back());
            }
            maximum_entry = std::max(maximum_entry, layer.entry_bytes);
            const auto layer_bytes = static_cast<long double>(layer.entry_bytes) * layer.source.experts;
            total_entry_bytes += layer_bytes;
            maximum_layer_bytes = std::max(maximum_layer_bytes, layer_bytes);
            layers.push_back(std::move(layer));
        }
        if (budget < maximum_entry) throw std::invalid_argument("expert cache budget cannot hold one expert");
        pool = std::make_unique<MlxStreamedWeightPool>(budget, maximum_entry);
        resources.set({0, 0, 0, backing, 0});
    }

    void end_prefill_priority() noexcept {
        prefill_priority_active = false;
        replaceable.clear();
    }
    void rebuild_replaceable() {
        replaceable.clear();
        try {
            for (auto key : lru) {
                auto& entry = entries.at(key);
                if (entry.preferred) continue;
                replaceable.push_back(key);
                entry.replaceable_recency = std::prev(replaceable.end());
            }
        } catch (const std::bad_alloc&) { end_prefill_priority(); }
    }
    void configure_prefill_priority(std::size_t index, std::size_t rows) {
        if (rows <= 6 || rows < std::size_t(layers[index].source.experts)) {
            if (prefill_priority_active) end_prefill_priority();
            return;
        }
        const auto usable = static_cast<long double>(budget) - 2 * maximum_layer_bytes;
        if (!prefill_priority_active && rows >= std::size_t(layers[index].source.experts) &&
            static_cast<long double>(budget) < total_entry_bytes && usable > 0 &&
            usable * layers[index].source.experts >= total_entry_bytes) {
            for (auto& [key, entry] : entries) entry.preferred = false;
            prefill_priority_active = true;
            rebuild_replaceable();
        }
    }
    void inserted(std::uint64_t key) {
        if (!prefill_priority_active) return;
        try {
            replaceable.push_front(key);
            entries.at(key).replaceable_recency = replaceable.begin();
        } catch (const std::bad_alloc&) { end_prefill_priority(); }
    }
    void touch(Entry& entry) {
        lru.splice(lru.begin(), lru, entry.recency);
        if (prefill_priority_active && !entry.preferred)
            replaceable.splice(replaceable.begin(), replaceable, entry.replaceable_recency);
    }
    void retain_prefill_tail(std::size_t index, const std::vector<std::int32_t>& ordered) {
        if (!prefill_priority_active) return;
        for (auto& [key, entry] : entries) if (std::size_t(key >> 32) == index) entry.preferred = false;
        rebuild_replaceable();
        if (!prefill_priority_active) return;
        const auto usable = std::max(0.L, static_cast<long double>(budget) - 2 * maximum_layer_bytes);
        const auto quota = std::min(ordered.size(), std::size_t(usable * layers[index].source.experts / total_entry_bytes));
        for (auto position = ordered.size() - quota; position < ordered.size(); ++position) {
            auto& entry = entries.at((std::uint64_t(index) << 32) | std::uint32_t(ordered[position]));
            entry.preferred = true;
            replaceable.erase(entry.replaceable_recency);
        }
    }
    void erase(std::unordered_map<std::uint64_t, Entry>::iterator found) {
        const auto key = found->first;
        for (auto& table : layers.at(std::size_t(key >> 32)).tables)
            table.data<std::uint64_t>()[std::uint32_t(key)] = 0;
        resident -= found->second.bytes;
        pool->release(found->second.offset, found->second.bytes);
        if (prefill_priority_active && !found->second.preferred) replaceable.erase(found->second.replaceable_recency);
        lru.erase(found->second.recency); entries.erase(found); ++counters.evictions;
    }
    bool evict_one(const std::unordered_set<std::uint64_t>& protected_keys) {
        const auto& candidates = prefill_priority_active ? replaceable : lru;
        for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
            if (protected_keys.contains(*it)) continue;
            erase(entries.find(*it)); return true;
        }
        if (prefill_priority_active) for (auto it = lru.rbegin(); it != lru.rend(); ++it) {
            if (protected_keys.contains(*it)) continue;
            erase(entries.find(*it)); return true;
        }
        return false;
    }
    void evict(std::size_t incoming, const std::unordered_set<std::uint64_t>& protected_keys = {}) {
        while ((resident > budget || incoming > budget - resident) && evict_one(protected_keys)) {}
        if (resident > budget || incoming > budget - resident)
            throw std::runtime_error("expert cache budget cannot hold this batch");
        resources.set({0, 0, pool->capacity(), backing, 0});
    }

    void finish_prefetch(std::size_t layer_index = SIZE_MAX, std::span<const std::int32_t> selected = {}) {
        if (!prefetched) return;
        auto state = std::move(*prefetched);
        prefetched.reset();
        const auto start = Clock::now();
        try {
            counters.io_seconds += state.read.get();
            counters.prefill_wait_seconds += std::chrono::duration<double>(Clock::now() - start).count();
            const auto count = state.entries.size();
            for (auto& [key, entry] : state.entries) {
                if (state.layer == layer_index && !std::binary_search(selected.begin(), selected.end(), std::int32_t(key))) {
                    pool->release(entry.offset, entry.bytes); resident -= entry.bytes;
                    entry.offset = SIZE_MAX;
                    continue;
                }
                lru.push_front(key); entry.recency = lru.begin();
                try { entries.emplace(key, entry); } catch (...) { lru.pop_front(); throw; }
                inserted(key);
                entry.offset = SIZE_MAX;
            }
            counters.read_calls += state.calls;
            counters.loads += count;
            counters.prefill_expert_reads += count;
            counters.bytes_read += count * layers[state.layer].payload_bytes;
            counters.prefill_bytes_read += count * layers[state.layer].payload_bytes;
            ++counters.prefill_read_batches;
        } catch (...) {
            for (const auto& [key, entry] : state.entries) if (entry.offset != SIZE_MAX) {
                pool->release(entry.offset, entry.bytes); resident -= entry.bytes;
            }
            ++counters.failures;
            throw;
        }
    }

    void prefetch_next(std::size_t current, std::size_t rows) {
        if (!prefill_read_ahead || prefetched || current + 1 >= layers.size()) return;
        const auto index = current + 1;
        auto& layer = layers[index];
        if (rows < std::size_t(layer.source.experts) || rows < std::size_t(layers[current].source.experts)) return;
        std::size_t reserve = 0;
        for (int expert = 0; expert < layer.source.experts; ++expert) {
            const auto key = (std::uint64_t(index) << 32) | std::uint32_t(expert);
            if (!entries.contains(key)) reserve += layer.entry_bytes;
        }
        if (!reserve || resident > budget || reserve > budget - resident) return;
        Prefetch state; state.layer = index;
        struct ReadJob {
            std::string tensor;
            std::uint64_t offset;
            std::vector<std::span<std::byte>> targets;
        };
        std::vector<ReadJob> jobs;
        try {
            state.entries.reserve(layer.source.experts);
            for (int expert = 0; expert < layer.source.experts; ++expert) {
                const auto key = (std::uint64_t(index) << 32) | std::uint32_t(expert);
                if (entries.contains(key)) continue;
                const auto offset = pool->allocate(layer.entry_bytes);
                if (!offset) {
                    for (const auto& item : state.entries) pool->release(item.second.offset, item.second.bytes);
                    resources.set({0, 0, pool->capacity(), backing, 0});
                    return;
                }
                state.entries.push_back({key, Entry{*offset, layer.entry_bytes, {}}});
            }
            for (std::size_t field = 0; field < layer.source.fields.size(); ++field) {
                const auto& source = layer.source.fields[field];
                const auto maximum = std::max<std::size_t>(1, (32u << 20) / source.expert_bytes);
                for (std::size_t first = 0; first < state.entries.size();) {
                    auto last = first + 1;
                    while (last < state.entries.size() && last - first < maximum &&
                        state.entries[last].first == state.entries[last - 1].first + 1) ++last;
                    ReadJob job{source.tensor, source.offset +
                        std::uint64_t(std::uint32_t(state.entries[first].first)) * source.expert_bytes, {}};
                    for (auto item = first; item < last; ++item)
                        job.targets.emplace_back(pool->data(state.entries[item].second.offset + layer.offsets[field]), source.expert_bytes);
                    jobs.push_back(std::move(job)); first = last;
                }
            }
            state.calls = jobs.size();
            state.read = std::async(std::launch::async, [this, jobs = std::move(jobs)] {
                const auto start = Clock::now();
                io.run(0, std::int64_t(jobs.size()), 1, workers, [&](auto first, auto last) {
                    for (auto item = first; item < last; ++item) {
                        const auto& job = jobs[std::size_t(item)];
                        model.readv_range(job.tensor, job.offset, job.targets);
                    }
                });
                return std::chrono::duration<double>(Clock::now() - start).count();
            });
            resident += state.entries.size() * layer.entry_bytes;
            prefetched.emplace(std::move(state));
            ++counters.overlapped_read_submissions;
            resources.set({0, 0, pool->capacity(), backing, 0});
        } catch (...) {
            if (state.read.valid()) state.read.wait();
            for (const auto& item : state.entries) pool->release(item.second.offset, item.second.bytes);
            resources.set({0, 0, pool->capacity(), backing, 0});
            throw;
        }
    }

    void prepare(std::size_t index, const std::vector<std::int32_t>& selected,
        std::size_t first_fields, const std::function<void()>& stage) {
        auto& layer = layers.at(index);
        std::vector<std::pair<std::uint64_t, Entry>> pending;
        pending.reserve(selected.size());
        bool staged = false;
        struct Cleanup {
            Impl& cache;
            std::vector<std::pair<std::uint64_t, Entry>>& pending;
            bool& staged;
            ~Cleanup() {
                if (staged && std::any_of(pending.begin(), pending.end(),
                    [](const auto& item) { return item.second.offset != SIZE_MAX; })) synchronize();
                for (const auto& item : pending) if (item.second.offset != SIZE_MAX)
                    cache.pool->release(item.second.offset, item.second.bytes);
                cache.resources.set({0, 0, cache.pool->capacity(), cache.backing, 0});
            }
        } cleanup{*this, pending, staged};
        std::unordered_set<std::uint64_t> protected_keys;
        std::size_t reserve = 0;
        for (const auto expert : selected) {
            const auto key = (std::uint64_t(index) << 32) | std::uint32_t(expert);
            protected_keys.insert(key);
            if (!entries.contains(key)) reserve += layer.entry_bytes;
        }
        evict(reserve, protected_keys);
        for (const auto expert : selected) {
            const auto key = (std::uint64_t(index) << 32) | std::uint32_t(expert);
            ++counters.requests;
            if (const auto found = entries.find(key); found != entries.end()) {
                ++counters.hits; touch(found->second); continue;
            }
            auto offset = pool->allocate(layer.entry_bytes);
            while (!offset && evict_one(protected_keys)) offset = pool->allocate(layer.entry_bytes);
            if (!offset) throw std::runtime_error("expert batch cannot fit the memory pool");
            pending.push_back({key, Entry{*offset, layer.entry_bytes, {}}});
        }
        struct ReadJob { std::size_t field, first, last; };
        std::vector<ReadJob> jobs;
        std::size_t first_jobs = 0;
        for (std::size_t field = 0; field < layer.source.fields.size(); ++field) {
            const auto maximum = std::max<std::size_t>(1, (32u << 20) / layer.source.fields[field].expert_bytes);
            for (std::size_t first = 0; first < pending.size();) {
                auto last = first + 1;
                while (last < pending.size() && last - first < maximum && pending[last].first == pending[last - 1].first + 1) ++last;
                jobs.push_back({field, first, last}); first = last;
            }
            if (field + 1 == first_fields) first_jobs = jobs.size();
        }
        for (auto& table : layer.tables)
            std::fill_n(table.data<std::uint64_t>(), layer.source.experts, std::uint64_t{0});
        const auto publish = [&](std::size_t first, std::size_t last) {
            const auto assign = [&](std::int32_t expert, std::size_t offset) {
                for (auto field = first; field < last; ++field)
                    layer.tables[field].data<std::uint64_t>()[expert] = pool->address(offset + layer.offsets[field]);
            };
            for (const auto expert : selected) {
                const auto found = entries.find((std::uint64_t(index) << 32) | std::uint32_t(expert));
                if (found != entries.end()) assign(expert, found->second.offset);
            }
            for (const auto& [key, entry] : pending) assign(std::int32_t(key), entry.offset);
        };
        const auto start = Clock::now();
        const auto read = [&](std::int64_t begin, std::int64_t end) {
            for (auto job = begin; job < end; ++job) {
                const auto& read = jobs[std::size_t(job)];
                const auto& field = layer.source.fields[read.field];
                std::vector<std::span<std::byte>> targets;
                targets.reserve(read.last - read.first);
                for (auto item = read.first; item < read.last; ++item)
                    targets.emplace_back(pool->data(pending[item].second.offset + layer.offsets[read.field]), field.expert_bytes);
                model.readv_range(field.tensor, field.offset + std::uint64_t(std::uint32_t(pending[read.first].first)) * field.expert_bytes, targets);
            }
        };
        try {
            if (first_jobs && first_jobs < jobs.size()) {
                io.run_staged(std::int64_t(jobs.size()), std::int64_t(first_jobs), workers, read, [&] {
                    publish(0, first_fields);
                    staged = true;
                    stage();
                });
                ++counters.overlapped_read_batches;
            } else io.run(0, std::int64_t(jobs.size()), 1, workers, read);
        } catch (...) {
            if (staged) synchronize();
            throw;
        }
        counters.io_seconds += std::chrono::duration<double>(Clock::now() - start).count(); counters.read_calls += jobs.size();
        publish(staged ? first_fields : 0, layer.tables.size());
        for (auto& [key, entry] : pending) {
            lru.push_front(key); entry.recency = lru.begin();
            try { entries.emplace(key, entry); } catch (...) { lru.pop_front(); throw; }
            inserted(key);
            resident += entry.bytes; entry.offset = SIZE_MAX;
            counters.bytes_read += layer.payload_bytes; ++counters.misses; ++counters.loads;
        }
    }
};

MlxExpertSegmentCache::MlxExpertSegmentCache(const MfqContainer& model, std::vector<MlxExpertSegments> layers,
    std::size_t budget, int workers, bool prefill_read_ahead)
    : impl_(std::make_shared<Impl>(model, std::move(layers), budget, workers, prefill_read_ahead)) {}
const std::vector<array>& MlxExpertSegmentCache::address_tables(std::size_t layer) const { return impl_->layers.at(layer).tables; }
const std::shared_ptr<MlxWeightResidency>& MlxExpertSegmentCache::residency() const noexcept { return impl_->pool->residency(); }

array MlxExpertSegmentCache::apply(std::size_t index, const array& ids, std::size_t first_fields,
    const std::function<array()>& first_stage, const std::function<array(const array&, bool)>& second_stage,
    const array* parallel) {
    std::lock_guard lock(impl_->mutex);
    const auto& layer = impl_->layers.at(index);
    if (!first_fields || first_fields >= layer.source.fields.size())
        throw std::invalid_argument("invalid expert pipeline field boundary");
    if (ids.ndim() != 2 || ids.shape(0) < 1 || ids.shape(1) < 1 || (ids.dtype() != int32 && ids.dtype() != uint32))
        throw std::invalid_argument("expert cache requires rank-two integer routes");
    const auto begin = Clock::now();
    auto routes = contiguous(astype(ids, int32)); routes.eval();
    impl_->wait_pending();
    impl_->counters.route_sync_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
    std::vector<std::int32_t> selected(routes.data<std::int32_t>(), routes.data<std::int32_t>() + routes.size());
    std::sort(selected.begin(), selected.end()); selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    if (selected.front() < 0 || selected.back() >= layer.source.experts)
        throw std::invalid_argument("expert routes are outside the model");
    if (parallel && !parallel->is_available() && std::any_of(selected.begin(), selected.end(), [&](auto expert) {
            return !impl_->entries.contains((std::uint64_t(index) << 32) | std::uint32_t(expert));
        })) async_eval({*parallel});
    impl_->finish_prefetch(index, selected);
    impl_->configure_prefill_priority(index, std::size_t(ids.shape(0)));
    std::vector<std::size_t> last_use(layer.source.experts, 0);
    for (std::size_t position = 0; position < routes.size(); ++position)
        last_use[routes.data<std::int32_t>()[position]] = position;
    const auto batch_size = impl_->pool->batch_capacity(layer.entry_bytes);
    if (!batch_size) throw std::runtime_error("expert cache budget cannot hold one expert");
    std::optional<array> output;
    Dtype dtype = float16;
    for (std::size_t start = 0; start < selected.size(); start += batch_size) {
        const std::vector<std::int32_t> batch(selected.begin() + start, selected.begin() + std::min(selected.size(), start + batch_size));
        const auto io_before = impl_->counters.io_seconds;
        const auto prepare_start = Clock::now();
        std::optional<array> intermediate;
        impl_->prepare(index, batch, first_fields, [&] {
            intermediate = first_stage();
            std::vector<array> submitted{*intermediate};
            if (parallel && !parallel->is_available()) submitted.push_back(*parallel);
            if (!impl_->prefill_read_ahead || std::size_t(ids.shape(0)) < std::size_t(layer.source.experts))
                async_eval(submitted);
            else eval(submitted);
        });
        impl_->counters.prepare_seconds += std::max(0.0, std::chrono::duration<double>(Clock::now() - prepare_start).count()
            - (impl_->counters.io_seconds - io_before));
        const auto graph_start = Clock::now();
        if (!intermediate) intermediate = first_stage();
        if (selected.size() <= batch_size) impl_->prefetch_next(index, std::size_t(ids.shape(0)));
        auto value = second_stage(*intermediate, selected.size() > batch_size);
        if (!output) dtype = value.dtype();
        if (selected.size() > batch_size) value = astype(value, float32);
        output = output ? *output + value : std::move(value);
        if (start + batch_size < selected.size()) output->eval();
        else { impl_->pending = *output; async_eval({*output}); }
        auto recency = batch;
        std::sort(recency.begin(), recency.end(), [&](auto a, auto b) { return last_use[a] < last_use[b]; });
        for (const auto expert : recency) {
            const auto key = (std::uint64_t(index) << 32) | std::uint32_t(expert);
            impl_->touch(impl_->entries.at(key));
        }
        impl_->retain_prefill_tail(index, recency);
        impl_->counters.routed_graph_seconds += std::chrono::duration<double>(Clock::now() - graph_start).count();
    }
    return output->dtype() == dtype ? *output : astype(*output, dtype);
}

std::size_t MlxExpertSegmentCache::limit_bytes() const noexcept { return impl_->budget; }
std::size_t MlxExpertSegmentCache::backing_bytes() const noexcept { return impl_->backing; }
std::size_t MlxExpertSegmentCache::resident_bytes() const noexcept { return impl_->pool->capacity(); }
std::size_t MlxExpertSegmentCache::set_limit(std::size_t bytes) {
    if (!bytes) throw std::invalid_argument("expert cache budget must be nonzero");
    std::lock_guard lock(impl_->mutex);
    impl_->wait_pending();
    impl_->finish_prefetch();
    const auto before = impl_->pool->capacity(), target = bytes / std::size_t(getpagesize()) * std::size_t(getpagesize());
    if (!target) throw std::invalid_argument("expert cache budget is smaller than a page");
    if (target < impl_->maximum_entry) throw std::invalid_argument("expert cache budget cannot hold one expert");
    impl_->end_prefill_priority();
    for (auto it = impl_->entries.begin(); it != impl_->entries.end();) {
        auto current = it++;
        if (current->second.offset > target || current->second.bytes > target - current->second.offset) impl_->erase(current);
    }
    impl_->pool->set_limit(bytes); impl_->budget = bytes; impl_->evict(0);
    return before - impl_->pool->capacity();
}
MlxSsdExpertCacheStats MlxExpertSegmentCache::stats() const {
    std::lock_guard lock(impl_->mutex);
    auto result = impl_->counters;
    result.resident_experts = impl_->entries.size(); result.resident_bytes = impl_->pool->capacity();
    std::size_t payload = 0, metadata = 0;
    for (const auto& [key, entry] : impl_->entries) payload += impl_->layers.at(std::size_t(key >> 32)).payload_bytes;
    for (const auto& layer : impl_->layers) for (const auto& table : layer.tables) metadata += table.buffer_size();
    result.cached_payload_bytes = payload; result.cache_metadata_bytes = metadata;
    result.pinned_cache_bytes = impl_->pool->pinned_bytes();
    return result;
}
void MlxExpertSegmentCache::clear() {
    std::lock_guard lock(impl_->mutex);
    impl_->wait_pending();
    impl_->finish_prefetch();
    impl_->end_prefill_priority();
    impl_->entries.clear(); impl_->lru.clear(); impl_->resident = 0;
    for (auto& layer : impl_->layers) for (auto& table : layer.tables)
        std::fill_n(table.data<std::uint64_t>(), layer.source.experts, std::uint64_t{0});
    impl_->pool->clear(); impl_->resources.set({0, 0, 0, impl_->backing, 0});
}
} // namespace mfq::metal
