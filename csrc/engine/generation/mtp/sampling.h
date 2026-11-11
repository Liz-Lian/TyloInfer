#pragma once

#include "engine/generation/mtp/policy.h"
#include "engine/generation/sampling.h"
#include "engine/ops.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::engine::mtp {

/** Own the ordinary tensor and sampling operations used by speculative generation. */
class Sampling {
public:
    using Tensor = ::tyloi::Tensor;
    explicit Sampling(Device device) : device_(device) {}

    Tensor input_ids, counts, known_ids_host, known_ids_device;
    bool penalties = false;

    /** Release request-owned sampling tensors before borrowed storage is returned. */
    void release() noexcept {
        input_ids = {};
        counts = {};
        known_ids_host = {};
        known_ids_device = {};
        sampler_.reset();
        penalties = false;
    }

    /** Prepare prompt IDs and the sampler before model-specific initialization. */
    void initialize(const std::vector<std::int64_t>& prompt, const MfqSamplingParams& parameters) {
        input_ids = ids_for(prompt);
        sampler_.emplace(parameters, device_);
        penalties = sampler().has_penalties();
    }
    /** Initialize token counts after model-specific initialization. */
    void count_prompt(std::int64_t vocabulary) {
        if (!penalties) return;
        counts = ops::zeros({vocabulary}, TensorOptions().device(device_).dtype(kInt32));
        ops::add_token_counts(counts, input_ids);
    }
    /** Return the sampler initialized for the active request. */
    Sampler& sampler() { return *sampler_; }
    /** Create device token IDs with a single batch dimension. */
    Tensor ids_for(std::vector<std::int64_t> tokens) {
        known_ids_host = ops::reshape(ops::tensor(tokens, Device(kCPU)), {1, -1});
        known_ids_device = ops::copy(known_ids_host, device_);
        return known_ids_device;
    }
    /** Accumulate token IDs into the supplied count backing. */
    static void add_counts(Tensor counts, const Tensor& ids) { ops::add_token_counts(counts, ids); }
    /** Report whether a tensor handle has a value. */
    static bool defined(const Tensor& value) { return value.defined(); }
    /** Return a tensor dimension. */
    static std::int64_t size(const Tensor& value, int axis) { return value.size(axis); }
    /** Return the tensor rank. */
    static int rank(const Tensor& value) { return static_cast<int>(value.dim()); }
    /** Clone device data or move host history to the sampling device. */
    Tensor clone(const Tensor& value) const {
        return value.is_cuda() ? ops::clone(value) : ops::copy(value, device_);
    }
    /** Return a contiguous slice of the requested axis. */
    static Tensor slice(const Tensor& value, int axis, std::int64_t offset, std::int64_t length) {
        return ops::contiguous(ops::slice(value, axis, offset, offset + length));
    }
    /** Reshape a tensor without changing its values. */
    static Tensor reshape(Tensor value, std::initializer_list<std::int64_t> shape) {
        return ops::reshape(value, shape);
    }
    /** Concatenate on the host when any input is host-resident. */
    static Tensor concatenate(const std::vector<Tensor>& values, int axis) {
        if (std::any_of(values.begin(), values.end(), [](const auto& value) { return !value.is_cuda(); })) {
            auto host = values;
            for (auto& value : host) value = ops::copy(value, Device(kCPU));
            return ops::contiguous(ops::cat(host, axis));
        }
        return ops::contiguous(ops::cat(values, axis));
    }
    /** Sample one token, preserving logits when applying penalties. */
    std::int32_t sample_normal(Tensor logits, Tensor token_counts) {
        if (penalties) logits = ops::clone(logits);
        return static_cast<std::int32_t>(ops::item(engine::sample_logits(sampler(), std::move(logits), token_counts)));
    }
    /** Sample greedy tokens for a matrix of independent rows. */
    std::vector<std::int32_t> greedy_tokens(Tensor logits) {
        if (penalties || !sampler().greedy() || logits.dim() != 2)
            throw std::invalid_argument("batched greedy MTP sampling geometry is invalid");
        const auto values = ops::integers(ops::sample_greedy(ops::contiguous(logits)));
        return {values.begin(), values.end()};
    }
    /** Sample one token using an independent copy of the constraint cursor. */
    std::int32_t sample_constrained(Tensor logits, Tensor token_counts, const MfqTokenConstraintPtr& constraint) {
        if (!constraint && sampler().greedy() && !penalties) return sample_normal(logits, token_counts);
        if (penalties) logits = ops::clone(logits);
        auto cursor = constraint ? constraint->clone() : MfqTokenConstraintPtr{};
        return static_cast<std::int32_t>(ops::item(engine::sample_logits(sampler(), std::move(logits), token_counts, cursor)));
    }
    /** Produce the full probability distribution for one row. */
    std::vector<float> probabilities(Tensor logits, Tensor token_counts, const MfqSamplingParams& parameters) {
        logits = adjusted(std::move(logits), token_counts, parameters);
        const auto values = ops::floats(logits);
        return distribution(values, parameters.temperature, parameters.top_k, parameters.top_p);
    }
    /** Produce a compact top-k distribution for one row. */
    CompactDistribution compact_probabilities(Tensor logits, Tensor token_counts, const MfqSamplingParams& parameters) {
        validate_compact(parameters);
        logits = adjusted(std::move(logits), token_counts, parameters);
        return std::move(compact_rows(ops::top_k(logits, parameters.top_k), parameters).front());
    }
    /** Produce compact distributions for an unpenalized logits matrix. */
    std::vector<CompactDistribution> compact_probability_rows(Tensor logits, const MfqSamplingParams& parameters) {
        validate_compact(parameters);
        if (penalties || logits.dim() != 2)
            throw std::invalid_argument("batched compact MTP sampling geometry is invalid");
        return compact_rows(ops::top_k(ops::contiguous(logits), parameters.top_k), parameters);
    }
    /** Compute radix top-k rows for validating captured graph output. */
    static std::vector<CompactDistribution> compact_radix_rows(const Tensor& logits, const MfqSamplingParams& parameters) {
        validate_compact(parameters);
        return compact_rows(ops::top_k(logits, parameters.top_k, true), parameters);
    }
    /** Decode packed top-k rows retained by a captured graph. */
    static std::vector<CompactDistribution> compact_from_host(const Tensor& host, const MfqSamplingParams& parameters) {
        validate_compact(parameters);
        if (host.dim() != 2 || host.size(1) < parameters.top_k)
            throw std::invalid_argument("captured MTP top-k packet geometry is invalid");
        const auto packets = ops::integers(host);
        ops::TopK selected;
        selected.rows = host.size(0);
        selected.columns = parameters.top_k;
        selected.values.reserve(std::size_t(selected.rows * selected.columns));
        selected.indices.reserve(std::size_t(selected.rows * selected.columns));
        for (std::int64_t row = 0; row < selected.rows; ++row) {
            for (std::int64_t col = 0; col < selected.columns; ++col) {
                const auto packet = std::uint64_t(packets[std::size_t(row * host.size(1) + col)]);
                const auto raw = std::uint32_t(packet >> 32);
                float value;
                std::memcpy(&value, &raw, sizeof(value));
                selected.values.push_back(value);
                selected.indices.push_back(std::uint32_t(packet));
            }
        }
        return compact_rows(selected, parameters);
    }
protected:
    std::optional<Sampler> sampler_;
private:
    static void validate_compact(const MfqSamplingParams& parameters) {
        if (parameters.top_k <= 0 || parameters.top_k > 64 || parameters.temperature <= 0.0)
            throw std::invalid_argument("compact MTP sampling requires top-k 1-64");
    }
    Tensor adjusted(Tensor logits, const Tensor& token_counts, const MfqSamplingParams& parameters) const {
        logits = ops::reshape(ops::contiguous(logits), {1, -1});
        if (penalties) {
            logits = ops::clone(logits);
            ops::apply_penalties(logits, token_counts, parameters.presence_penalty,
                parameters.frequency_penalty, parameters.repetition_penalty);
        }
        return logits;
    }
    static std::vector<CompactDistribution> compact_rows(const ops::TopK& selected, const MfqSamplingParams& parameters) {
        std::vector<CompactDistribution> result;
        result.reserve(std::size_t(selected.rows));
        for (std::int64_t row = 0; row < selected.rows; ++row) {
            const auto offset = std::size_t(row * selected.columns);
            result.push_back(compact_distribution_from_topk(selected.values.data() + offset,
                selected.indices.data() + offset, int(selected.columns), parameters.temperature, parameters.top_p));
        }
        return result;
    }
    Device device_;
};

} // namespace tyloi::engine::mtp
