#include "mfe_blob.h"
#include "mfe_index.h"
#include "engine/weights/encoding/bytes.h"
#include "engine/weights/encoding/fp8_sq_blob.h"
#include "engine/weights/encoding/mxfp4_sq_blob.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace mfq::mfe {

std::vector<std::byte> header(
    std::uint32_t experts,
    std::uint32_t output,
    std::uint32_t input,
    std::uint32_t pools) {
    std::vector<std::byte> result;
    for (const char value : std::string_view("MFE1")) {
        result.push_back(static_cast<std::byte>(value));
    }
    encoding::append_little<std::uint32_t>(result, experts);
    encoding::append_little<std::uint32_t>(result, 0);
    encoding::append_little<std::uint32_t>(result, 0);
    encoding::append_little<std::uint32_t>(result, pools);
    std::memcpy(result.data() + 8, &output, sizeof(output));
    std::memcpy(result.data() + 12, &input, sizeof(input));
    return result;
}

std::vector<std::byte> expert_prefix(
    std::int32_t expert,
    std::string_view dtype,
    std::uint64_t nbytes) {
    std::vector<std::byte> result;
    encoding::append_little<std::uint32_t>(result, 1);
    encoding::append_little<std::uint32_t>(
        result, static_cast<std::uint32_t>(dtype.size()));
    encoding::append_little<std::uint64_t>(result, nbytes);
    encoding::append_little<std::uint64_t>(result, 0);
    encoding::append_little<std::int32_t>(result, expert);
    for (const char value : dtype) {
        result.push_back(static_cast<std::byte>(value));
    }
    return result;
}

Data unpack(const std::vector<uint8_t>& blob) {
    auto index = mfq::mfe::parse(blob.size(), [&](std::size_t offset, std::uint8_t* data,
                                                std::size_t size) {
        std::memcpy(data, blob.data() + offset, size);
    });
    Data result;
    result.n_experts = index.experts;
    result.out_per_expert = index.output;
    result.neuron_len = index.width;
    result.pools.reserve(index.pools.size());
    for (auto& entry : index.pools) {
        const auto expert_count = entry.ids.size();
        Pool pool;
        pool.expert_ids = std::move(entry.ids);
        pool.dtype = std::move(entry.dtype);
        pool.runtime_payload.assign(blob.begin() + entry.runtime.offset,
                                    blob.begin() + entry.runtime.offset + entry.runtime.bytes);
        pool.payload.assign(blob.begin() + entry.payload.offset,
                            blob.begin() + entry.payload.offset + entry.payload.bytes);
        if (mfq::fp8sq::is_dtype(pool.dtype)) {
            if (!pool.runtime_payload.empty()) {
                throw std::runtime_error(
                    "unexpected MFE FP8-SQ runtime metadata");
            }
            const auto layout = mfq::fp8sq::parse(
                pool.dtype, pool.payload.data(), pool.payload.size());
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (layout.outputs != expected_rows ||
                    layout.width != result.neuron_len) {
                throw std::runtime_error(
                    "MFE FP8-SQ pool weight shape mismatch");
            }
        } else if (pool.dtype == "MXFP4-SQ") {
            if (!pool.runtime_payload.empty()) {
                throw std::runtime_error(
                    "unexpected MFE MXFP4-SQ runtime metadata");
            }
            const auto layout = mfq::sq::parse(
                pool.payload.data(), pool.payload.size());
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (layout.outputs != expected_rows ||
                    layout.width != result.neuron_len) {
                throw std::runtime_error(
                    "MFE MXFP4-SQ pool weight shape mismatch");
            }
        } else if (pool.dtype == "MXFP4") {
            pool.mxfp4 = mfq::mxfp4::unpack(pool.payload);
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (pool.mxfp4.out != expected_rows ||
                    pool.mxfp4.neuron_len != result.neuron_len) {
                throw std::runtime_error(
                    "MFE MXFP4 pool weight shape mismatch");
            }
        } else if (pool.dtype == "NINT8-0") {
            pool.q8_zero = mfq::nint8::unpack(pool.payload);
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (pool.q8_zero.axis != 0 ||
                pool.q8_zero.out != expected_rows ||
                pool.q8_zero.neuron_len != result.neuron_len ||
                pool.q8_zero.shape.size() != 2 ||
                pool.q8_zero.shape[0] != expected_rows ||
                pool.q8_zero.shape[1] != result.neuron_len) {
                throw std::runtime_error(
                    "MFE NINT8-0 pool weight shape mismatch");
            }
        } else if (pool.dtype == "NINT") {
            pool.weight = mfq::nint::unpack(pool.payload);
            int expected_rows = (int)expert_count * result.out_per_expert;
            if (pool.weight.axis != 0 || pool.weight.out != expected_rows ||
                pool.weight.neuron_len != result.neuron_len ||
                pool.weight.shape.size() != 2 ||
                pool.weight.shape[0] != expected_rows ||
                pool.weight.shape[1] != result.neuron_len) {
                throw std::runtime_error("MFE pool weight shape mismatch");
            }
        }
        result.pools.push_back(std::move(pool));
    }
    return result;
}

} // namespace mfq::mfe
