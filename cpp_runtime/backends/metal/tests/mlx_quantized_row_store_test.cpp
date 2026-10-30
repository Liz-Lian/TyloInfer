#include "mlx_quantized_row_store.h"
#include "mlx_tq.h"
#include "mlx_resource_telemetry.h"
#include "mlx_deepseek_v41_engram.h"
#include "gguf.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void append_little(std::string& output, T value) {
    using Unsigned = std::make_unsigned_t<T>;
    const auto bits = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        output.push_back(static_cast<char>(bits >> (index * 8)));
    }
}

void append_string(std::string& output, const std::string& value) {
    append_little<std::uint32_t>(
        output, static_cast<std::uint32_t>(value.size()));
    output.append(value);
}

std::string row_table_payload(std::uint64_t rows = 4, std::uint64_t width = 32) {
    std::string result{"MXT1", 4};
    result.push_back(1);
    result.push_back(8);
    result.push_back(0);
    result.push_back(0);
    for (const auto value : {rows, width, rows, width, rows, width / 32}) {
        append_little<std::uint64_t>(result, value);
    }
    for (std::uint64_t row = 0; row < rows; ++row) {
        result.append(
            static_cast<std::size_t>(width),
            static_cast<char>((row & 1u) == 0 ? 0x38 : 0x40));
    }
    result.append(static_cast<std::size_t>(rows * (width / 32)), static_cast<char>(127));
    return result;
}

void write_fixture(const std::filesystem::path& path,
                   const std::string& dtype = "MXFP8",
                   const std::string& payload = row_table_payload(),
                   const std::string& name = "table.weight", int split_no = -1) {
    std::string bytes{"MFQ1", 4};
    append_little<std::uint32_t>(bytes, 2);
    append_string(bytes, "unit-test");
    append_little<std::uint32_t>(bytes, split_no < 0 ? 0 : 4);
    if (split_no >= 0) {
        for (const auto& field : std::vector<std::pair<std::string, std::string>>{
                {"split.no", std::to_string(split_no)}, {"split.count", "3"},
                {"split.tensors.count", "3"}, {"split.records.count", "3"}}) {
            append_string(bytes, field.first);
            append_string(bytes, field.second);
        }
    }
    append_little<std::uint32_t>(bytes, 1);
    append_string(bytes, name);
    append_string(bytes, dtype);
    append_little<std::uint64_t>(bytes, payload.size());
    bytes.append(payload);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(stream), "cannot write MXFP8 row fixture");
}

std::string tq_payload(int seed, bool columns) {
    constexpr int rows = 4, width = 768;
    const std::array<int, 4> bits{1, 3, 5, 8};
    std::string result{"RTRELL05", 8};
    for (const auto value : {5, rows, width, int(columns), columns ? 0 : 16,
            width * (1 + 3 + 5 + 8) / 8}) append_little<std::uint32_t>(result, value);
    result.append(32, '\0');
    std::uint64_t words = 0;
    for (int row = 0; row < rows; ++row) {
        append_little<std::uint64_t>(result, words | (std::uint64_t(bits[row] - 1) << 32)
            | (std::uint64_t(0x2800 + row * 0x100) << 40));
        words += width * bits[row] / 32;
    }
    if (columns) for (int col = 0; col < width; ++col) append_little<std::uint16_t>(result, 0x3800 + col % 32);
    else result.append(rows * (width / 16) / 2, char(0x87));
    std::uint32_t state = seed;
    for (std::size_t word = 0; word < words; ++word) {
        state = state * 1664525u + 1013904223u;
        append_little<std::uint32_t>(result, state);
    }
    return result;
}

void write_tq_shards(const std::filesystem::path& path, const nlohmann::json& manifest,
        bool wrong_dtype = false, bool physical_alias = false) {
    const std::array<std::string, 2> payloads{tq_payload(1, false), tq_payload(2, true)};
    std::string bytes{"MFQ1", 4};
    append_little<std::uint32_t>(bytes, 2);
    append_string(bytes, "unit-test");
    append_little<std::uint32_t>(bytes, 1);
    append_string(bytes, "tq_row_shards"); append_string(bytes, manifest.dump());
    append_little<std::uint32_t>(bytes, physical_alias ? 3 : 2);
    for (int index = 0; index < (physical_alias ? 3 : 2); ++index) {
        append_string(bytes, index == 2 ? "table.weight" : "table.weight.tq_row_shard.0000" + std::to_string(index));
        append_string(bytes, wrong_dtype && index == 1 ? "F16" : "TQ");
        append_little<std::uint64_t>(bytes, payloads[index % 2].size());
    }
    bytes += payloads[0]; bytes += payloads[1];
    if (physical_alias) bytes += payloads[0];
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), bytes.size());
    require(bool(stream), "cannot write TQ row shard fixture");
}

void test_tq_shards(const std::filesystem::path& root) {
    using namespace mlx::core;
    using namespace mfq::metal;
    std::filesystem::create_directories(root);
    const nlohmann::json manifest = {{"table.weight",
        {"table.weight.tq_row_shard.00000", "table.weight.tq_row_shard.00001"}}};
    const auto path = root / "rows.mfq";
    write_tq_shards(path, manifest);
    const MfqContainer model(path);
    const auto initial = MlxResourceTelemetry::snapshot();
    MlxQuantizedRowStore store(model, "table.weight", 8, 513, 3, 2);
    const auto backing = model.record("table.weight.tq_row_shard.00000").nbytes
        + model.record("table.weight.tq_row_shard.00001").nbytes;
    require(MlxResourceTelemetry::snapshot().ple_payload_bytes == initial.ple_payload_bytes + backing,
        "TQ sharded table backing is missing from SSD telemetry");
    std::vector<array> expected;
    for (int index = 0; index < 2; ++index) {
        const auto payload = tq_payload(index + 1, index == 1);
        const auto blob = std::span(reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size());
        expected.push_back(MlxTqWeight::from_blob(blob, 513).embedding(arange(4, int32)));
    }
    const std::array<std::int64_t, 6> requested{3, 4, 7, 0, 4, 3};
    const auto reference = take(concatenate(expected, 0), array(requested.data(), {6}, int64), 0);
    require(max(abs(astype(store.gather(requested), float32) - astype(reference, float32))).item<float>() == 0.f,
        "TQ row shard boundaries or request order differ");
    const auto before = store.stats();
    require(before.rows_loaded == 4 && before.cache_hits == 2 && before.resident_rows == 3
        && before.resident_payload_bytes <= before.cache_limit_bytes, "TQ row shard LRU accounting differs");
    require(MlxResourceTelemetry::snapshot().dynamic_weight_bytes == initial.dynamic_weight_bytes + before.resident_payload_bytes,
        "TQ hot rows are missing from resident telemetry");
    const std::array<std::int64_t, 1> cached{4};
    store.gather(cached).eval();
    require(store.stats().rows_loaded == before.rows_loaded, "TQ shard boundary row did not hit its cache");
    store.clear();
    const auto cleared = MlxResourceTelemetry::snapshot();
    require(cleared.dynamic_weight_bytes == initial.dynamic_weight_bytes
        && cleared.ple_payload_bytes == initial.ple_payload_bytes + backing,
        "TQ clear confused resident rows with SSD backing");
    const std::array<std::int64_t, 1> left{3}, right{4};
    auto first = std::async(std::launch::async, [&] { return store.prepare_rows(left); });
    auto second = std::async(std::launch::async, [&] { return store.prepare_rows(right); });
    auto actual = concatenate({first.get().decode(), second.get().decode()}, 0);
    require(max(abs(astype(actual, float32) - astype(take(concatenate(expected, 0), array({3, 4}), 0), float32))).item<float>() == 0.f,
        "TQ asynchronous shard reads differ");
    for (const auto invalid : {-1, 8}) {
        bool rejected = false;
        const std::array<std::int64_t, 1> ids{invalid};
        try { store.prepare_rows(ids); } catch (const std::out_of_range&) { rejected = true; }
        require(rejected, "TQ logical row bounds are not enforced");
    }
    const std::vector<nlohmann::json> invalid_manifests{
        nlohmann::json::array(), {{"table.weight", nlohmann::json::array()}},
        {{"table.weight", {"table.weight.tq_row_shard.00000", "table.weight.tq_row_shard.00000"}}},
        {{"table.weight", {"missing"}}}, {{"table.weight", {1}}}};
    for (std::size_t index = 0; index < invalid_manifests.size() + 3; ++index) {
        const auto invalid_path = root / ("invalid-" + std::to_string(index) + ".mfq");
        write_tq_shards(invalid_path, index < invalid_manifests.size() ? invalid_manifests[index] : manifest,
            index == invalid_manifests.size(), index == invalid_manifests.size() + 1);
        bool rejected = false;
        try {
            MlxQuantizedRowStore invalid(MfqContainer(invalid_path), "table.weight",
                index == invalid_manifests.size() + 2 ? 7 : 8, 513, 3);
        } catch (const std::exception&) { rejected = true; }
        require(rejected, "TQ invalid row-shard manifest was accepted");
    }
}

std::vector<float> values(mlx::core::array value) {
    value = mlx::core::astype(value, mlx::core::float32);
    value.eval();
    return {value.data<float>(), value.data<float>() + value.size()};
}

void test_legacy_table_rejection(const std::filesystem::path& path) {
    for (const auto* dtype : {"NINT", "NVQ", "MFE", "MXFP4-SQ", "MXFP8-SQ", "FP8-128SQ"}) {
        write_fixture(path, dtype, "unsupported");
        bool rejected = false;
        try { mfq::metal::MfqContainer model(path); }
        catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("does not support tensor format") != std::string::npos;
        }
        require(rejected, "row-table loading admitted a legacy quantization format");
    }
}

void require_row(
    const std::vector<float>& source,
    std::size_t row,
    float expected) {
    const auto begin = source.begin() + static_cast<std::ptrdiff_t>(row * 32);
    require(
        std::all_of(begin, begin + 32, [expected](float value) {
            return std::fabs(value - expected) < 1e-6f;
        }),
        "decoded MXFP8 row mismatch");
}

void test_shared_engram_rows(const std::filesystem::path& root) {
    const std::string name = "model.block.1.associative_memory.embedding.weight";
    const auto shared = root / "Engram" / "model-EW-Engram-MXFP8-00002-of-00003.mfq";
    std::filesystem::create_directories(shared.parent_path());
    write_fixture(shared, "MXFP8", row_table_payload(), name, 1);
    for (const std::string precision : {"Q3", "Q2.5"}) {
        const auto directory = root / precision;
        std::filesystem::create_directories(directory);
        const auto first = directory / ("model-EW-" + precision + "-00001-of-00003.mfq");
        const auto last = directory / ("model-EW-" + precision + "-00003-of-00003.mfq");
        write_fixture(first, "MXFP8", row_table_payload(), "model.input.weight", 0);
        write_fixture(last, "MXFP8", row_table_payload(), "model.output.weight", 2);
        for (const auto& entry : {first, last}) {
            mfq::metal::MfqContainer model(entry);
            require(model.source_paths().size() == 3 &&
                model.record(name).source_path == std::filesystem::canonical(shared),
                "Metal did not resolve the original shared Engram file");
            const auto initial = mfq::metal::MlxResourceTelemetry::snapshot();
            mfq::metal::MlxQuantizedRowStore store(model, name, 4, 32, 2, 2);
            require(mfq::metal::MlxResourceTelemetry::snapshot().ple_payload_bytes
                == initial.ple_payload_bytes + model.record(name).nbytes,
                "shared MXFP8 Engram backing is missing from SSD telemetry");
            const std::array<std::int64_t, 4> request{3, 0, 3, 2};
            const auto output = values(store.gather(request));
            for (std::size_t row = 0; row < request.size(); ++row)
                require_row(output, row, request[row] % 2 ? 2.0f : 1.0f);
            require(store.stats().rows_loaded == 3 && store.cached_rows() == 2,
                "shared MXFP8 Engram did not keep bounded row residency");
            store.clear();
            require_row(values(store.prepare_rows(std::span(request).first(1)).decode()), 0, 2.0f);
            require(!std::filesystem::exists(directory / shared.filename()),
                "shared Engram was copied into a precision directory");
        }
    }
}

} // namespace

void test_large_mxfp8_rows(const std::filesystem::path& root) {
    using namespace mfq::metal;
    constexpr int rows = 16385;
    for (int width : {32, 256}) {
        const auto path = root / ("large-" + std::to_string(width) + ".mfq");
        const auto payload = row_table_payload(rows, width);
        write_fixture(path, "MXFP8", payload);
        MfqContainer model(path);
        MlxQuantizedRowStore store(model, "table.weight", rows, width, 17, 2);
        std::vector<std::int64_t> ids;
        for (int row = rows - 1; row >= 0; --row) ids.push_back(row);
        ids.push_back(rows - 1); ids.push_back(0);
        const auto result = values(store.prepare_rows(ids).decode());
        for (std::size_t index = 0; index < ids.size(); ++index)
            for (int column = 0; column < width; ++column)
                require(result[index * width + column] == (ids[index] % 2 ? 2.f : 1.f),
                    "batched MXFP8 gather changed row order or values");
        auto stats = store.stats();
        require(stats.cache_misses == rows && stats.cache_hits == 2
            && stats.row_requests == ids.size() && stats.resident_rows == 17
            && stats.bytes_read == std::uint64_t(rows) * (width + width / 32)
            && stats.read_calls < rows / 2,
            "batched MXFP8 reads were not coalesced or counted accurately");
        const std::array<std::int64_t, 2> cached{0, 1};
        const auto cached_values = values(store.prepare_rows(cached).decode());
        require(cached_values.front() == 1.f && cached_values[width] == 2.f
            && store.stats().read_calls == stats.read_calls,
            "batched MXFP8 gather did not retain its row cache");
        for (int count : {16383, 16384}) {
            MlxQuantizedRowStore boundary(model, "table.weight", rows, width, 0, 2);
            auto selected = boundary.prepare_rows(std::span<const std::int64_t>(ids.data(), count));
            require(selected.values.size() == std::size_t(count) * width
                && boundary.stats().cache_misses == count
                && (count < 16384 ? boundary.stats().read_calls == 2 * count
                    : boundary.stats().read_calls < count / 2),
                "MXFP8 row batching threshold changed the small-read path");
        }
        MlxQuantizedRowStore mixed(model, "table.weight", rows, width, 1, 2);
        const std::array<std::int64_t, 1> warm{0};
        (void)mixed.prepare_rows(warm);
        auto mixed_result = mixed.prepare_rows(ids);
        require(mixed.stats().cache_misses == rows && mixed.stats().cache_hits == 3,
            "batched MXFP8 gather lost resident or duplicated rows");
        for (std::size_t index = 0; index < ids.size(); ++index)
            require(float(mixed_result.values[index * width]) == (ids[index] % 2 ? 2.f : 1.f),
                "batched MXFP8 gather mixed resident and missing row indices");
        const auto previous_requests = store.stats().row_requests;
        const std::array<std::int64_t, 2> invalid{0, rows};
        bool rejected = false;
        try { (void)store.prepare_rows(invalid); }
        catch (const std::out_of_range&) { rejected = true; }
        require(rejected && store.stats().row_requests == previous_requests,
            "invalid MXFP8 batch mutated row statistics");
        for (bool bad_scale : {false, true}) {
            auto damaged = payload;
            damaged[bad_scale ? damaged.size() - 1 : 56 + (rows - 1) * width] =
                static_cast<char>(bad_scale ? 255 : 0x7f);
            const auto damaged_path = root / (bad_scale ? "bad-scale.mfq" : "bad-value.mfq");
            write_fixture(damaged_path, "MXFP8", damaged);
            MlxQuantizedRowStore bad(MfqContainer(damaged_path), "table.weight", rows, width, 17, 2);
            bool failed = false;
            try { (void)bad.prepare_rows(ids); }
            catch (const std::runtime_error& error) {
                failed = std::string(error.what()) == (bad_scale
                    ? "native MXFP8 row contains an invalid E8M0 scale"
                    : "native MXFP8 row contains an E4M3 NaN");
            }
            require(failed && bad.stats().rows_loaded == 0 && bad.stats().resident_rows == 0,
                "failed MXFP8 batch partially entered the cache");
        }
    }
}

void test_interleaved_mxfp8_rows(const std::filesystem::path& root) {
    using namespace mfq::metal;
    constexpr int rows = 257, width = 256;
    std::string payload;
    for (int row = 0; row < rows; ++row) {
        payload.append(width, char(row % 2 ? 0x40 : 0x38));
        for (int group = 0; group < 8; ++group) payload.push_back(char(125 + (row + group) % 5));
    }
    const auto write = [&](const std::filesystem::path& path, const std::string& bytes) {
        auto* tensors = ggml_init({1 << 20, nullptr, false});
        auto* file = gguf_init_empty();
        gguf_set_val_str(file, "general.architecture", "deepseek41");
        const nlohmann::json config{{"model_type", "deepseek_v41"}, {"text_config", {
            {"model_type", "deepseek_v41_text"}, {"hidden_size", 256}, {"vocab_size", 3},
            {"num_hidden_layers", 2}, {"engram_layer_ids", nlohmann::json::array()}}}};
        gguf_set_val_str(file, "deepseek41.config", config.dump().c_str());
        gguf_set_val_str(file, "deepseek41.engram.encoding", "e4m3_e8m0_32_row264");
        auto* tensor = ggml_new_tensor_2d(tensors, GGML_TYPE_I8, 264, rows);
        ggml_set_name(tensor, "blk.1.engram_embd.weight");
        std::memcpy(tensor->data, bytes.data(), bytes.size());
        gguf_add_tensor(file, tensor);
        require(gguf_write_to_file(file, path.string().c_str(), false), "interleaved Engram fixture write failed");
        gguf_free(file); ggml_free(tensors);
    };
    const auto path = root / "interleaved.gguf";
    const auto name = "model.block.1.associative_memory.embedding.weight";
    write(path, payload);
    const MfqContainer model(path);
    for (const int capacity : {0, 2, 17}) {
        MlxQuantizedRowStore store(model, name, rows, width, capacity, 2);
        for (const auto& ids : std::vector<std::vector<int64_t>>{{256, 0, 256, 1, 0}, {2}, {1, 256}}) {
            const auto actual = values(store.prepare_rows(ids).decode());
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const auto row = ids[i / width];
                const auto expected = std::ldexp(row % 2 ? 2.f : 1.f, -2 + (row + int(i % width / 32)) % 5);
                require(actual[i] == expected, "interleaved Engram scalar or cached rows differ");
            }
        }
        store.clear();
        std::vector<int64_t> ids;
        for (int row = rows - 1; row >= 0; --row) ids.push_back(row);
        const auto actual = values(store.prepare_rows(ids).decode());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const auto row = ids[i / width];
            const auto expected = std::ldexp(row % 2 ? 2.f : 1.f, -2 + (row + int(i % width / 32)) % 5);
            require(actual[i] == expected, "interleaved Engram batched rows differ");
        }
        require(store.cached_rows() <= std::size_t(capacity), "interleaved Engram exceeded its row budget");
    }
    auto invalid = payload;
    invalid[256] = char(255);
    const auto invalid_path = root / "interleaved-invalid.gguf";
    write(invalid_path, invalid);
    bool rejected = false;
    try {
        const MfqContainer bad(invalid_path);
        MlxQuantizedRowStore store(bad, name, rows, width, 0, 1);
        const std::array<int64_t, 1> ids{0};
        store.prepare_rows(ids);
    } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "interleaved Engram accepted a nonfinite scale");
}

void test_engram_prefetch_order(const std::filesystem::path& path) {
    using namespace mfq::metal;
    struct Record { std::string name, dtype, payload; };
    std::vector<Record> records;
    const auto dense = [&](const std::string& name, int rows, int width) {
        std::string payload;
        append_little<std::uint32_t>(payload, 2);
        append_little<std::int64_t>(payload, rows);
        append_little<std::int64_t>(payload, width);
        const std::vector<float> data(rows * width, 0.125f);
        payload.append(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        records.push_back({name, "F32", std::move(payload)});
    };
    for (int layer : {1, 14}) {
        const auto prefix = "model.block." + std::to_string(layer) + ".associative_memory";
        dense(prefix + ".query.weight", 4, 4);
        dense(prefix + ".key.weight", 4, 4);
        dense(prefix + ".projection.weight", 20, 32);
        records.push_back({prefix + ".embedding.weight", "MXFP8", row_table_payload()});
    }
    std::string bytes{"MFQ1", 4};
    append_little<std::uint32_t>(bytes, 2);
    append_string(bytes, "unit-test");
    append_little<std::uint32_t>(bytes, 0);
    append_little<std::uint32_t>(bytes, records.size());
    for (const auto& record : records) {
        append_string(bytes, record.name);
        append_string(bytes, record.dtype);
        append_little<std::uint64_t>(bytes, record.payload.size());
    }
    for (const auto& record : records) bytes.append(record.payload);
    { std::ofstream file(path, std::ios::binary);
      file.write(bytes.data(), bytes.size());require(bool(file), "cannot write Engram fixture"); }
    DeepseekV41Config config;
    config.hidden = 4; config.hc_mult = 4;
    config.engram_layer_ids = {1, 14}; config.engram_num_embeddings = {4, 4};
    config.engram_head_dim = 32; config.engram_max_ngram_size = 2; config.engram_n_heads = 1;
    MfqContainer model(path);
    auto first = MlxDeepseekV41Engram::load(model, config, 1);
    auto second = MlxDeepseekV41Engram::load(model, config, 14);
    const auto batch = [](int tokens = 1) {
        DeepseekV41EngramHashBatch value;
        value.batch = value.hash_columns = 1; value.tokens = tokens;
        value.layers = 2; value.values.resize(2 * tokens);
        std::fill(value.values.begin() + tokens, value.values.end(), 1);
        value.prefetched_rows.resize(2);
        return value;
    };
    auto normal = batch();
    first.prefetch(normal); second.prefetch(normal);
    require_row(values(normal.prefetched_rows[0].get().decode()), 0, 1.f);
    require_row(values(normal.prefetched_rows[1].get().decode()), 0, 2.f);
    for (int tokens : {1, 2, 3, 6, 8}) {
        auto short_input = batch(tokens);
        std::promise<MlxQuantizedRowBatch> short_failure;
        short_input.prefetched_rows[0] = short_failure.get_future().share();
        short_failure.set_exception(std::make_exception_ptr(std::runtime_error("earlier Engram read failed")));
        second.prefetch(short_input);
        require_row(values(short_input.prefetched_rows[1].get().decode()), 0, 2.f);
    }
    auto failed = batch(16);
    std::promise<MlxQuantizedRowBatch> failure;
    failed.prefetched_rows[0] = failure.get_future().share();
    failure.set_exception(std::make_exception_ptr(std::runtime_error("earlier Engram read failed")));
    const auto requests = second.ssd_stats().row_requests;
    second.prefetch(failed);
    bool propagated = false;
    try { (void)failed.prefetched_rows[1].get(); }
    catch (const std::runtime_error& error) { propagated = std::string(error.what()) == "earlier Engram read failed"; }
    require(propagated && second.ssd_stats().row_requests == requests,
        "later Engram prefetch ignored its failed predecessor");
    auto reverse = batch(16);
    std::promise<MlxQuantizedRowBatch> later_failure;
    reverse.prefetched_rows[1] = later_failure.get_future().share();
    later_failure.set_exception(std::make_exception_ptr(std::runtime_error("later Engram read failed")));
    first.prefetch(reverse);
    require_row(values(reverse.prefetched_rows[0].get().decode()), 0, 1.f);
}

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("mfq-mxfp8-row-store-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(root);
        test_large_mxfp8_rows(root);
        test_interleaved_mxfp8_rows(root);
        test_engram_prefetch_order(root / "engram-prefetch.mfq");
        const auto initial = mfq::metal::MlxResourceTelemetry::snapshot();
        test_tq_shards(root / "tq");
        test_shared_engram_rows(root / "shared");
        const auto released = mfq::metal::MlxResourceTelemetry::snapshot();
        require(released.ple_payload_bytes == initial.ple_payload_bytes
            && released.dynamic_weight_bytes == initial.dynamic_weight_bytes,
            "destroyed row stores leaked resource telemetry");
        const auto path = root / "fixture.mfq";
        write_fixture(path);
        mfq::metal::MfqContainer model(path);
        mfq::metal::MlxQuantizedRowStore store(
            model, "table.weight", 4, 32, 2, 2);

        const std::array<std::int64_t, 3> request{0, 1, 0};
        auto decoded = store.gather(request);
        require(
            decoded.shape() == mlx::core::Shape{3, 32},
            "MXFP8 gather shape mismatch");
        const auto output = values(std::move(decoded));
        require_row(output, 0, 1.0f);
        require_row(output, 1, 2.0f);
        require_row(output, 2, 1.0f);
        auto stats = store.stats();
        require(
            stats.row_requests == 3 && stats.cache_hits == 1 &&
            stats.cache_misses == 2 && stats.rows_loaded == 2 &&
            stats.bytes_read == 66 && stats.read_calls == 4 &&
            stats.resident_rows == 2 &&
            stats.resident_payload_bytes == 128 &&
            stats.cache_limit_bytes == 128,
            "MXFP8 row cache statistics mismatch");

        (void)store.gather(std::span<const std::int64_t>(request.data(), 1));
        stats = store.stats();
        require(
            stats.rows_loaded == 2 && stats.cache_hits == 2,
            "resident MXFP8 row did not hit the cache");

        store.clear();
        const std::array<std::int64_t, 1> even_request{2};
        const std::array<std::int64_t, 1> odd_request{3};
        auto even_future = std::async(std::launch::async, [&] {
            return store.prepare_rows(even_request);
        });
        auto odd_future = std::async(std::launch::async, [&] {
            return store.prepare_rows(odd_request);
        });
        require_row(values(even_future.get().decode()), 0, 1.0f);
        require_row(values(odd_future.get().decode()), 0, 2.0f);

        const auto previous_stream = mlx::core::default_stream(mlx::core::Device::gpu);
        const auto runtime_stream = mlx::core::new_thread_unsafe_stream(mlx::core::Device::gpu);
        mlx::core::set_default_stream(runtime_stream);
        test_tq_shards(root / "tq-runtime-stream");
#if !TYLOINFER_ENABLE_MFQ_COMPAT
        test_legacy_table_rejection(root / "legacy.mfq");
#endif
        mlx::core::synchronize();
        mlx::core::set_default_stream(previous_stream);

        std::filesystem::remove_all(root);
        std::cout << "MXFP8/TQ row store passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
        std::cerr << "MXFP8 row store failed: " << error.what() << '\n';
        return 1;
    }
}
