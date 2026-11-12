#include "tq_load.h"
#include "tensor.h"
#include "engine/models/source.h"
#include "engine/weights/tq/load.h"
#include "engine/weights/tq/source.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::diagnostics::numerics {
namespace {
/** Publish a scalar observed from the actual loader. */
void boolean(Case& value, const std::string& name, bool observed) {
    const std::uint8_t byte = observed ? 1 : 0;
    value.output(name, Json::array(), "bool", &byte, sizeof(byte));
}

/** Check that a source record contains exactly the persisted fixed payload. */
bool same_payload(const tyloi::source::Source& source, const std::string& name,
                  const std::vector<std::uint8_t>& expected) {
    const auto bytes = source.read(name);
    if (bytes.size() != expected.size()) return false;
    for (std::size_t i = 0; i < bytes.size(); ++i)
        if (std::to_integer<std::uint8_t>(bytes[i]) != expected[i]) return false;
    return true;
}
} // namespace

bool run_tq_load(Case& value) {
    const auto& data = value.data();
    const auto op = data.at("op").get<std::string>();
    if (op != "tq_load_valid" && op != "tq_load_legacy" && op != "tq_load_directory")
        return false;
    if (data.at("source").value("generator", std::string()) != "tests/numerics/tq_load_ref.py" ||
        data.value("actual_role", std::string("native")) != "native")
        throw std::invalid_argument("unsupported native TQ loading case");
    const auto& config = data.at("config");
    const auto name = config.at("tensor").get<std::string>();
    const auto model = value.asset_path("model");
    const auto payload = value.asset("payload");
    TensorWork work(value);
    Json details = Json::object();
    if (op == "tq_load_legacy") {
        bool rejected = false;
        try {
            auto source = tyloi::open_model_source(model);
            try {
                (void)tyloi::tq_table(*source, name);
            } catch (const std::invalid_argument& error) {
                const std::string message = error.what();
                if (message != "expected TQ tensor: " + name) throw;
                rejected = true;
                details = {{"stage", "tq_table"}, {"source_accepted", true}, {"message", message}};
            }
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            if (message != "TyloInfer does not support tensor format " +
                               config.at("dtype").get<std::string>() + ": " + name)
                throw;
            rejected = true;
            details = {{"stage", "source"}, {"source_accepted", false}, {"message", message}};
        }
        boolean(value, "rejected", rejected);
    } else {
        auto source = tyloi::open_model_source(model);
        if (!same_payload(*source, name, payload))
            throw std::invalid_argument("TQ source record differs from the fixed payload");
        if (op == "tq_load_directory") {
            bool rejected = false;
            try {
                (void)tyloi::tq_table(*source, name);
            } catch (const std::invalid_argument& error) {
                const std::string message = error.what();
                if (message != "invalid TQ expert container span" &&
                    message != "TQ expert container length differs")
                    throw;
                rejected = true;
                details = {{"stage", "tq_table"}, {"message", message}};
            }
            boolean(value, "rejected", rejected);
        } else {
            const auto shape = tyloi::tq_shape(*source, name);
            auto table = tyloi::tq_table(*source, name);
            value.output("shape", Json::array({shape.size()}), "int64", shape.data(),
                         shape.size() * sizeof(std::int64_t));
            boolean(value, "payload_equal", same_payload(*source, name, payload));
            const auto expected_shape = config.at("logical_shape").get<std::vector<std::int64_t>>();
            if (expected_shape.size() != 3 || expected_shape[0] <= 0 || expected_shape[1] <= 0 ||
                expected_shape[0] > INT32_MAX || expected_shape[1] > INT32_MAX)
                throw std::invalid_argument("invalid TQ case logical shape");
            bool rows_equal = table->rows() == std::uint64_t(expected_shape[0] * expected_shape[1]) &&
                              table->width() == expected_shape[2];
            const auto parts = data.at("assets").contains("part1") ? 2 : 1;
            const auto rows_per_part = table->rows() / parts;
            for (int i = 0; i < parts; ++i) {
                const auto expected = value.asset("part" + std::to_string(i));
                rows_equal = rows_equal && table->slice_rows(i * rows_per_part, (i + 1) * rows_per_part) == expected;
            }
            boolean(value, "rows_equal", rows_equal);
            if (!same_payload(*source, "projection", value.asset("part0")))
                throw std::invalid_argument("TQ projection differs from the fixed row payload");
            const auto projection = load_tq_weight(*source, "projection", work.device().index, false);
            const bool loaded = projection.defined() && projection.device().is_cpu() &&
                                projection.size(0) == std::int64_t(rows_per_part) &&
                                projection.size(1) == expected_shape[2] && projection.image().defined();
            boolean(value, "projection_loaded", loaded);
            details = {{"table_segments", table->segment_count()}, {"projection_device", "cpu"}};
        }
    }
    work.finish("open_model_source+tq_table", std::move(details));
    return true;
}
} // namespace tyloi::diagnostics::numerics
