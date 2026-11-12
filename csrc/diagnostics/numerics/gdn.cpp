#include "gdn.h"
#include "tensor.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/native/env.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace tyloi::diagnostics::numerics {
namespace {
/** Restore the original decode column setting after the fixed CUDA case. */
class ColumnSetting {
  public:
    explicit ColumnSetting(bool enabled) : enabled_(enabled) {
        if (enabled_) {
            if (const auto* old = tyloi::cuda::env_lookup("TYLOINFER_GDN_COLUMN"))
                old_ = old;
            tyloi::cuda::env_set("TYLOINFER_GDN_COLUMN", "1");
        }
    }
    ~ColumnSetting() noexcept {
        if (!enabled_)
            return;
        try {
            if (old_)
                tyloi::cuda::env_set("TYLOINFER_GDN_COLUMN", old_->c_str());
            else
                tyloi::cuda::env_clear("TYLOINFER_GDN_COLUMN");
        } catch (const std::exception& error) {
            std::cerr << "GDN environment cleanup: " << error.what() << '\n';
        }
    }

  private:
    bool enabled_;
    std::optional<std::string> old_;
};
} // namespace

bool run_gdn(Case& value) {
    const auto& data = value.data();
    if (data.at("op") != "gated_delta_net")
        return false;
    const auto source = data.at("source").at("test").get<std::string>();
    if (source == "tests/test_kernels_ops.py::test_gdn_matches_reference")
        throw std::invalid_argument("CPU GDN case has no native CPU producer in this worker");
    const bool column =
        source == "tests/test_kernels_cuda.py::test_gdn_cuda_decode_column_path_matches_ref";
    if (!column && source != "tests/test_kernels_cuda.py::test_gdn_cuda_scalar_gate_matches_ref" &&
        source != "tests/test_kernels_cuda.py::test_gdn_cuda_kda_matches_ref" &&
        source != "tests/test_kernels_cuda.py::test_gdn_cuda_d128_shared_mem")
        throw std::invalid_argument("unsupported original CUDA GDN case");
    const auto& config = data.at("config");
    if (config.at("cuda_column_path").get<bool>() != column ||
        config.at("initial_state") != (column ? "random" : "zero"))
        throw std::invalid_argument("GDN initial state or column policy differs from source");
    ColumnSetting setting(column);
    TensorWork work(value);
    auto q = work.input("q");
    auto k = work.input("k");
    auto v = work.input("v");
    auto g = work.input("g");
    auto beta = work.input("beta");
    MfqOptional<tyloi::DenseTensor> state = mfq_nullopt;
    if (column) {
        state = work.input("state");
    } else {
        const auto initial = value.input("state");
        if (!std::all_of(initial.begin(), initial.end(),
                         [](std::uint8_t byte) { return byte == 0; }))
            throw std::invalid_argument("GDN omitted initial state requires the saved zero state");
    }
    auto result = gdn_cuda(q, k, v, g, beta, state);
    if (result.size() != 2)
        throw std::runtime_error("GDN returned an invalid observation count");
    auto output = work.keep(result[0]);
    auto final_state = work.keep(result[1]);
    work.output("output", output);
    if (data.at("outputs").contains("state"))
        work.output("state", final_state);
    work.finish("gdn_cuda", {{"column_override", column}}, Json::object());
    return true;
}
} // namespace tyloi::diagnostics::numerics
