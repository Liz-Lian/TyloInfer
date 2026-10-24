#include "mfq_container.h"
#include "mlx_tq.h"
#include "mlx_platform.h"

#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlx::core;
using namespace mfq::metal;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: tyloinfer-metal-tq-benchmark model.tq cases.json");
        MfqContainer model(argv[1]);
        std::ifstream config_file(argv[2]);
        const auto config = nlohmann::json::parse(config_file);
        const int repeats = config.value("repeats", 3);
        if (repeats < 1 || repeats > 100) throw std::invalid_argument("invalid repeat count");
        for (const auto& entry : config.at("cases")) {
            const auto names = entry.at("weights").get<std::vector<std::string>>();
            if (names.empty()) throw std::invalid_argument("benchmark case has no weights");
            std::vector<MlxTqWeight> weights;
            for (const auto& name : names) {
                auto weight = MlxTqWeight::load(model, name);
                if (entry.contains("expert")) weight = weight.expert_view(entry.at("expert").get<int>());
                weights.push_back(std::move(weight));
            }
            const auto operation = entry.value("operation", std::string("matmul"));
            if (operation != "matmul" && operation != "routed" && operation != "swiglu" && operation != "reduce")
                throw std::invalid_argument("unknown benchmark operation");
            if ((operation == "swiglu" && weights.size() != 2)
                || ((operation == "routed" || operation == "reduce") && weights.size() != 1))
                throw std::invalid_argument("benchmark weight count differs from operation");
            const int width = weights.front().input_size();
            const int experts = weights.front().experts();
            const int routes = entry.value("routes", 1);
            const float limit = entry.value("limit", 0.f);
            const bool gelu = entry.value("gelu", false);
            const bool repeated_routes = entry.value("repeated_routes", false);
            if (repeated_routes && operation == "matmul")
                throw std::invalid_argument("repeated routes require an expert operation");
            if (routes < 1 || routes > experts) throw std::invalid_argument("invalid route count");
            const auto dtype_name = entry.value("dtype", std::string("float16"));
            const auto dtype = dtype_name == "float16" ? float16 : dtype_name == "bfloat16" ? bfloat16 : float32;
            if (dtype_name != "float16" && dtype_name != "bfloat16" && dtype_name != "float32")
                throw std::invalid_argument("invalid activation dtype");
            for (int tokens : entry.at("tokens").get<std::vector<int>>()) {
                if (tokens < 1 || tokens > 131072) throw std::invalid_argument("invalid token count");
                if (repeated_routes && tokens > 16)
                    throw std::invalid_argument("repeated-route benchmark supports at most sixteen tokens");
                const int steps = entry.value("steps", tokens <= 16 ? 48 : tokens <= 256 ? 8 : 3);
                if (steps < 1 || steps > 4096) throw std::invalid_argument("invalid step count");
                const bool routed_input = entry.value("routed_input", operation == "reduce");
                const auto elements = std::int64_t(tokens) * (routed_input ? routes : 1) * width;
                if (elements > std::numeric_limits<int>::max()) throw std::invalid_argument("benchmark input exceeds backend dimension limit");
                auto input = reshape(astype(sin(arange(int(elements), float32)) * .1f, dtype),
                    routed_input ? Shape{tokens, routes, width} : Shape{tokens, width});
                input.eval();
                std::mt19937 random(20261023);
                std::vector<array> schedules;
                std::vector<std::optional<array>> reuse_plans;
                for (int step = 0; step < steps; ++step) {
                    std::vector<std::int32_t> ids;
                    for (int token = 0; token < tokens; ++token) {
                        if (repeated_routes && token) {
                            for (int route = 0; route < routes; ++route) ids.push_back(ids[route]);
                            continue;
                        }
                        std::vector<int> selected;
                        while (int(selected.size()) < routes) {
                            const int expert = random() % experts;
                            if (std::find(selected.begin(), selected.end(), expert) == selected.end()) selected.push_back(expert);
                        }
                        ids.insert(ids.end(), selected.begin(), selected.end());
                    }
                    schedules.emplace_back(ids.begin(), Shape{tokens, routes}, int32);
                    schedules.back().eval();
                    reuse_plans.push_back(repeated_routes
                        ? make_tq_reuse_plan(ids, tokens, routes, experts) : std::nullopt);
                }
                auto route_weights = full({tokens, routes}, 1.f / routes, float32);
                route_weights.eval();
                auto call = [&](const array& source, int step) -> std::vector<array> {
                    const auto& ids = schedules[step % steps];
                    const auto& plan = reuse_plans[step % steps];
                    const auto* reuse = plan ? &*plan : nullptr;
                    if (operation == "matmul") return MlxTqWeight::matmul_many(weights, source);
                    if (operation == "routed") return {weights.front().routed_matmul(source, ids, reuse)};
                    if (operation == "reduce") return {weights.front().routed_matmul_reduce(source, ids, route_weights, reuse)};
                    return {weights.front().swiglu(weights.back(), source, ids, limit, gelu, reuse)};
                };
                for (int step = 0; step < std::min(steps, 4); ++step) {
                    auto output = call(input, step);
                    eval(output);
                    for (const auto& value : output)
                        if (!all(isfinite(value)).item<bool>()) throw std::runtime_error("nonfinite benchmark output");
                }
                const bool chained = entry.value("chained", config.value("chained", false));
                auto gain = array(.0001f, dtype);
                gain.eval();
                auto feedback = [&](const std::vector<array>& output) {
                    auto values = reshape(output.front(), {-1});
                    if (values.size() < input.size()) values = tile(values, {int((input.size() + values.size() - 1) / values.size())});
                    values = reshape(slice(values, {0}, {int(input.size())}), input.shape());
                    return add(input, multiply(values, gain));
                };
                if (chained) {
                    auto source = input;
                    for (int step = 0; step < std::min(steps, 4); ++step) source = feedback(call(source, step));
                    source.eval();
                }
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    double build_us = 0., eval_us = 0.;
                    auto source = input;
                    const auto chain_start = Clock::now();
                    for (int step = 0; step < steps; ++step) {
                        const auto start = Clock::now();
                        auto output = call(chained ? source : input, step);
                        if (chained) {
                            source = feedback(output);
                            continue;
                        }
                        const auto built = Clock::now();
                        eval(output);
                        const auto done = Clock::now();
                        build_us += std::chrono::duration<double, std::micro>(built - start).count();
                        eval_us += std::chrono::duration<double, std::micro>(done - built).count();
                    }
                    if (chained) {
                        const auto built = Clock::now();
                        source.eval();
                        build_us = std::chrono::duration<double, std::micro>(built - chain_start).count();
                        eval_us = std::chrono::duration<double, std::micro>(Clock::now() - built).count();
                    }
                    std::cout << nlohmann::json{{"name", entry.at("name")}, {"operation", operation},
                        {"M", tokens}, {"K", width}, {"N", weights.front().output_size()}, {"experts", experts},
                        {"routes", routes}, {"dtype", dtype_name}, {"repeat", repeat}, {"steps", steps},
                        {"build_us", build_us / steps}, {"eval_us", eval_us / steps}, {"wall_us", (build_us + eval_us) / steps},
                        {"nax_disabled", mlx_metal_nax_disabled()}, {"chained", chained},
                        {"repeated_routes", repeated_routes},
                        {"boundary", chained ? "dependent projections with input feedback; host graph, dispatch and one final GPU wait; load and compilation excluded"
                            : "host graph construction, dispatch and GPU completion per call; load and compilation excluded"}}
                        << std::endl;
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
