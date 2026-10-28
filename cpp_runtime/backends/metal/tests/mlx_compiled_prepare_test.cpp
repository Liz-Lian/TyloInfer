#include "mlx_kernel_prepare.h"
#include <mlx/mlx.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace mlx::core;
int main() {
    try {
        set_default_device(Device::gpu);
        auto function = compile([](const std::vector<array>& inputs) {
            return std::vector<array>{sin(inputs[0]) + exp(inputs[1]), tanh(inputs[0] + inputs[1])};
        });
        const auto input = reshape(sin(arange(48, float32)), {1, 48});
        const auto other = reshape(cos(arange(48, float32)), {1, 48});
        const auto outputs = function({input, other});
        const auto destination = reshape(sin(arange(32768 * 48, float32)), {32768, 48});
        const auto indices = zeros({1, 48}, int32);
        const auto scattered = put_along_axis(destination, indices, other, 0);
        const auto gathered = take(input, array({0, 0}, int32), 0);
        mfq::metal::MlxKernelPreparation preparation(128, 128);
        for (const auto& output : outputs) preparation.add(output);
        preparation.add(scattered);
        preparation.add(gathered);
        const auto jobs = preparation.size();
        if (!jobs) throw std::runtime_error("compiled leaf was not prepared");
        for (const auto& output : outputs) preparation.add(output);
        preparation.add(scattered);
        preparation.add(gathered);
        if (preparation.size() != jobs) throw std::runtime_error("compiled leaf preparation was duplicated");
        preparation.finish();
        if (input.is_available() || other.is_available() || outputs[0].is_available() || outputs[1].is_available() ||
            destination.is_available() || indices.is_available() || scattered.is_available() || gathered.is_available())
            throw std::runtime_error("preparation executed original input or output graphs");
        const auto error = max(abs(outputs[0] - (sin(input) + exp(other)))).item<float>();
        if (!std::isfinite(error) || error > .00001f) throw std::runtime_error("compiled leaf preparation changed the calculation");
        const auto scatter_error = max(abs(scattered - concatenate({other,
            slice(destination, {1, 0}, {32768, 48})}, 0))).item<float>();
        if (!std::isfinite(scatter_error) || scatter_error > .00001f)
            throw std::runtime_error("scatter preparation changed the calculation");
        if (max(abs(gathered - concatenate({input, input}, 0))).item<float>() != 0.f)
            throw std::runtime_error("gather preparation changed the calculation");
        std::cout << "Compiled leaf preparation uses isolated synthetic inputs and preserves original graphs\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
