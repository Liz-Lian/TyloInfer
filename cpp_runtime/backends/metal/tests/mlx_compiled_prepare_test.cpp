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
        mfq::metal::MlxKernelPreparation preparation(128, 128);
        for (const auto& output : outputs) preparation.add(output);
        const auto jobs = preparation.size();
        if (!jobs) throw std::runtime_error("compiled leaf was not prepared");
        for (const auto& output : outputs) preparation.add(output);
        if (preparation.size() != jobs) throw std::runtime_error("compiled leaf preparation was duplicated");
        preparation.finish();
        if (input.is_available() || other.is_available() || outputs[0].is_available() || outputs[1].is_available())
            throw std::runtime_error("preparation executed original input or output graphs");
        const auto error = max(abs(outputs[0] - (sin(input) + exp(other)))).item<float>();
        if (!std::isfinite(error) || error > .00001f) throw std::runtime_error("compiled leaf preparation changed the calculation");
        std::cout << "Compiled leaf preparation uses isolated synthetic inputs and preserves original graphs\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
