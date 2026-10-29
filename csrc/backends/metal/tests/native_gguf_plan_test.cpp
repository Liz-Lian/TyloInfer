#include "ops/Linear.hpp"
#include "ops/MoE.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition) {
    if (!condition) throw std::runtime_error("native prefill plan contract failed");
}
template<class F> void rejects(F&& call) {
    try { call(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("native prefill plan accepted an invalid row count");
}
}

int main() {
    try {
        using namespace splash::ops;
        static_assert(SPLASH_PREFILL_TOKEN_BUDGET == 4096);
        for (const auto tile : {MoeGgufTile::Staged, MoeGgufTile::Register}) {
            for (const auto top_k : {1u, 10u, 16u}) {
                const MoeShape shape{2560, 512, top_k, 640, GGUF_FMT_AF2G64, true};
                const MoeConfig config{moeGgufPrefillTile(shape, 4096, tile), tile, FloatTile::Simdgroup};
                const auto plan = MoE::prefillPlan(shape, 4096, config);
                require(plan.rows() == 4096 && plan.phase() == MoePhase::Prefill);
                const auto& work = plan.workspace();
                const auto grouped_rows = work.groupedInputBytes / (shape.hiddenSize * 2);
                require(grouped_rows == std::uint64_t(plan.maximumTiles()) * plan.tileRows());
                require(grouped_rows >= std::uint64_t(4096) * shape.routesPerToken());
                require(work.selectedExpertsBytes == std::uint64_t(4096) * shape.routesPerToken() * 4);
                rejects([&] { (void)MoE::prefillPlan(shape, 0, config); });
                rejects([&] { (void)MoE::prefillPlan(shape, 4097, config); });
                const MoeConfig decode_config{MoeExpertTile::M8, tile, FloatTile::Simdgroup};
                for (const auto lanes : {1u, 2u, 3u, 4u}) {
                    const auto decode = MoE::decodePlan(shape, lanes, decode_config);
                    require(decode.rows() == lanes * 8 && decode.tileRows() == 8);
                }
            }
        }
        for (const auto family : {9u, 10u}) {
            splash::DeviceCapabilities device;
            device.appleGpuFamily = family;
            device.gpuCoreCount = 40;
            const Linear linear(device);
            for (const auto matrix : {LinearMatrix{320, 10240}, LinearMatrix{10240, 320}}) {
                const auto plan = linear.plan({matrix, 4096, LinearPhase::Prefill, LinearEpilogue::None});
                require(plan.workload().rows == 4096 && plan.storageRows() == 4096);
                rejects([&] { (void)linear.plan({matrix, 4097, LinearPhase::Prefill, LinearEpilogue::None}); });
                for (const auto lanes : {1u, 2u, 3u, 4u}) {
                    const auto decode = linear.plan({matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::None});
                    require(decode.workload().rows == lanes * 8);
                }
            }
        }
        std::cout << "native 4k prefill and decode plan contracts passed without GPU execution\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
