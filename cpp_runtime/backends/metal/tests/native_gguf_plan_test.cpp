#include "ops/Linear.hpp"
#include "ops/MoE.hpp"

#include <iostream>
#include <string>
#include <stdexcept>

namespace {
void require(bool condition) {
    if (!condition) throw std::runtime_error("native plan contract failed");
}
template<class F> void rejects(F&& call) {
    try { call(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("native plan accepted an invalid row count");
}
void check_gate_up_tables(uint32_t gate, uint32_t up, uint32_t shared_gate, uint32_t shared_up, bool mixed) {
    using namespace splash::ops;
    std::size_t slot = 0;
    const auto buffer = [&](uint64_t bytes) { return splash::metal::MetalBuffer(slot++, bytes); };
    const auto segment = [&](uint32_t format, uint32_t n, uint32_t k) {
        const auto& f = kQuantFormats[format];
        const uint64_t groups = uint64_t(n) * k / 32;
        return QuantizedSegment::planes(format, n, k, buffer(groups * f.plane0_bytes),
            buffer(groups * f.plane1_bytes), buffer(groups / f.meta_groups * f.meta_bytes));
    };
    const MoeShape shape{1024, 16, 2, 256, gate, true};
    BlockMoeWeights weights;
    weights.router = QuantizedSegment::floats(shape.experts, shape.hiddenSize,
        buffer(uint64_t(shape.experts) * shape.hiddenSize * 4));
    weights.sharedScalarGate = QuantizedSegment::floats(1, shape.hiddenSize, buffer(shape.hiddenSize * 4));
    weights.slots = shape.experts; weights.slotTable = buffer(shape.experts * 4);
    weights.gate = {segment(gate, shape.experts * shape.expertIntermediateSize, shape.hiddenSize),
        segment(shared_gate, shape.expertIntermediateSize, shape.hiddenSize)};
    weights.up = {segment(up, shape.experts * shape.expertIntermediateSize, shape.hiddenSize),
        segment(shared_up, shape.expertIntermediateSize, shape.hiddenSize)};
    weights.down = {segment(gate, shape.experts * shape.hiddenSize, shape.expertIntermediateSize),
        segment(shared_gate, shape.hiddenSize, shape.expertIntermediateSize)};
    for (const uint32_t rows : {3u, 65u, 4096u}) {
        const MoeConfig config{rows == 3 ? MoeExpertTile::M8 : MoeExpertTile::M32,
            MoeGgufTile::Staged, FloatTile::Simdgroup};
        const auto plan = rows == 3 ? MoE::decodeRowsPlan(shape, rows, config) : MoE::prefillPlan(shape, rows, config);
        const uint64_t input_bytes = uint64_t(rows) * shape.hiddenSize * 2;
        MoeBuffers buffers{buffer(input_bytes), buffer(input_bytes), buffer(input_bytes)};
        for (const auto& field : kMoeScratchFields)
            buffers.scratch.*field.buffer = buffer(plan.workspace().*field.bytes);
        splash::metal::CommandGraph graph;
        MoE::add(graph, buffers, weights, plan);
        int gate_up = 0;
        for (const auto& dispatch : graph.dispatches())
            if (dispatch.pipelineName.starts_with("moe_expert_gguf_m") &&
                dispatch.pipelineName.find("_gate_up") != std::string::npos) {
                require(dispatch.pipelineName.ends_with("_mixed") == mixed);
                ++gate_up;
            }
        require(gate_up == 1);
    }
}
}

int main() {
    try {
        using namespace splash::ops;
        check_gate_up_tables(GGUF_FMT_Q4K, GGUF_FMT_Q5K, GGUF_FMT_Q4K, GGUF_FMT_Q4K, false);
        check_gate_up_tables(GGUF_FMT_Q5K, GGUF_FMT_IQ3XXS, GGUF_FMT_Q4K, GGUF_FMT_Q4K, false);
        check_gate_up_tables(GGUF_FMT_Q5K, GGUF_FMT_IQ4XS, GGUF_FMT_Q4K, GGUF_FMT_Q4K, true);
        check_gate_up_tables(GGUF_FMT_Q4K, GGUF_FMT_Q5K, GGUF_FMT_IQ4XS, GGUF_FMT_IQ4XS, false);
        check_gate_up_tables(GGUF_FMT_Q4K, GGUF_FMT_Q5K, GGUF_FMT_Q4K, GGUF_FMT_IQ4XS, true);
        check_gate_up_tables(GGUF_FMT_AF4G64, GGUF_FMT_AF8G64, GGUF_FMT_AF4G64, GGUF_FMT_AF8G64, true);
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
                for (uint32_t rows = 1; rows <= 32; ++rows) {
                    const auto decode = MoE::decodeRowsPlan(shape, rows, decode_config);
                    require(decode.rows() == rows && decode.tileRows() == 8 && decode.phase() == MoePhase::Decode);
                }
                rejects([&] { (void)MoE::decodeRowsPlan(shape, 0, decode_config); });
                rejects([&] { (void)MoE::decodeRowsPlan(shape, 33, decode_config); });
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
