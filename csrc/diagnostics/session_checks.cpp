#include "model_checks.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/deepseek_v4/model.h"
#include "engine/models/deepseek_v4/state.h"
#include "engine/models/glm_dsa/model.h"
#include "engine/models/glm_dsa/state.h"
#include "engine/models/session/snapshot.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
/** Keep synthetic model identities, request storage and snapshots alive until drain. */
struct Work {
    tyloi::cuda::Program& program;
    tyloi::cuda::deepseek_v4::Model dsv4_model;
    tyloi::cuda::glm_dsa::Model glm_model;
    std::unique_ptr<tyloi::cuda::deepseek_v4::State> dsv4;
    std::unique_ptr<tyloi::cuda::glm_dsa::State> glm;
    TextSessionState dsv4_saved, glm_saved;
    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& error) { std::cerr << "session_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "session_cleanup error=unknown\n"; }
    }
};

void require_equal(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool same(const tyloi::Tensor& value, const tyloi::Tensor& saved) {
    return value.dense().equal(saved.dense());
}
} // namespace

int run_text_session_state_check(tyloi::cuda::Program& program) {
    const auto device = program.executor().device();
    auto cuda_float = tyloi::TensorOptions().device(device).dtype(tyloi::kFloat32);
    Work work{program};
    const auto values = [&](std::vector<int64_t> shape, tyloi::ScalarType dtype, float offset) {
        int64_t elements = 1;
        for (const auto extent : shape) elements *= extent;
        return (tyloi::cuda::arange(elements, cuda_float) + offset)
            .reshape(shape).to(dtype).contiguous();
    };

    // Only cache layout is needed: no model weights or numerical forward are involved.
    work.dsv4_model.layers.resize(1);
    auto& dsv4_layout = work.dsv4_model.layers[0];
    dsv4_layout.device = device.index;
    dsv4_layout.head_dim = 8;
    dsv4_layout.max_positions = 64;
    dsv4_layout.compressor.ratio = dsv4_layout.indexer_compressor.ratio = 4;
    dsv4_layout.compressor.head_dim = 8;
    dsv4_layout.indexer_compressor.head_dim = 4;
    dsv4_layout.indexer_compressor.overlap = true;
    dsv4_layout.indexer_compressor.cache_quant_mode = 2;
    work.dsv4 = std::make_unique<tyloi::cuda::deepseek_v4::State>(work.dsv4_model);
    auto& dsv4 = *work.dsv4;
    auto& block = dsv4.layers[0];
    block.local_cache = values({1, 128, 8}, tyloi::kFloat16, 1.0f);
    const auto initialize_pool = [&](tyloi::cuda::deepseek_v4::PoolState& pool,
            int64_t head_dim, bool overlap, float offset) {
        pool.ratio = 4;
        pool.head_dim = head_dim;
        pool.overlap = overlap;
        pool.cache_quant_mode = overlap ? 2 : 0;
        pool.capacity = 16;
        const auto width = overlap ? 2 * head_dim : head_dim;
        pool.state_kv = values({1, 4, width}, tyloi::kFloat32, offset);
        pool.state_gate = values({1, 4, width}, tyloi::kFloat32, offset + 100.0f);
        if (overlap) {
            pool.previous_kv = values({1, 4, head_dim}, tyloi::kFloat32, offset + 200.0f);
            pool.previous_gate = values({1, 4, head_dim}, tyloi::kFloat32, offset + 300.0f);
        }
        pool.pool = values({1, 16, head_dim}, tyloi::kFloat16, offset + 400.0f);
    };
    initialize_pool(block.compressor, 8, false, 10.0f);
    initialize_pool(block.indexer_compressor, 4, true, 20.0f);
    dsv4.advance(10);
    std::vector<int64_t> tokens(10);
    std::iota(tokens.begin(), tokens.end(), 0);
    work.dsv4_saved = dsv4.capture_text_session_state(tokens);
    for (auto& tensor : dsv4.storage()) tensor.dense().zero_();
    dsv4.cache_pos = 0;
    dsv4.restore_text_session_state(work.dsv4_saved);
    const auto& saved_dsv4 = std::get<std::vector<tyloi::models::deepseek_v4::BlockSnapshot>>(
        work.dsv4_saved.payload).at(0);
    require_equal(same(block.local_cache, saved_dsv4.local_cache),
        "DeepSeek V4 local session cache restore failed");
    const auto check_pool = [&](const tyloi::cuda::deepseek_v4::PoolState& restored,
            const tyloi::models::deepseek_v4::PoolSnapshot& saved) {
        require_equal(same(restored.state_kv, saved.state_kv) && same(restored.state_gate, saved.state_gate),
            "DeepSeek V4 compressor recurrent state restore failed");
        if (saved.overlap)
            require_equal(same(restored.previous_kv, saved.previous_kv) && same(restored.previous_gate, saved.previous_gate),
                "DeepSeek V4 overlap compressor state restore failed");
        require_equal(restored.capacity == saved.capacity &&
            same(restored.pool.dense().narrow(1, 0, saved.pool.size(1)), saved.pool),
            "DeepSeek V4 compressed pool restore failed");
    };
    check_pool(block.compressor, saved_dsv4.compressor);
    check_pool(block.indexer_compressor, saved_dsv4.indexer_compressor);
    require_equal(dsv4.position() == 10 && work.dsv4_saved.bytes > 0,
        "DeepSeek V4 session position restore failed");

    work.glm_model.layers.resize(2);
    for (std::size_t index = 0; index < 2; ++index) {
        auto& layout = work.glm_model.layers[index];
        layout.device = device.index;
        layout.full_indexer = index == 0;
        layout.config.max_position_embeddings = 16;
        layout.config.index_head_dim = 4;
    }
    work.glm = std::make_unique<tyloi::cuda::glm_dsa::State>(work.glm_model);
    auto& glm = *work.glm;
    for (std::size_t index = 0; index < 2; ++index) {
        auto& layer = glm.layers[index];
        layer.kv_cache = values({1, 1, 16, 6}, tyloi::kFloat16, 1000.0f + index * 100.0f);
        if (index == 0) layer.index_cache = values({1, 16, 4}, tyloi::kFloat16, 2000.0f);
    }
    glm.advance(10);
    work.glm_saved = glm.capture_text_session_state(tokens);
    for (auto& layer : glm.layers) {
        layer.kv_cache.zero_();
        if (layer.index_cache.defined()) layer.index_cache.zero_();
    }
    glm.selections.at(device.index).topk_indices = tyloi::cuda::ones({1, 1, 1}, cuda_float.dtype(tyloi::kInt32));
    glm.cache_pos = 0;
    glm.restore_text_session_state(work.glm_saved);
    const auto& saved_glm = std::get<std::vector<tyloi::models::glm_dsa::BlockSnapshot>>(work.glm_saved.payload);
    for (std::size_t index = 0; index < 2; ++index) {
        const auto& saved = saved_glm[index];
        const auto& restored = glm.layers[index];
        require_equal(restored.kv_cache.size(2) == saved.kv_capacity &&
            same(restored.kv_cache.narrow(2, 0, saved.kv_cache.size(2)), saved.kv_cache),
            "GLM DSA MLA session cache restore failed");
        if (saved.full_indexer)
            require_equal(restored.index_cache.size(1) == saved.index_capacity &&
                same(restored.index_cache.narrow(1, 0, saved.index_cache.size(1)), saved.index_cache),
                "GLM DSA index session cache restore failed");
    }
    require_equal(glm.position() == 10 && work.glm_saved.bytes > 0 &&
        !glm.selections.at(device.index).topk_indices.defined(), "GLM DSA session metadata restore failed");
    program.drain();
    std::cout << "text_session_state_check dsv4=1 glm_dsa=1\n";
    return 0;
}
