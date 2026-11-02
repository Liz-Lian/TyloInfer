#pragma once

#include "engine/models/common/session/state.h"
#include "engine/models/deepseek_v4/snapshot.h"
#include "engine/models/deepseek_v41/snapshot.h"
#include "engine/models/glm_dsa/snapshot.h"
#include "engine/models/qwen4_exp/snapshot.h"

#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

enum class TextSessionStateKind : int {
    Unsupported,
    FullAttention,
    HybridAttention,
    DeepseekV4,
    GlmDsa,
    Qwen4,
    DeepseekV41,
};

struct MtpSessionState {
    std::variant<
        std::vector<KvSnapshot>,
        mfq::models::qwen4_exp::MtpSnapshot,
        mfq::models::deepseek_v41::MtpSnapshot> payload = std::vector<KvSnapshot>{};
    mfq_tensor_backend::Tensor last_target_hidden;
    int64_t cache_pos = 0;
    size_t bytes = 0;
};

using TextSessionPayload = std::variant<
    std::monostate,
    std::vector<KvSnapshot>,
    std::vector<HybridBlockSessionState>,
    std::vector<mfq::models::deepseek_v4::BlockSnapshot>,
    std::vector<mfq::models::glm_dsa::BlockSnapshot>,
    mfq::models::qwen4_exp::Snapshot,
    mfq::models::deepseek_v41::Snapshot>;

struct TextSessionState {
    std::vector<int64_t> tokens;
    std::string input_key;
    TextSessionPayload payload;
    std::optional<MtpSessionState> mtp;
    int64_t cache_pos = 0;
    int64_t decode_position_delta = 0;
    size_t bytes = 0;

    template <class Visit>
    void tensors(const Visit& tensor) const {
        const auto payload_tensors = [&](const auto& value) {
            if constexpr (requires { value.tensors(tensor); }) {
                value.tensors(tensor);
            } else if constexpr (!std::is_same_v<std::decay_t<decltype(value)>, std::monostate>) {
                for (const auto& block : value) block.tensors(tensor);
            }
        };
        std::visit(payload_tensors, payload);
        if (mtp) {
            tensor(mtp->last_target_hidden);
            std::visit(payload_tensors, mtp->payload);
        }
    }

    TextSessionStateKind kind() const noexcept {
        return static_cast<TextSessionStateKind>(payload.index());
    }
};
