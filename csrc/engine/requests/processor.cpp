#include "processor.h"
#include "engine/models/source.h"
#include "engine/models/prepare.h"

#include "constraint.h"
#include "tokenizer.h"

#include "chat/common.h"
#include "json-schema-to-grammar.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

namespace mfq::engine {
namespace {

using json = nlohmann::ordered_json;

} // namespace

InferenceInputError::InferenceInputError(
        InferenceInputErrorCode input_code,
        std::string message,
        std::string input_field)
    : std::runtime_error(std::move(message)),
      code(input_code),
      field(std::move(input_field)) {}

struct TextProcessor::Impl {
    Impl(
            std::unique_ptr<MfqTokenizer> input_tokenizer,
            std::int32_t expected_vocabulary,
            std::string input_model_type)
        : tokenizer(std::move(input_tokenizer)),
          model_type(std::move(input_model_type)) {
        if (expected_vocabulary > 0 &&
                tokenizer->vocab_size() != expected_vocabulary) {
            throw std::runtime_error(
                "tokenizer/model vocabulary mismatch: tokenizer=" +
                std::to_string(tokenizer->vocab_size()) + " model=" +
                std::to_string(expected_vocabulary));
        }
        const auto source = tokenizer->chat_template();
        if (!source.empty()) {
            templates = common_chat_templates_init(tokenizer->context(), "");
            if (!templates) {
                throw std::runtime_error(
                    "cannot initialize tokenizer.chat_template");
            }
        }
        capabilities.thinking =
            source.find("enable_thinking") != std::string::npos;
        if (source.find("reasoning_effort") != std::string::npos) {
            for (const char* value : {"high", "max"}) {
                if (source.find(std::string("'") + value + "'") !=
                        std::string::npos ||
                    source.find(std::string("\"") + value + "\"") !=
                        std::string::npos) {
                    capabilities.reasoning_effort_values.emplace_back(value);
                }
            }
        }
    }

    std::unique_ptr<MfqTokenizer> tokenizer;
    common_chat_templates_ptr templates;
    std::string model_type;
    ChatTemplateCapabilities capabilities;
    std::map<std::string, std::pair<std::string, bool>> prefix_policies;
    std::mutex prefix_policy_mutex;

    std::pair<std::string, bool> prefix_policy(const common_chat_templates_inputs& inputs) {
        std::lock_guard<std::mutex> lock(prefix_policy_mutex);
        json key{{"kwargs", inputs.chat_template_kwargs}, {"thinking", inputs.enable_thinking},
            {"tool_choice", inputs.tool_choice}, {"parallel", inputs.parallel_tool_calls}};
        for (const auto& tool : inputs.tools)
            key["tools"].push_back({tool.name, tool.description, tool.parameters});
        const auto signature = key.dump();
        if (const auto found = prefix_policies.find(signature); found != prefix_policies.end())
            return found->second;
        std::pair<std::string, bool> policy;
        try {
            auto probe = inputs;
            probe.messages = {common_chat_msg{"user", "probe"}};
            probe.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
            probe.add_generation_prompt = true;
            const auto with = common_chat_templates_apply(templates.get(), probe).prompt;
            probe.add_generation_prompt = false;
            const auto without = common_chat_templates_apply(templates.get(), probe).prompt;
            if (with.starts_with(without)) {
                policy.first = with.substr(without.size());
                probe.messages.push_back(common_chat_msg{"assistant", "reply"});
                probe.messages.push_back(common_chat_msg{"user", "next"});
                policy.second = common_chat_templates_apply(templates.get(), probe).prompt.starts_with(with);
            }
        } catch (const std::exception&) {
        }
        if (prefix_policies.size() >= 16) prefix_policies.clear();
        prefix_policies.emplace(signature, policy);
        return policy;
    }
};

std::unique_ptr<TextProcessor> TextProcessor::load(const ModelSource& source,
        const std::string& tokenizer_path, std::int32_t vocabulary, const std::string& model_type) {
    if (source.has_asset(kTokenizerGgufAsset)) {
        const auto bytes = source.read_asset(kTokenizerGgufAsset);
        return std::make_unique<TextProcessor>(std::vector<std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(bytes.data()),
            reinterpret_cast<const std::uint8_t*>(bytes.data()) + bytes.size()), vocabulary, model_type);
    }
    return std::make_unique<TextProcessor>(tokenizer_path, vocabulary, model_type);
}

TextProcessor::TextProcessor(
        const std::string& tokenizer_path,
        std::int32_t expected_vocabulary,
        std::string model_type)
    : impl_(std::make_unique<Impl>(
          std::make_unique<MfqTokenizer>(tokenizer_path),
          expected_vocabulary, std::move(model_type))) {}

TextProcessor::TextProcessor(
        const std::vector<std::uint8_t>& tokenizer_gguf,
        std::int32_t expected_vocabulary,
        std::string model_type)
    : impl_(std::make_unique<Impl>(
          std::make_unique<MfqTokenizer>(tokenizer_gguf),
          expected_vocabulary, std::move(model_type))) {}

TextProcessor::~TextProcessor() = default;

std::int32_t TextProcessor::vocab_size() const {
    return impl_->tokenizer->vocab_size();
}

ChatTemplateCapabilities TextProcessor::chat_template_capabilities() const {
    return impl_->capabilities;
}

InferenceRequest TextProcessor::prepare(
        InferenceInput input,
        std::int64_t max_context) const {
    InferenceRequest work;
    work.chat = input.chat;
    work.stream = input.stream;
    work.include_usage = input.include_usage;
    work.sampling = input.sampling;
    work.cache_plan = std::move(input.cache_plan);
    work.stops = std::move(input.stops);

    std::string prompt = std::move(input.prompt);
    bool parse_special = false;
    if (input.chat) {
        if (!input.chat_input) {
            throw InferenceInputError(
                InferenceInputErrorCode::Invalid,
                "chat input is missing", "messages");
        }
        auto& chat = *input.chat_input;
        work.chat_parser.reasoning_format = chat.reasoning_format;
        work.chat_parser.reasoning_in_content =
            input.stream && chat.reasoning_format ==
                COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
        if (chat.preformatted_prompt) {
            prompt = *chat.preformatted_prompt;
            work.cache_plan.cache_output_tokens = true;
            work.chat_parser.parse_tool_calls = false;
            if (!chat.template_inputs.json_schema.empty()) {
                common_chat_params constraint;
                constraint.grammar = json_schema_to_grammar(
                    json::parse(chat.template_inputs.json_schema));
                work.token_constraint = make_chat_token_constraint(
                    *impl_->tokenizer, constraint);
            }
        } else {
            if (!impl_->templates) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Unsupported,
                    "the tokenizer has no chat template; provide input.preformatted_prompt",
                    "messages");
            }
            auto inputs = std::move(chat.template_inputs);
            if (chat.parallel_tool_calls) {
                inputs.parallel_tool_calls = *chat.parallel_tool_calls;
            } else {
                inputs.parallel_tool_calls = common_chat_templates_get_caps(
                    impl_->templates.get()).at("supports_parallel_tool_calls");
            }
            common_chat_params params;
            try {
                params = common_chat_templates_apply(
                    impl_->templates.get(), inputs);
            } catch (const std::exception& error) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Invalid,
                    std::string("chat template application failed: ") +
                        error.what(),
                    "messages");
            }
            prompt = params.prompt;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_NONE && inputs.add_generation_prompt) {
                const auto policy = impl_->prefix_policy(inputs);
                const bool open_think = !params.thinking_start_tag.empty() &&
                    policy.first.find(params.thinking_start_tag) != std::string::npos &&
                    (params.thinking_end_tag.empty() || policy.first.find(params.thinking_end_tag) == std::string::npos);
                work.cache_plan.cache_output_tokens = policy.second && !open_think;
                if (!work.cache_plan.cache_output_tokens && !policy.first.empty() && prompt.ends_with(policy.first))
                    work.cache_plan.stable_prefix_tokens = impl_->tokenizer->tokenize(
                        prompt.substr(0, prompt.size() - policy.first.size()), true).size();
            }
            work.token_constraint = make_chat_token_constraint(
                *impl_->tokenizer, params);
            work.chat_parser.format = params.format;
            work.chat_parser.generation_prompt = params.generation_prompt;
            work.chat_parser.parse_tool_calls = true;
            work.chat_parser.is_continuation =
                inputs.continue_final_message !=
                COMMON_CHAT_CONTINUATION_NONE;
            if (!params.parser.empty()) {
                work.chat_parser.parser.load(params.parser);
            }
            for (const auto& text : params.preserved_tokens) {
                const auto tokens = impl_->tokenizer->tokenize(text, true);
                work.preserved_tokens.insert(tokens.begin(), tokens.end());
            }
            work.stops.insert(
                work.stops.end(),
                params.additional_stops.begin(), params.additional_stops.end());
        }
        parse_special = true;
    }

    work.prompt = impl_->tokenizer->tokenize(prompt, parse_special);
    if (work.prompt.empty()) {
        throw InferenceInputError(
            InferenceInputErrorCode::Invalid,
            "prompt tokenized to an empty sequence", "prompt");
    }
    if (input.benchmark_prompt_tokens) {
        const auto count = *input.benchmark_prompt_tokens;
        if (count > static_cast<std::size_t>(max_context) ||
            count + work.sampling.max_tokens > static_cast<std::size_t>(max_context)) {
            throw InferenceInputError(InferenceInputErrorCode::Invalid,
                "benchmark prompt and output exceed the loaded context capacity", "prompt");
        }
        const auto source = work.prompt;
        work.prompt.resize(count);
        for (std::size_t i = source.size(); i < count; ++i)
            work.prompt[i] = source[i % source.size()];
    }
    if (work.cache_plan.stable_prefix_tokens == 0)
        work.cache_plan.stable_prefix_tokens = work.prompt.size();
    if (!input.chat) work.cache_plan.cache_output_tokens = true;
    models::prepare_prefix(impl_->model_type, *impl_->tokenizer, input, work);

    if (!input.prefix_cache_enabled) {
        work.cache_plan.stable_prefix_tokens = 0;
        work.cache_plan.cache_output_tokens = false;
    }
    if (input.media) {
        const auto suffix_tokens = work.prompt.size() - std::min(
            work.cache_plan.stable_prefix_tokens, work.prompt.size());
        models::prepare_media(
            *impl_->tokenizer, impl_->tokenizer->vocab_size(),
            work.prompt, *input.media);
        if (work.cache_plan.stable_prefix_tokens > 0)
            work.cache_plan.stable_prefix_tokens = work.prompt.size() - suffix_tokens;
        work.vision = std::move(input.media);
    }
    if (max_context > 0 &&
            static_cast<std::int64_t>(work.prompt.size()) +
                    work.sampling.max_tokens > max_context) {
        throw InferenceInputError(
            InferenceInputErrorCode::ContextLength,
            input.media
                ? "expanded multimodal prompt plus max_tokens exceed the model context window"
                : "prompt tokens plus max_tokens exceed the model context window",
            "max_tokens");
    }

    std::vector<std::string> unique_stops;
    unique_stops.reserve(work.stops.size());
    for (auto& stop : work.stops) {
        if (std::find(unique_stops.begin(), unique_stops.end(), stop) ==
                unique_stops.end()) {
            unique_stops.push_back(std::move(stop));
        }
    }
    work.stops = std::move(unique_stops);
    return work;
}

std::size_t TextProcessor::count_tokens(const std::string& prompt) const {
    if (prompt.empty() || prompt.size() > 262144)
        throw std::invalid_argument("token count requires a nonempty prompt of at most 262144 bytes");
    return impl_->tokenizer->tokenize(prompt, false, false).size();
}

const MfqTokenizer& TextProcessor::tokenizer() const {
    return *impl_->tokenizer;
}

void TextProcessor::prepare_duplex_session(
        const std::string& system_prompt,
        MfqDuplexSessionParams& parameters) const {
    models::prepare_duplex_session(*impl_->tokenizer, system_prompt, parameters);
}

void TextProcessor::prepare_duplex_step(
        const std::string& text,
        MfqDuplexStepInput& step) const {
    models::prepare_duplex_step(*impl_->tokenizer, text, step);
}

std::string TextProcessor::decode_tokens(
        const std::vector<std::int64_t>& tokens,
        const std::unordered_set<std::int64_t>& excluded) const {
    std::string result;
    for (const auto token : tokens) {
        if (excluded.count(token) == 0) {
            result += impl_->tokenizer->piece(token, false);
        }
    }
    return result;
}

} // namespace mfq::engine
