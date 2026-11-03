#include "table.h"

namespace mfq::engine {

void prepare_request(Request &current, const TextProcessor *text, const EngineInfo &info, bool batched, bool mtp_available) {
    auto request = current.submitted;
    const bool raw = !request.token_ids.empty();
    InferenceRequest input;
    const MfqTokenizer *tokenizer = nullptr;
    if (raw) {
        input.prompt = std::move(request.token_ids);
        input.sampling = request.input.sampling;
        input.cache_plan = request.input.cache_plan;
        input.stop_token_ids = info.stop_token_ids;
    } else {
        if (!text) throw std::invalid_argument("text input requires a tokenizer");
        tokenizer = &text->tokenizer();
        input = text->prepare(std::move(request.input), info.max_context);
    }
    const auto plan = plan_generation(input.prompt, info.vocab_size, info.max_context,
        input.sampling.max_tokens, input.cache_plan.stable_prefix_tokens);
    input.sampling.max_tokens = plan.generation_tokens;
    input.cache_plan.stable_prefix_tokens = plan.stable_prefix_tokens;
    try {
        current.input = std::move(input);
        current.output.emplace(current.input, tokenizer, current.submitted.id);
    }
    catch (...) { current.input = {}; throw; }
    current.output->metrics.mtp.available = mtp_available;
    current.batched = batched;
}

void finish_request(const RequestId &id, Request &request, bool &healthy, EngineStepResult &result) {
    try {
        if (request.error)
            std::rethrow_exception(request.error);
        auto delta = request.output->finish();
        if (!delta.diffs.empty())
            result.events.push_back({id, std::move(delta)});
        const UsageUpdate usage{request.input.prompt.size(),
            static_cast<std::size_t>(request.output->result.completion_tokens)};
        result.events.push_back({id, usage});
        if (request.output->result.cancelled)
            result.events.push_back({id, Cancelled{usage, request.output->metrics}});
        else
            result.events.push_back({id,
                Completed{request.output->result.finish_reason, usage, request.output->metrics}});
    } catch (const MfqResourceExhausted &error) {
        result.events.push_back({id, Failed{healthy ? "resource_exhausted" : "backend_failure", error.what(), healthy}});
    } catch (const InferenceInputError &error) {
        const auto code = !healthy                                            ? "backend_failure"
                          : error.code == InferenceInputErrorCode::Unsupported ? "unsupported_input"
                                                                               : "invalid_request";
        result.events.push_back({id, Failed{code, error.what()}});
    } catch (const std::invalid_argument &error) {
        result.events.push_back(
            {id, Failed{healthy ? "invalid_request" : "backend_failure", error.what()}});
    } catch (const std::exception &error) {
        healthy = false;
        result.events.push_back({id, Failed{"backend_failure", error.what()}});
    } catch (...) {
        healthy = false;
        result.events.push_back({id, Failed{"backend_failure", "unknown execution failure"}});
    }
}

} // namespace mfq::engine
