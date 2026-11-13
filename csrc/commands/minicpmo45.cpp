#include "minicpmo45.h"
#include "cli.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/causal_model.h"
#include "engine/models/media.h"
#include "engine/models/speech.h"
#include "engine/ops.h"
#include "eval/tensors.h"
#include "eval/minicpmo45.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

using tyloi::indexing::Slice;

std::string minicpmo45_duplex_step_prefix(
        const std::string & prefix,
        int64_t step) {
    std::ostringstream stream;
    stream << prefix << ".step" << std::setfill('0')
           << std::setw(4) << step;
    return stream.str();
}

} // namespace

namespace tyloi::cuda::minicpmo45 {

bool parse_command_option(
        std::string_view option,
        commands::ArgCursor& args,
        CommandOptions& result) {
    using commands::integer;
    using commands::usage_error;
    if (option == "--minicpmo-input-prefix") {
        result.input_prefix = args.value(option);
    } else if (option == "--minicpmo-output-prefix") {
        result.output_prefix = args.value(option);
    } else if (option == "--minicpmo-tts-steps") {
        result.tts_steps = integer<int64_t>(args.value(option), option);
        if (result.tts_steps < 0) {
            usage_error("--minicpmo-tts-steps must be non-negative");
        }
    } else if (option == "--minicpmo-duplex-input-prefix") {
        result.duplex_input_prefix = args.value(option);
    } else if (option == "--minicpmo-duplex-output-prefix") {
        result.duplex_output_prefix = args.value(option);
    } else if (option == "--minicpmo-duplex-steps") {
        result.duplex_steps = integer<int64_t>(args.value(option), option);
        if (result.duplex_steps < 0) {
            usage_error("--minicpmo-duplex-steps must be non-negative");
        }
    } else if (option == "--minicpmo-duplex-max-speak-tokens") {
        result.duplex_max_speak_tokens =
            integer<int64_t>(args.value(option), option);
        if (result.duplex_max_speak_tokens <= 0) {
            usage_error(
                "--minicpmo-duplex-max-speak-tokens must be positive");
        }
    } else if (option == "--minicpmo-duplex-seed") {
        result.duplex_seed = integer<int64_t>(args.value(option), option);
    } else if (option == "--minicpmo-duplex-greedy") {
        result.duplex_greedy = true;
    } else {
        return false;
    }
    return true;
}

void validate_command_options(const CommandOptions& options) {
    if (!options.output_prefix.empty() && options.input_prefix.empty()) {
        commands::usage_error(
            "--minicpmo-output-prefix requires --minicpmo-input-prefix");
    }
    if (!options.duplex_output_prefix.empty() &&
            options.duplex_input_prefix.empty()) {
        commands::usage_error(
            "--minicpmo-duplex-output-prefix requires "
            "--minicpmo-duplex-input-prefix");
    }
}

using tyloi::eval::read_tensor;
using tyloi::eval::write_tensor;
using tyloi::eval::read_scalar;

int run_composite(Program& program, const CommandOptions& options) {
    if (options.input_prefix.empty() || options.output_prefix.empty())
        throw std::runtime_error("MiniCPM-o composite graph requires input and output prefixes");
    tyloi::cuda::NoGradGuard no_grad;
    auto scope = program.executor().enter(program.config().layer_placement.primary_device());
    program.load(true);
    auto& model = program.model();
    auto& state = program.state();
    const auto* media = model.media();
    if (!media) throw std::runtime_error("MiniCPM-o model has no media capability");
    const int primary = program.config().layer_placement.primary_device();
    const auto mode = program.executor().forward_mode(primary);
    const tyloi::models::ForwardOptions forward_options{mode.kv_capacity, mode.trace, mode.trace_layer, primary};
    struct Work {
        Program& program;
        tyloi::models::MediaInput media_input;
        tyloi::models::MediaResult media_result;
        tyloi::models::Input input;
        tyloi::models::SpeechResult speech_result;
        tyloi::Tensor raw_ids, raw_condition, positions, mask, hidden, logits, condition, text_ids, text_hidden;
        std::unique_ptr<tyloi::models::Media::Request> media_request;
        std::unique_ptr<tyloi::models::Speech::Request> speech_request;
        ~Work() noexcept {
            try { program.drain(); }
            catch (const std::exception& error) { std::cerr << "minicpmo_cleanup error=" << error.what() << '\n'; }
            catch (...) { std::cerr << "minicpmo_cleanup error=unknown\n"; }
        }
    } work{program};
    const auto& in = options.input_prefix;
    const auto& out = options.output_prefix;
    auto ids = read_tensor(in + ".input_ids.pt", true);
    work.raw_ids = tyloi::Tensor(ids);
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    ids = ids.to(tyloi::Device(tyloi::kCUDA, primary), tyloi::kInt64).contiguous();
    work.media_input.ids = tyloi::Tensor(ids);
    work.positions = tyloi::Tensor(read_tensor(in + ".position_ids.pt", false));
    work.mask = tyloi::Tensor(read_tensor(in + ".attention_mask.pt", false));
    work.media_input.pixels = tyloi::Tensor(read_tensor(in + ".pixel_values.pt", false));
    work.media_input.patch_mask = tyloi::Tensor(read_tensor(in + ".patch_mask.pt", false));
    work.media_input.target_sizes = tyloi::Tensor(read_tensor(in + ".target_sizes.pt", false));
    work.media_input.image_bounds = tyloi::Tensor(read_tensor(in + ".image_bounds.pt", false));
    work.media_input.audio_features = tyloi::Tensor(read_tensor(in + ".audio_features.pt", false));
    work.media_input.audio_lengths = tyloi::Tensor(read_tensor(in + ".audio_lengths.pt", false));
    work.media_input.audio_bounds = tyloi::Tensor(read_tensor(in + ".audio_bounds.pt", false));
    work.media_request = media->encode(work.media_input);
    bool encoded = false;
    while (auto step = work.media_request->next()) {
        if (step.workspace || step.graph || step.layer)
            throw std::logic_error("MiniCPM-o media request requires unsupported resource handling");
        if (step.value) { work.media_result = std::move(*step.value); encoded = true; }
    }
    if (!encoded) throw std::runtime_error("MiniCPM-o media request produced no result");
    state.reset(ids.size(0));
    work.input.ids = work.media_input.ids;
    work.input.embeddings = work.media_result.input_embeddings;
    work.input.positions = work.positions;
    work.input.attention_mask = work.mask;
    work.input.advance = work.positions.defined();
    work.hidden = model.forward(work.input, state, forward_options);
    work.logits = model.logits(work.hidden);
    const auto save = [&](const tyloi::Tensor& value, const char* name) {
        if (value.defined()) write_tensor(value.dense(), out + name);
    };
    save(work.media_result.vision_states, ".vision_states.pt");
    save(work.media_result.image_embeddings, ".image_embeddings.pt");
    save(work.media_result.audio_embeddings, ".audio_embeddings.pt");
    save(work.media_result.input_embeddings, ".input_embeddings.pt");
    save(work.hidden, ".hidden_states.pt");
    save(work.logits, ".logits.pt");
    if (options.tts_steps > 0) {
        const auto* speech = model.speech();
        if (!speech) throw std::runtime_error("MiniCPM-o model has no speech capability");
        auto condition = read_tensor(in + ".tts_inputs_embeds.pt", false);
        work.raw_condition = tyloi::Tensor(condition);
        if (!condition.defined()) {
            auto bound = read_tensor(in + ".tts_bound.pt", true).to(tyloi::kCPU, tyloi::kInt64).contiguous();
            if (bound.dim() != 1 || bound.numel() != 2)
                throw std::runtime_error("MiniCPM-o tts_bound must contain [begin,end]");
            const auto begin = bound[0].item<int64_t>(), end = bound[1].item<int64_t>();
            if (begin < 0 || end <= begin || end > ids.size(-1) || ids.size(0) != 1)
                throw std::runtime_error("MiniCPM-o TTS text span is invalid");
            work.text_ids = tyloi::Tensor(ids.index({0, Slice(begin, end)}));
            work.text_hidden = tyloi::Tensor(work.hidden.dense().index({0, Slice(begin, end), Slice()}));
            work.condition = speech->condition(work.text_ids, work.text_hidden);
        } else {
            work.condition = tyloi::Tensor(condition.to(tyloi::Device(tyloi::kCUDA, primary)).contiguous());
        }
        tyloi::models::SpeechOptions speech_options;
        speech_options.steps = options.tts_steps;
        work.speech_request = speech->generate(work.condition, speech_options, true);
        bool generated = false;
        while (auto step = work.speech_request->next()) {
            if (step.workspace || step.graph || step.layer)
                throw std::logic_error("MiniCPM-o speech request requires unsupported resource handling");
            if (step.value) { work.speech_result = std::move(*step.value); generated = true; }
        }
        if (!generated) throw std::runtime_error("MiniCPM-o speech request produced no result");
        save(work.condition, ".tts_inputs_embeds.pt");
        save(work.speech_result.codes, ".tts_codes.pt");
        save(work.speech_result.logits, ".tts_logits.pt");
    }
    program.drain();
    std::cout << "minicpmo45_composite=ok" << " batch=" << ids.size(0) << " tokens=" << ids.size(1)
              << " images=" << (work.media_input.image_bounds ? work.media_input.image_bounds.size(0) : 0)
              << " audios=" << (work.media_input.audio_bounds ? work.media_input.audio_bounds.size(0) : 0)
              << " tts_steps=" << options.tts_steps << "\n";
    return 0;
}

int run_eval_batch(Program& program, std::int64_t context_size, std::int64_t vision_batch_size) {
    return tyloi::eval::minicpmo45_batch(program, context_size, vision_batch_size);
}

int run_duplex(Program& program, const CommandOptions& options) {
    const auto& in = options.duplex_input_prefix;
    const auto& out = options.duplex_output_prefix;
    if (in.empty() || out.empty() || options.duplex_steps <= 0 ||
        options.duplex_max_speak_tokens < 2 || options.duplex_seed < 0)
        throw std::runtime_error("MiniCPM-o duplex mode requires prefixes, positive steps, "
                                 "at least two generation slots, and a non-negative seed");
    tyloi::cuda::NoGradGuard no_grad;
    auto scope = program.executor().enter(program.config().layer_placement.primary_device());
    program.load(true);
    auto& model = program.model();
    auto& state = program.state();
    const int primary = program.config().layer_placement.primary_device();
    const auto mode = program.executor().forward_mode(primary);
    const tyloi::models::ForwardOptions forward_options{mode.kv_capacity, mode.trace, mode.trace_layer, primary};
    struct Work {
        Program& program;
        MfqDuplexSessionParams parameters;
        tyloi::models::Duplex::StepInput input;
        tyloi::models::Duplex::StepOutput output;
        tyloi::Tensor state_tensor;
        std::unique_ptr<tyloi::models::Duplex> session;
        ~Work() noexcept {
            try { program.drain(); }
            catch (const std::exception& error) { std::cerr << "minicpmo_cleanup error=" << error.what() << '\n'; }
            catch (...) { std::cerr << "minicpmo_cleanup error=unknown\n"; }
            if (session) {
                try { session->stop(); }
                catch (const std::exception& error) { std::cerr << "minicpmo_cleanup error=" << error.what() << '\n'; }
                catch (...) { std::cerr << "minicpmo_cleanup error=unknown\n"; }
            }
        }
    } work{program};
    const auto values = [](tyloi::DenseTensor value) {
        if (!value.defined()) return std::vector<int64_t>{};
        value = value.to(tyloi::kCPU, tyloi::kInt64).contiguous().reshape({-1});
        if (!value.numel()) return std::vector<int64_t>{};
        const auto* data = value.data_ptr<int64_t>();
        return std::vector<int64_t>(data, data + value.numel());
    };
    auto special = read_tensor(in + ".special_ids.pt", true);
    work.parameters.special_ids = values(special);
    if (work.parameters.special_ids.size() != 15)
        throw std::runtime_error("MiniCPM-o duplex special_ids must contain 15 token IDs");
    if (std::any_of(work.parameters.special_ids.begin(), work.parameters.special_ids.end(),
                    [](int64_t id) { return id < 0; }))
        throw std::runtime_error("MiniCPM-o duplex special token IDs must be non-negative");
    work.parameters.forbidden_ids = values(read_tensor(in + ".forbidden_ids.pt", false));
    auto system = read_tensor(in + ".system_ids.pt", false);
    if (system.defined() && system.numel() &&
        (system.dim() > 2 || (system.dim() == 2 && system.size(0) != 1)))
        throw std::runtime_error("MiniCPM-o duplex system IDs must contain one sequence");
    work.parameters.system_prefix = values(system);
    work.parameters.greedy = options.duplex_greedy;
    work.parameters.seed = static_cast<std::uint64_t>(options.duplex_seed);
    work.session = model.duplex(state);
    if (!work.session) throw std::runtime_error("MiniCPM-o model has no duplex capability");
    work.session->start(work.parameters, forward_options);
    program.drain();
    int64_t listen_steps = 0, speak_steps = 0, chunk_index = 0;
    MfqDuplexStepResult last;
    for (int64_t step = 0; step < options.duplex_steps; ++step) {
        const auto input = minicpmo45_duplex_step_prefix(in, step);
        const auto output = minicpmo45_duplex_step_prefix(out, step);
        if (read_scalar(input + ".reset_session.pt", 0) != 0) {
            work.session->reset(forward_options);
            program.drain();
            chunk_index = 0;
        }
        work.input = {};
        work.output = {};
        work.input.pixels = tyloi::Tensor(read_tensor(input + ".pixel_values.pt", false));
        work.input.patch_mask = tyloi::Tensor(read_tensor(input + ".patch_mask.pt", false));
        work.input.target_sizes = tyloi::Tensor(read_tensor(input + ".target_sizes.pt", false));
        work.input.image_slice_counts = tyloi::Tensor(read_tensor(input + ".image_slice_counts.pt", false));
        work.input.audio_features = tyloi::Tensor(read_tensor(input + ".audio_features.pt", false));
        work.input.text_ids = tyloi::Tensor(read_tensor(input + ".text_ids.pt", false));
        work.input.audio_prefix_extra_frames = read_scalar(
            input + ".audio_prefix_extra_frames.pt", chunk_index == 0 ? 0 : 2);
        work.input.audio_suffix_extra_frames = read_scalar(
            input + ".audio_suffix_extra_frames.pt", 2);
        work.input.force_listen = read_scalar(input + ".force_listen.pt", 0) != 0;
        work.input.max_new_speak_tokens = options.duplex_max_speak_tokens;
        work.output = work.session->step(work.input, forward_options);
        const auto& result = work.output.result;
        const auto save = [&](const tyloi::Tensor& value, const char* suffix) {
            if (value.defined()) write_tensor(value.dense(), output + suffix);
        };
        save(work.output.decision_logits, ".decision_logits.pt");
        save(work.output.audio_embeddings, ".audio_embeddings.pt");
        save(work.output.generated_ids, ".generated_ids.pt");
        save(work.output.tts_codes, ".tts_codes.pt");
        work.state_tensor = tyloi::Tensor(tyloi::cuda::tensor(std::vector<int64_t>{
            result.is_listen ? 1 : 0, result.end_of_turn ? 1 : 0,
            result.language_cache_position, result.audio_cache_position,
            result.tts_cache_position, result.audio_chunk_index},
            tyloi::TensorOptions().dtype(tyloi::kInt64)));
        save(work.state_tensor, ".state.pt");
        if (result.is_listen) ++listen_steps; else ++speak_steps;
        last = result;
        chunk_index = result.audio_chunk_index;
        program.drain();
    }
    program.drain();
    work.session->stop();
    work.session.reset();
    std::cout << "minicpmo45_duplex=ok" << " steps=" << options.duplex_steps
              << " listens=" << listen_steps << " speaks=" << speak_steps
              << " language_cache=" << last.language_cache_position
              << " audio_cache=" << last.audio_cache_position
              << " tts_cache=" << last.tts_cache_position << " seed=" << options.duplex_seed << "\n";
    return 0;
}

} // namespace tyloi::cuda::minicpmo45
