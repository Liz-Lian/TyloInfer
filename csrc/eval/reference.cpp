#include "eval/reference.h"
#include "engine/program.h"
#include "engine/models/layers.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/ops.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyloi::eval {
namespace {

std::size_t elements(std::int64_t rows, std::int64_t columns) {
    if (rows < 1 || columns < 1 || std::uint64_t(rows) >
        std::uint64_t(std::numeric_limits<std::streamsize>::max()) / sizeof(float) / std::uint64_t(columns))
        throw std::overflow_error("reference matrix geometry overflows");
    return std::size_t(rows) * std::size_t(columns);
}

class Hidden {
public:
    Hidden(const std::filesystem::path& output, std::int64_t context, std::int64_t width)
        : path_(output.string() + ".hidden"), context_(context), width_(width),
          buffer_(elements(context, width)), bytes_(buffer_.size() * sizeof(float)) {
        if (std::filesystem::exists(path_))
            throw std::runtime_error("reference hidden-state scratch already exists");
        stream_.open(path_, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
        if (!stream_) throw std::runtime_error("cannot create reference hidden-state scratch");
    }
    ~Hidden() {
        stream_.close();
        std::error_code error;
        std::filesystem::remove(path_, error);
    }
    void store(std::uint32_t chunk, const Tensor& value) {
        if (value.numel() != std::int64_t(buffer_.size()))
            throw std::runtime_error("reference hidden-state geometry differs");
        auto cpu = ops::contiguous(ops::cast(value, kFloat32, Device(kCPU)));
        stream_.seekp(offset(chunk));
        stream_.write(reinterpret_cast<const char*>(cpu.dense().data_ptr<float>()), bytes_);
        stream_.flush();
        if (!stream_) throw std::runtime_error("cannot write reference hidden-state scratch");
    }
    Tensor restore(std::uint32_t chunk, Device device) {
        stream_.seekg(offset(chunk));
        stream_.read(reinterpret_cast<char*>(buffer_.data()), bytes_);
        if (!stream_) throw std::runtime_error("cannot read reference hidden-state scratch");
        auto host = ops::from_blob(buffer_.data(), {1, context_, width_},
            TensorOptions().device(kCPU).dtype(kFloat32));
        return ops::contiguous(ops::cast(host, kFloat16, device));
    }
private:
    std::streamoff offset(std::uint32_t chunk) const {
        if (chunk && bytes_ > std::uint64_t(std::numeric_limits<std::streamoff>::max()) / chunk)
            throw std::overflow_error("reference hidden-state file offset overflows");
        return std::streamoff(chunk) * bytes_;
    }
    std::filesystem::path path_;
    std::int64_t context_, width_;
    std::vector<float> buffer_;
    std::size_t bytes_;
    std::fstream stream_;
};

bool out_of_memory(const std::exception& error) {
    const std::string message = error.what();
    return message.find("out of memory") != std::string::npos ||
        message.find("memory allocation") != std::string::npos;
}

void cleanup(cuda::Program& program, std::unique_ptr<models::Layers::State>& state) noexcept {
    try { program.drain(); }
    catch (const std::exception& error) { std::cerr << "reference_cleanup error=" << error.what() << '\n'; }
    catch (...) { std::cerr << "reference_cleanup error=unknown\n"; }
    state.reset();
}

} // namespace

int reference(cuda::Program& program, const std::filesystem::path& input,
        const std::filesystem::path& output, std::int64_t requested_context, bool resident) {
    std::ifstream tokens_file(input, std::ios::binary);
    std::uint32_t geometry[2]{};
    tokens_file.read(reinterpret_cast<char*>(geometry), sizeof(geometry));
    const auto chunks = geometry[0], context = geometry[1];
    if (!tokens_file || !chunks || context < 32 || context != requested_context ||
        std::uint64_t(chunks) * context > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("invalid native reference token geometry");
    std::vector<std::int32_t> tokens(std::size_t(chunks) * context);
    tokens_file.read(reinterpret_cast<char*>(tokens.data()), tokens.size() * sizeof(tokens[0]));
    if (!tokens_file || tokens_file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("invalid native reference token payload");
    if (std::filesystem::exists(output)) throw std::runtime_error("reference scratch output already exists");
    program.load_layers();
    auto& model = program.layers();
    auto& executor = program.executor();
    const auto info = model.info();
    if (info.layers < 1 || info.width < 1 || info.vocab < 1)
        throw std::runtime_error("invalid native reference model geometry");
    for (const auto token : tokens)
        if (token < 0 || token >= info.vocab)
            throw std::runtime_error("reference token is outside the source vocabulary");
    const int primary = program.config().layer_placement.primary_device();
    auto scope = executor.enter(primary);
    const Device device(kCUDA, primary);
    const auto mode = executor.forward_mode(primary);
    const models::ForwardOptions options{mode.kv_capacity, mode.trace, mode.trace_layer, primary};
    Hidden hidden(output, context, info.width);
    std::unique_ptr<models::Layers::State> state;
    try {
        std::vector<std::int64_t> ids(context);
        for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
            for (std::uint32_t position = 0; position < context; ++position)
                ids[position] = tokens[std::size_t(chunk) * context + position];
            auto input_ids = ops::reshape(ops::tensor(ids, Device(kCPU)), {1, context});
            hidden.store(chunk, model.embedding(input_ids));
        }
        auto positions = ops::arange(0, context, device);
        if (resident) {
            try { model.load_resident(); }
            catch (const std::exception& error) {
                if (!out_of_memory(error)) throw;
                program.drain();
                model.release();
                executor.trim(device);
                resident = false;
                std::cout << "{\"event\":\"source_loading\",\"layerwise\":true,\"requested_layerwise\":false,"
                             "\"fallback_reason\":\"insufficient_memory\"}" << std::endl;
            }
        }
        for (int index = 0; index < info.layers; ++index) {
            if (!resident) model.load_layer(index);
            for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
                state = model.make_state(index);
                hidden.store(chunk, model.forward(index, hidden.restore(chunk, device), *state, positions, options));
                program.drain();
                state.reset();
            }
            program.drain();
            if (!resident) {
                model.release();
                executor.trim(device);
            }
            std::cout << "{\"event\":\"wt2_layer\",\"layer\":" << index + 1
                      << ",\"layers\":" << info.layers << "}" << std::endl;
        }
        program.drain();
        model.release();
        executor.trim(device);
        std::ofstream logits(output, std::ios::binary);
        if (!logits) throw std::runtime_error("cannot create reference logits scratch");
        const std::int64_t first = context / 2, scored = context - first - 1;
        for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
            auto normalized = model.normalize(hidden.restore(chunk, device), 1, context);
            for (std::int64_t begin = 0; begin < scored; begin += 32) {
                const auto count = std::min<std::int64_t>(32, scored - begin);
                auto value = ops::cast(ops::reshape(ops::slice(normalized, 1, first + begin,
                    first + begin + count), {count, info.width}), kFloat16);
                std::vector<float> row_logits(elements(count, info.vocab));
                for (std::int64_t row = 0; row < info.vocab; row += 4096) {
                    const auto rows = std::min<std::int64_t>(4096, info.vocab - row);
                    auto result = ops::contiguous(ops::cast(model.head(value, row, rows), kFloat32, Device(kCPU)));
                    for (std::int64_t token = 0; token < count; ++token)
                        std::memcpy(row_logits.data() + token * info.vocab + row,
                            result.dense().data_ptr<float>() + token * rows, rows * sizeof(float));
                }
                logits.write(reinterpret_cast<const char*>(row_logits.data()), row_logits.size() * sizeof(float));
                if (!logits) throw std::runtime_error("cannot write reference logits scratch");
            }
            std::cout << "{\"event\":\"wt2_head\",\"chunk\":" << chunk + 1
                      << ",\"chunks\":" << chunks << "}" << std::endl;
        }
        program.drain();
        logits.close();
        if (!logits) throw std::runtime_error("cannot finalize reference logits scratch");
        return 0;
    } catch (...) {
        const auto failure = std::current_exception();
        cleanup(program, state);
        std::rethrow_exception(failure);
    }
}

} // namespace tyloi::eval
