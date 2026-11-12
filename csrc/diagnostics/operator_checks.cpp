#include "operator_checks.h"

#include "engine/program.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/native/context.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
/** Retain every input and output until the execution owner has drained. */
struct Tensors {
    tyloi::DenseTensor q, k, v, g, beta, state, qk, weight, bias;
    std::vector<tyloi::DenseTensor> result, host, staging;
};

/** Check dimensions and byte counts before allocating or reading a tensor. */
std::size_t checked_f32_count(const std::vector<int64_t>& shape) {
    std::size_t count = 1;
    const auto max_count = std::min(
        std::numeric_limits<std::size_t>::max() / sizeof(float),
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float));
    for (const auto extent : shape) {
        if (extent <= 0 || static_cast<std::uint64_t>(extent) > max_count / count)
            throw std::invalid_argument("operator diagnostic tensor shape is too large");
        count *= static_cast<std::size_t>(extent);
    }
    return count;
}

/** Check multiplication and addition before constructing convolution shapes. */
int64_t checked_product(int64_t a, int64_t b) {
    if (a <= 0 || b <= 0 || a > std::numeric_limits<int64_t>::max() / b)
        throw std::invalid_argument("operator diagnostic tensor shape is too large");
    return a * b;
}

/** Drain during unwinding before the retained tensor handles are destroyed. */
class DrainGuard {
public:
    explicit DrainGuard(tyloi::cuda::Program& program) : program_(program) {}
    ~DrainGuard() noexcept {
        if (!active_) return;
        try { program_.drain(); }
        catch (const std::exception& error) { std::cerr << "operator_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "operator_cleanup error=unknown\n"; }
    }
    /** Drain explicitly so a successful operation cannot hide a drain error. */
    void finish() { program_.drain(); active_ = false; }
private:
    tyloi::cuda::Program& program_;
    bool active_ = true;
};
}

/** Read a fixed-shape F32 tensor in the legacy operator file format. */
static tyloi::DenseTensor read_f32_tensor(
        const std::filesystem::path & path,
        const std::vector<int64_t> & shape,
        std::vector<tyloi::DenseTensor>& staging,
        tyloi::Device target) {
    std::vector<float> values(checked_f32_count(shape));
    std::ifstream input(path, std::ios::binary);
    TYLOINFER_RUNTIME_CHECK(input, "failed to open ", path.string());
    input.read(
        reinterpret_cast<char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    TYLOINFER_RUNTIME_CHECK(
        input.gcount() ==
            static_cast<std::streamsize>(values.size() * sizeof(float)),
        "short f32 tensor read from ", path.string());
    auto host = tyloi::cuda::from_blob(
               values.data(), shape,
               tyloi::TensorOptions().dtype(tyloi::kFloat32))
        .clone();
    staging.push_back(host);
    auto device = host.to(target);
    staging.push_back(device);
    return device.contiguous();
}

/** Write a tensor in the legacy contiguous F32 file format. */
static void write_f32_tensor(
        const std::filesystem::path & path,
        const tyloi::DenseTensor & tensor,
        std::vector<tyloi::DenseTensor>& retained) {
    retained.push_back(tensor.to(tyloi::kCPU).to(tyloi::kFloat32).contiguous());
    const auto& host = retained.back();
    std::ofstream output(path, std::ios::binary);
    TYLOINFER_RUNTIME_CHECK(output, "failed to create ", path.string());
    output.write(
        reinterpret_cast<const char *>(host.data_ptr<float>()),
        static_cast<std::streamsize>(host.numel() * sizeof(float)));
    TYLOINFER_RUNTIME_CHECK(output, "failed to write ", path.string());
}

int run_gdn_operator_check(
        tyloi::cuda::Program& program,
        const std::string & input_dir,
        const std::string & output_path,
        const std::string & state_path,
        int64_t tokens,
        int64_t q_heads,
        int64_t v_heads,
        int64_t head_dim) {
    TYLOINFER_RUNTIME_CHECK(
        tokens >= 1 && q_heads >= 1 && v_heads >= q_heads && head_dim >= 1,
        "invalid GDN diagnostic shape");
    TYLOINFER_RUNTIME_CHECK(
        v_heads % q_heads == 0,
        "GDN diagnostic value heads must be divisible by query heads");
    checked_f32_count({1, tokens, q_heads, head_dim});
    checked_f32_count({1, tokens, v_heads, head_dim});
    checked_f32_count({1, tokens, v_heads});
    checked_f32_count({1, v_heads, head_dim, head_dim});
    const std::filesystem::path root(input_dir);
    const tyloi::Device device(tyloi::kCUDA, tyloi::cuda::current_stream().device_index());
    Tensors tensors;
    DrainGuard drain(program);
    tensors.staging.reserve(12);
    auto& [q, k, v, g, beta, state, qk, weight, bias, result, host, staging] = tensors;
    q = read_f32_tensor(
        root / "q.bin", {1, tokens, q_heads, head_dim}, staging, device)
                 .permute({0, 2, 1, 3})
                 .contiguous();
    k = read_f32_tensor(
        root / "k.bin", {1, tokens, q_heads, head_dim}, staging, device)
                 .permute({0, 2, 1, 3})
                 .contiguous();
    v = read_f32_tensor(
        root / "v.bin", {1, tokens, v_heads, head_dim}, staging, device)
                 .permute({0, 2, 1, 3})
                 .contiguous();
    g = read_f32_tensor(
        root / "g.bin", {1, tokens, v_heads}, staging, device)
                 .permute({0, 2, 1})
                 .contiguous();
    beta = read_f32_tensor(
        root / "beta.bin", {1, tokens, v_heads}, staging, device)
                    .permute({0, 2, 1})
                    .contiguous();
    state = read_f32_tensor(
        root / "state.bin", {1, v_heads, head_dim, head_dim}, staging, device);
    result = gdn_inplace_transposed_tiled_cuda(
        q, k, v, g, beta, state);
    write_f32_tensor(output_path, result[0], host);
    write_f32_tensor(state_path, result[1], host);
    drain.finish();
    std::cout << "mfq_gdn_operator"
              << " t=" << tokens
              << " hq=" << q_heads
              << " hv=" << v_heads
              << " d=" << head_dim
              << " output=" << output_path
              << " state=" << state_path << "\n";
    return 0;
}

int run_linear_conv_operator_check(
        tyloi::cuda::Program& program,
        const std::string & input_dir,
        const std::string & output_dir,
        int64_t tokens,
        int64_t q_heads,
        int64_t v_heads,
        int64_t key_dim,
        int64_t value_dim,
        int64_t kernel_size,
        double eps) {
    TYLOINFER_RUNTIME_CHECK(
        tokens >= 2 && q_heads >= 1 && v_heads >= 1 &&
            key_dim >= 1 && value_dim >= 1 &&
            kernel_size >= 2 && kernel_size <= 8,
        "invalid linear-conv diagnostic shape");
    const int64_t qk_width = checked_product(checked_product(2, q_heads), key_dim);
    const int64_t v_width = checked_product(v_heads, value_dim);
    if (qk_width > std::numeric_limits<int64_t>::max() - v_width)
        throw std::invalid_argument("operator diagnostic tensor shape is too large");
    const int64_t channels = qk_width + v_width;
    checked_f32_count({1, kernel_size - 1, channels});
    checked_f32_count({1, tokens, qk_width});
    checked_f32_count({1, tokens, v_width});
    checked_f32_count({channels, 1, kernel_size});
    const std::filesystem::path input_root(input_dir);
    const std::filesystem::path output_root(output_dir);
    const tyloi::Device device(tyloi::kCUDA, tyloi::cuda::current_stream().device_index());
    std::filesystem::create_directories(output_root);
    Tensors tensors;
    DrainGuard drain(program);
    tensors.staging.reserve(10);
    auto& [q, k, v, g, beta, state, qk, weight, bias, result, host, staging] = tensors;
    state = read_f32_tensor(
        input_root / "state.bin",
        {1, kernel_size - 1, channels}, staging, device);
    qk = read_f32_tensor(
                  input_root / "qk.bin",
                  {1, tokens, qk_width}, staging, device)
                  .to(tyloi::kHalf)
                  .contiguous();
    v = read_f32_tensor(
                 input_root / "v.bin",
                 {1, tokens, v_width}, staging, device)
                 .to(tyloi::kHalf)
                 .contiguous();
    weight = read_f32_tensor(
        input_root / "weight.bin",
        {channels, 1, kernel_size}, staging, device);
    if (std::filesystem::exists(input_root / "bias.bin")) {
        bias = read_f32_tensor(
            input_root / "bias.bin", {channels}, staging, device);
    } else {
        bias = tyloi::cuda::empty(
            {0},
            tyloi::TensorOptions()
                .device(device)
                .dtype(tyloi::kFloat32));
    }
    result = linear_conv_qkv_prefill_cuda(
        state, qk, v, weight, bias,
        q_heads, v_heads, key_dim, value_dim, eps);
    write_f32_tensor(output_root / "q.bin", result[0], host);
    write_f32_tensor(output_root / "k.bin", result[1], host);
    write_f32_tensor(output_root / "v.bin", result[2], host);
    write_f32_tensor(output_root / "state.bin", result[3], host);
    drain.finish();
    std::cout << "mfq_linear_conv_operator"
              << " t=" << tokens
              << " hq=" << q_heads
              << " hv=" << v_heads
              << " dk=" << key_dim
              << " dv=" << value_dim
              << " kernel=" << kernel_size
              << " output=" << output_dir << "\n";
    return 0;
}
