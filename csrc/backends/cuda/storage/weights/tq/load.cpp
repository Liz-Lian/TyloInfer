#include "storage/weights/tq/load.h"

#include "model_source/tq_source.h"

#include <climits>
#include <cstring>
#include <stdexcept>
#include <utility>

std::vector<std::uint8_t> tq_image(
        const std::vector<std::uint8_t>& blob, const mfq::TqLayout& layout) {
    // Keep the packed words aligned in the single image consumed by CUDA.
    const auto words_offset = (layout.packed + 3) & ~std::size_t(3);
    const auto relative_offset = words_offset + layout.packed_bytes;
    std::vector<std::uint8_t> image(relative_offset + 32);
    std::memcpy(image.data(), blob.data(), layout.packed);
    std::memcpy(image.data() + words_offset,
                blob.data() + layout.packed, layout.packed_bytes);
    constexpr std::uint16_t relative[]{
        0x3400, 0x34d0, 0x35ca, 0x36f7, 0x3830, 0x390a, 0x3a10, 0x3b4b,
        0x3c63, 0x3d47, 0x3e59, 0x3fa4, 0x4098, 0x4187, 0x42a6, 0x4400};
    std::memcpy(image.data() + relative_offset, relative, sizeof(relative));
    return image;
}

TqWeight tq_view(mfq_tensor_backend::Tensor image,
                 const mfq::TqLayout& layout, std::int64_t columns) {
    if (columns < 1 || columns > layout.width || layout.rows > INT_MAX ||
        layout.width > INT_MAX) {
        throw std::invalid_argument("invalid TQ CUDA logical shape");
    }
    const auto words_offset = (layout.packed + 3) & ~std::size_t(3);
    const auto relative_offset = words_offset + layout.packed_bytes;
    TqWeight result;
    result.image = std::move(image);
    if (result.image.is_cuda()) {
        result.gemm = tq_gemm_resources(result.image.get_device());
    }
    auto* data = result.image.data_ptr<std::uint8_t>();
    int gslog = 0;
    for (auto gs = layout.gs; gs > 1; gs >>= 1) ++gslog;
    result.view = {
        reinterpret_cast<const std::uint32_t*>(data + words_offset),
        reinterpret_cast<const std::uint64_t*>(data + 64),
        data + layout.scales, data + relative_offset, nullptr,
        static_cast<int>(layout.rows), static_cast<int>(layout.width),
        static_cast<int>(columns), gslog, int(layout.columns)};
    result.out = layout.rows;
    result.neuron_len = columns;
    return result;
}

TqWeight to_device_tq(const std::vector<std::uint8_t>& blob,
                      std::int64_t columns, int device, bool cuda) {
    const mfq::TqLayout layout(blob.data(), blob.size());
    const auto image = tq_image(blob, layout);
    const MfqCudaGuard guard(device);
    auto storage = mfq_tensor_backend::from_blob(
        const_cast<std::uint8_t*>(image.data()),
        {static_cast<std::int64_t>(image.size())},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8)).clone();
    if (cuda) {
        storage = storage.to(
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, device),
            false, false).contiguous();
    }
    return tq_view(std::move(storage), layout, columns);
}

TqWeight load_tq_weight(const mfq::ModelSource& source,
                         const std::string& name, int device, bool cuda) {
    const auto shape = mfq::tq_projection_shape(source, name);
    const auto bytes = source.read(name);
    std::vector<std::uint8_t> blob(bytes.size());
    std::memcpy(blob.data(), bytes.data(), bytes.size());
    auto result = to_device_tq(blob, shape[1], device, cuda);
    if (result.out != shape[0]) {
        throw std::invalid_argument("TQ projection output shape differs: " + name);
    }
    return result;
}

std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(
        CudaExecutionContext& execution, const mfq::ModelSource& source,
        const std::string& name) {
    const auto shape = mfq::tq_shape(source, name);
    if (shape.size() != 3) {
        throw std::invalid_argument(
            "TQ expert bank must be [experts,out,in]: " + name);
    }
    auto table = mfq::tq_table(source, name);
    if (shape[0] > UINT32_MAX || table->rows() !=
            std::uint64_t(shape[0]) * std::uint64_t(shape[1]) ||
        table->width() != shape[2]) {
        throw std::invalid_argument("TQ routed projection shape differs: " + name);
    }
    auto cache = ensure_tq_expert_cache(execution);
    const auto source_id = register_tq_expert_source(cache, table->storage_bytes());
    return std::make_shared<TqRoutedWeight>(
        std::move(table), std::move(cache), source_id,
        shape[0], shape[1], shape[2]);
}
