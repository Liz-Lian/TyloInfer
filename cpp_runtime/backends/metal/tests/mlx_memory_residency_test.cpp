#include "mlx_memory_residency.h"
#include "mlx_pinned_pool_storage.h"

#include <mlx/allocator.h>
#include <mlx/mlx.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <optional>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/resource.h>

static std::optional<vm_region_basic_info_data_64_t> page_info(void* pointer) {
    mach_vm_address_t address = reinterpret_cast<std::uintptr_t>(pointer);
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    if (result != KERN_SUCCESS || address > reinterpret_cast<std::uintptr_t>(pointer))
        return std::nullopt;
    return info;
}

static bool user_wired(void* pointer) {
    const auto info = page_info(pointer);
    if (!info) throw std::runtime_error("cannot inspect anonymous pool page wiring");
    return info->user_wired_count != 0;
}

static void* check_pool_page_lock() {
    mfq::metal::detail::MlxPinnedPoolStorage storage;
    const auto page = std::size_t(getpagesize());
    storage.bytes = page * 4;
    storage.data = mmap(nullptr, storage.bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (storage.data == MAP_FAILED) throw std::runtime_error("cannot map page-lock fixture");
    auto* tail = static_cast<std::byte*>(storage.data) + page * 2;
    if (!storage.pin_prefix(page * 2) || storage.locked_bytes != page * 2
        || !user_wired(storage.data) || user_wired(tail))
        throw std::runtime_error("anonymous pool prefix was not page-locked");
    struct Restore {
        rlimit previous{};
        Restore() {
            if (getrlimit(RLIMIT_MEMLOCK, &previous)) throw std::runtime_error("cannot read page-lock limit");
            auto limit = previous; limit.rlim_cur = 0;
            if (setrlimit(RLIMIT_MEMLOCK, &limit)) throw std::runtime_error("cannot constrain page-lock fixture");
        }
        ~Restore() { if (setrlimit(RLIMIT_MEMLOCK, &previous)) std::abort(); }
    };
    {
        Restore limit;
        if (storage.pin_prefix(page * 4) || storage.lock_error == 0 || storage.locked_bytes != page * 2
            || !user_wired(storage.data) || user_wired(tail))
            throw std::runtime_error("failed pool pin changed the previous locked prefix");
    }
    if (!storage.pin_prefix(page * 4) || !user_wired(tail))
        throw std::runtime_error("pool page-lock retry failed");
    if (!storage.pin_prefix(page) || storage.locked_bytes != page || user_wired(tail))
        throw std::runtime_error("pool shrink retained a page lock beyond its budget");
    if (!storage.pin_prefix(0) || storage.locked_bytes || user_wired(storage.data))
        throw std::runtime_error("pool page locks were not released");
    if (storage.pin_prefix(storage.bytes + page) || storage.lock_error != EINVAL || storage.locked_bytes)
        throw std::runtime_error("invalid pool page lock modified the mapping");
    if (!storage.pin_prefix(storage.bytes)) throw std::runtime_error("cannot prepare pinned pool destruction");
    return storage.data;
}

int main() {
    const auto released = check_pool_page_lock();
    if (page_info(released)) throw std::runtime_error("pinned pool destruction retained its mapping");
    using mfq::metal::MlxMemoryResidency;
    using mlx::core::array;
    constexpr std::size_t limit = 128 << 20;
    constexpr int bytes = 96 << 20;
    mlx::core::set_default_device(mlx::core::Device::gpu);
    if (!__builtin_available(macOS 15, *)) return 0;
    MlxMemoryResidency::configure(limit);
    auto allocate = [](int size) {
        auto value = array(mlx::core::allocator::malloc(size),
            mlx::core::Shape{size}, mlx::core::uint8);
        std::memset(value.data<std::uint8_t>(), 1, size);
        mfq::metal::MlxWeightResidency::track(value);
        return value;
    };
    std::optional<array> scratch{allocate(bytes)};
    std::optional<array> weights{allocate(bytes)};
    const auto saturated = MlxMemoryResidency::wired_bytes();
    scratch.reset();
    mlx::core::clear_cache();
    const auto before = MlxMemoryResidency::wired_bytes();
    assert(before < bytes);
    mlx::core::set_wired_limit(limit);
    assert(MlxMemoryResidency::wired_bytes() == before);
    MlxMemoryResidency::refresh();
    const auto after = MlxMemoryResidency::wired_bytes();
    assert(after >= bytes && after <= limit);
    bool weight_budget_rejected = false;
    try { mfq::metal::MlxWeightResidency::finish_load(bytes - 1); }
    catch (const std::runtime_error&) { weight_budget_rejected = true; }
    assert(weight_budget_rejected);
    mfq::metal::MlxWeightResidency::begin_load();
    mfq::metal::MlxWeightResidency::track(*weights);
    auto alias = mlx::core::reshape(*weights, mlx::core::Shape{bytes / 2, 2});
    mlx::core::eval(alias);
    mfq::metal::MlxWeightResidency::track(alias);
    auto residency = mfq::metal::MlxWeightResidency::finish_load(limit);
    assert(residency && residency->bytes() == weights->buffer_size());
    assert(weights->data<std::uint8_t>()[0] == 1);
    assert(weights->data<std::uint8_t>()[bytes - 1] == 1);
    assert(!mfq::metal::MlxWeightResidency::finish_load(limit));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    assert(residency->bytes() == weights->buffer_size());
    assert(weights->data<std::uint8_t>()[0] == 1);
    assert(weights->data<std::uint8_t>()[bytes - 1] == 1);
    alias = array({0}, mlx::core::uint8);
    std::optional<array> excess{allocate(bytes)};
    bool rejected = false;
    try { MlxMemoryResidency::refresh(); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    excess.reset();
    weights.reset();
    mlx::core::clear_cache();
    for (int attempt = 0; attempt < 100 && residency->bytes() != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    assert(residency->bytes() == 0);
    residency.reset();
    mlx::core::set_wired_limit(0);
    assert(mlx::core::get_active_memory() < (1 << 20));
    std::cout << "Metal residency: saturated=" << saturated
              << " released=" << before << " rewired=" << after << '\n';
}
