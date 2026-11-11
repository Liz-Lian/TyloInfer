#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/working_set.h"

#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/generation/sequence.h"

#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace tyloi::cuda {
namespace {
#if TYLOINFER_ENABLE_MFQ_COMPAT
using Key = std::array<std::uintptr_t, 11>;
Key key(const NintTensor& w) {
    return {reinterpret_cast<std::uintptr_t>(w.q_packed.data_ptr()),
        reinterpret_cast<std::uintptr_t>(w.row_q_bits.data_ptr()),
        reinterpret_cast<std::uintptr_t>(w.row_q_bit_offsets.data_ptr()),
        reinterpret_cast<std::uintptr_t>(w.sub_scale.data_ptr()),
        reinterpret_cast<std::uintptr_t>(w.sub_min.data_ptr()),
        reinterpret_cast<std::uintptr_t>(w.neuron_scale.data_ptr()),
        reinterpret_cast<std::uintptr_t>(w.neuron_min.data_ptr()),
        std::uintptr_t(w.neuron_len), std::uintptr_t(w.gs), std::uintptr_t(w.out),
        std::uintptr_t(w.q_packed.get_device())};
}
#endif
class WorkingSet;
thread_local WorkingSet* active = nullptr;

class WorkingSet final : public tyloi::Executor::Work {
    class Activation final : public tyloi::Executor::Scope {
    public:
        explicit Activation(WorkingSet& owner) : previous_(active) { active = &owner; }
        ~Activation() override { active = previous_; }
    private:
        WorkingSet* previous_;
    };
public:
    explicit WorkingSet(Executor& executor) : executor_(executor) {}
    std::unique_ptr<tyloi::Executor::Scope> enter() override {
        if (&Executor::current() != &executor_) throw std::logic_error("prefill work executor is not active");
        return std::make_unique<Activation>(*this);
    }
    void step(const tyloi::LayerEvent& event) override {
        using Phase = tyloi::LayerEvent::Phase;
#if TYLOINFER_ENABLE_MFQ_COMPAT
        using Mode = tyloi::LayerEvent::Mode;
#endif
        switch (event.phase) {
        case Phase::begin: {
            if (layer_) throw std::logic_error("prefill layer already started");
            layer_ = true;
#if TYLOINFER_ENABLE_MFQ_COMPAT
            enabled_ = event.mode == Mode::packed;
            if (event.mode == Mode::tiled) {
                const auto option = options::nint_prefill_decode_reuse();
                enabled_ = !option || *option;
            }
#else
            (void)event.mode;
#endif
            break;
        }
        case Phase::clear:
            if (!layer_) throw std::logic_error("prefill layer is not active");
#if TYLOINFER_ENABLE_MFQ_COMPAT
            decoded_.clear();
#endif
            break;
        case Phase::end:
            if (!layer_) throw std::logic_error("prefill layer is not active");
#if TYLOINFER_ENABLE_MFQ_COMPAT
            decoded_.clear();
#endif
            layer_ = false;
            enabled_ = false;
            break;
        }
    }
    void finish() override {
        if (layer_) throw std::logic_error("prefill ended within a layer");
    }
    bool enabled() const noexcept { return layer_ && enabled_; }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    DenseTensor find(const NintTensor& weight) const {
        if (!enabled()) return {};
        const auto found = decoded_.find(key(weight));
        return found == decoded_.end() ? DenseTensor{} : found->second;
    }
    void store(const NintTensor& weight, const DenseTensor& value) {
        if (enabled()) decoded_.emplace(key(weight), value);
    }
#endif
private:
    Executor& executor_;
    bool layer_ = false;
    bool enabled_ = false;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::map<Key, DenseTensor> decoded_;
#endif
};
}

std::unique_ptr<tyloi::Executor::Work> make_working_set(Executor& executor) {
    return std::make_unique<WorkingSet>(executor);
}
bool decoded_work_active() noexcept { return active && active->enabled(); }
#if TYLOINFER_ENABLE_MFQ_COMPAT
DenseTensor decoded_work_find(const NintTensor& weight) {
    return active ? active->find(weight) : DenseTensor{};
}
void decoded_work_store(const NintTensor& weight, const DenseTensor& value) {
    if (active) active->store(weight, value);
}
#endif
}

#endif // TYLOINFER_NATIVE_CUDA
