#pragma once
#include "mfq_tensor_backend.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "recurrent_copy_kernel.h"

namespace mfq::cuda {
// A rollback restores independent recurrent states. Retain their owners while
// collecting copies, then submit them together; dependent copies retain order.
class RecurrentCopyBatch {
    struct Copy { Tensor destination,source; };
    inline static thread_local RecurrentCopyBatch* current_=nullptr;
    RecurrentCopyBatch* previous_=nullptr;
    std::vector<Copy> owners_;
    std::vector<DmaCopy> copies_;
    cudaStream_t stream_=nullptr;
    int device_=-1;
    bool enabled_=false;
    std::size_t batched_=0;
    static bool overlaps(const void* a,std::size_t an,const void* b,std::size_t bn) {
        const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
        return x<=y?y-x<an:x-y<bn;
    }
    void append(Tensor& destination,const Tensor& source) {
        if(!enabled_ || !destination.is_cuda() || !source.is_cuda() ||
                destination.device()!=source.device() || !destination.is_contiguous() ||
                !source.is_contiguous() || destination.scalar_type()!=source.scalar_type() ||
                destination.numel()!=source.numel()) {
            flush();destination.copy_(source);return;
        }
        if(!source.numel() || destination.data_ptr()==source.data_ptr())return;
        if(overlaps(destination.data_ptr(),destination.nbytes(),source.data_ptr(),source.nbytes())) {
            flush();destination.copy_(source);return;
        }
        const auto stream=current_stream(destination.get_device()).stream();
        if(device_!=destination.get_device() || stream_!=stream) {
            flush();device_=destination.get_device();stream_=stream;
        }
        // CUDA batch copies are unordered. A data dependency must form a new
        // packet, including overlapping destinations, even when owners differ.
        for(const auto& copy:copies_)if(
                overlaps(copy.destination,copy.bytes,source.data_ptr(),source.nbytes()) ||
                overlaps(copy.source,copy.bytes,destination.data_ptr(),destination.nbytes()) ||
                overlaps(copy.destination,copy.bytes,destination.data_ptr(),destination.nbytes())) {
            flush();break;
        }
        copies_.push_back({destination.data_ptr(),source.data_ptr(),source.nbytes()});
        owners_.push_back({destination,source});
    }
public:
    explicit RecurrentCopyBatch(bool enabled=true):previous_(current_) {
        if(previous_)previous_->flush();
        const auto* option=std::getenv("TYLOINFER_RECURRENT_COPY_BATCH");
        enabled_=enabled && (!option || option[0]!='0');
        current_=this;
    }
    RecurrentCopyBatch(const RecurrentCopyBatch&)=delete;
    RecurrentCopyBatch& operator=(const RecurrentCopyBatch&)=delete;
    // The caller flushes explicitly before consuming restored state. An
    // exception cancels pending copies and lets request recovery reset state.
    ~RecurrentCopyBatch(){current_=previous_;}
    void flush() {
        if(copies_.empty())return;
        DeviceGuard guard(device_);
        for(const auto& owner:owners_) {
            owner.destination.record_stream(reinterpret_cast<std::uintptr_t>(stream_));
            owner.source.record_stream(reinterpret_cast<std::uintptr_t>(stream_));
        }
        copy_recurrent_states(copies_.data(),copies_.size(),stream_);
        batched_+=copies_.size();
        copies_.clear();owners_.clear();
    }
    std::size_t batched_copies()const{return batched_;}
    static void copy(Tensor& destination,const Tensor& source) {
        if(current_)current_->append(destination,source);else destination.copy_(source);
    }
};
inline void copy_recurrent_state(Tensor& destination,const Tensor& source) {
    RecurrentCopyBatch::copy(destination,source);
}
}
#else
namespace mfq::cuda {
inline void copy_recurrent_state(mfq_tensor_backend::Tensor& destination,
        const mfq_tensor_backend::Tensor& source) {destination.copy_(source);}
}
#endif
