#pragma once
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "mfq_cuda_context.h"
#include "mfq_native_tensor.h"
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <cstring>

// One bounded pinned allocation; expert leases own independently reusable
// intervals. Releasing an entire lease never allocates, including under pressure.
class MoePinnedExpertArena {
    using Free=std::map<std::size_t,std::size_t>;
    using Fits=std::set<std::pair<std::size_t,std::size_t>>;
    struct State {
        mfq::cuda::HostBuffer storage;
        void* mapped=nullptr;
        std::mutex mutex;
        Free free;
        Fits fits;
        std::size_t used=0;
        explicit State(std::size_t bytes):storage(bytes,true) {
            TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,storage.data(),0));
            free.emplace(0,bytes);fits.emplace(bytes,0);mfq::cuda::charge_tensor_host_bytes(bytes);
        }
        ~State(){mfq::cuda::tensor_host_bytes.fetch_sub(storage.size());}
        // Caller holds mutex; both tree nodes were allocated by the owner.
        void release(Free::node_type interval,Fits::node_type size_interval,bool allocated=true) {
            if(allocated)used-=interval.mapped();
            auto next=free.lower_bound(interval.key());
            if(next!=free.begin()) {
                const auto previous=std::prev(next);
                if(previous->first+previous->second==interval.key()) {
                    fits.erase({previous->second,previous->first});
                    interval.key()=previous->first;interval.mapped()+=previous->second;
                    free.erase(previous);
                }
            }
            if(next!=free.end() && interval.key()+interval.mapped()==next->first) {
                fits.erase({next->second,next->first});
                interval.mapped()+=next->second;free.erase(next);
            }
            size_interval.value()={interval.mapped(),interval.key()};
            free.insert(std::move(interval));fits.insert(std::move(size_interval));
        }
    };
public:
    struct Block {
        std::shared_ptr<State> state;
        Free::node_type interval;
        Fits::node_type size_interval;
        ~Block() {
            if(!state)return;
            std::lock_guard lock(state->mutex);
            state->release(std::move(interval),std::move(size_interval));
        }
        // Only an exclusive owner may narrow a lease. Live bytes retain the
        // same address; unused paired-image space becomes reusable intervals.
        std::size_t retain_range(std::size_t offset,std::size_t bytes) {
            const auto original=interval.mapped();
            if(offset%16 || !bytes || bytes>SIZE_MAX-15 || offset>original ||
                    bytes>original-offset)throw std::invalid_argument("Invalid pinned expert subrange");
            bytes=(bytes+15)&~std::size_t(15);
            const auto base=interval.key(),tail=original-offset-bytes;
            if(!offset && !tail)return 0;
            Free released;Fits sizes;
            if(offset){released.emplace(base,offset);sizes.emplace(offset,base);}
            if(tail){released.emplace(base+offset+bytes,tail);sizes.emplace(tail,base+offset+bytes);}
            // Allocate metadata before changing either tree or the live lease.
            std::lock_guard lock(state->mutex);
            interval.key()=base+offset;interval.mapped()=bytes;
            size_interval.value()={bytes,base+offset};
            while(!released.empty()) {
                auto node=released.extract(released.begin());
                auto size=sizes.extract({node.mapped(),node.key()});
                state->release(std::move(node),std::move(size));
            }
            return original-bytes;
        }
        // The caller owns every view of this idle lease. Shift into the
        // preceding hole without registering pages or allocating tree nodes.
        std::size_t compact_left() {
            std::lock_guard lock(state->mutex);
            const auto original=interval.key(),bytes=interval.mapped();
            auto prior=state->free.lower_bound(original);
            if(prior==state->free.begin())return 0;
            --prior;
            if(prior->first+prior->second!=original)return 0;
            const auto base=prior->first,hole=prior->second;
            auto released=state->free.extract(prior);
            auto size=state->fits.extract({hole,base});
            std::memmove(static_cast<std::uint8_t*>(state->storage.data())+base,data(),bytes);
            interval.key()=base;size_interval.value()={bytes,base};
            released.key()=base+bytes;size.value()={hole,base+bytes};
            state->release(std::move(released),std::move(size),false);
            return bytes;
        }
        std::uint8_t* data() const {return static_cast<std::uint8_t*>(state->storage.data())+interval.key();}
        std::uint8_t* mapped_data() const {return static_cast<std::uint8_t*>(state->mapped)+interval.key();}
        std::size_t bytes() const {return interval.mapped();}
        std::size_t offset() const {return interval.key();}
        const void* allocation_id() const {return state.get();}
    };
    explicit MoePinnedExpertArena(std::size_t bytes):state_(std::make_shared<State>(bytes)) {}
    std::shared_ptr<Block> allocate(std::size_t bytes) {
        if(!bytes || bytes>SIZE_MAX-15)return {};
        bytes=(bytes+15)&~std::size_t(15);
        std::lock_guard lock(state_->mutex);
        const auto fit=state_->fits.lower_bound({bytes,0});
        if(fit==state_->fits.end())return {};
        // A full arena is a common probe under the host pinning limit. Do
        // not allocate three temporary heap nodes just to reject that probe.
        auto block=std::make_shared<Block>();
        Free node;node.emplace(0,bytes);block->interval=node.extract(node.begin());
        Fits size_node;size_node.emplace(bytes,0);block->size_interval=size_node.extract(size_node.begin());
        // The size index avoids scanning fragmented intervals and preserves
        // larger holes for the next, larger expert format.
        {
            auto it=state_->free.find(fit->second);
            const auto offset=it->first;
            if(it->second==bytes) {state_->free.erase(it);state_->fits.erase(fit);}
            else {
                auto rest=state_->free.extract(it);rest.key()+=bytes;rest.mapped()-=bytes;
                auto size_rest=state_->fits.extract(fit);size_rest.value()={rest.mapped(),rest.key()};
                state_->free.insert(std::move(rest));state_->fits.insert(std::move(size_rest));
            }
            block->interval.key()=offset;block->state=state_;state_->used+=bytes;return block;
        }
    }
    void* data() const {return state_->storage.data();}
    const void* allocation_id() const noexcept {return state_.get();}
    std::size_t bytes() const {return state_->storage.size();}
    std::size_t used_bytes() const {std::lock_guard lock(state_->mutex);return state_->used;}
    std::size_t largest_free_bytes() const {
        std::lock_guard lock(state_->mutex);
        return state_->fits.empty()?0:state_->fits.rbegin()->first;
    }
    mfq::cuda::Tensor tensor() const {
        auto storage=std::make_shared<mfq::cuda::TensorStorage>();
        const auto device=mfq::cuda::Device{mfq::cuda::DeviceType::cpu,0};
        storage->owner=state_;storage->base=data();storage->bytes=bytes();storage->device=device;storage->pinned=true;
        const std::array<std::int64_t,1> shape{std::int64_t(bytes())};
        return mfq::cuda::Tensor(storage,mfq::cuda::make_contiguous_view(data(),shape,mfq::cuda::ScalarType::uint8,device));
    }
private:
    std::shared_ptr<State> state_;
};
#endif
