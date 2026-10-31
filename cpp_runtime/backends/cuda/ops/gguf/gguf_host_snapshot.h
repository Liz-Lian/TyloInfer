#pragma once
#include "tyloinfer/model_source.h"
#include "tyloinfer/gguf_quant.h"
#include "mfq_cuda_context.h"
#include "mfq_native_tensor.h"
#include "strata/core/pinned.hpp"
#include "strata/core/pin_budget.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

// One original GGUF RAM image, using Strata's registration and resident-page
// locking directly. It can be dropped as a whole when RAM is needed for KV;
// the existing bounded/exclusive cache remains the fallback.
class GgufHostSnapshot {
public:
    struct Bank {
        std::size_t offset,bytes,expert_bytes,stride,experts;
        std::size_t expert_offset(std::size_t expert) const {return offset+expert*stride;}
    };
    struct Registration {
        std::size_t first,last;
        std::map<int,std::uint8_t*> mapped;
    };
    std::map<std::string,Bank> banks;
    std::vector<std::uint64_t> bounds{0};
    std::vector<Registration> registrations;
    std::unique_ptr<strata::core::PinnedArena> arena;
    std::size_t bytes=0,payload=0;
    ~GgufHostSnapshot() {
        if(arena){arena.reset();mfq::cuda::tensor_host_bytes.fetch_sub(bytes);}
    }
    static std::shared_ptr<GgufHostSnapshot> plan(const mfq::ModelSource& source) {
        auto result=std::make_shared<GgufHostSnapshot>();
        for(const auto& tensor:source.tensors()) {
            // These are canonical routed FFN roles, shared by GGUF model
            // architectures. A 3-D shape alone never classifies a tensor.
            bool routed=false;
            for(const auto* role:{".mlp.experts.gate.weight",".mlp.experts.up.weight",".mlp.experts.down.weight"})
                routed|=tensor.name.ends_with(role);
            if(!routed || tensor.name.starts_with("predictor.") || !mfq::gguf::is_quantized(tensor.dtype) || !tensor.shape || tensor.shape->size()!=3)continue;
            const auto type=mfq::gguf::quant_type(tensor.dtype);
            const auto& shape=*tensor.shape;
            if(shape[0]<=0 || shape[1]<=0 || shape[2]<=0 ||
                    std::uint64_t(shape[1])>SIZE_MAX/mfq::gguf::row_bytes(type,shape[2]))return {};
            const auto expert=std::size_t(shape[1])*mfq::gguf::row_bytes(type,shape[2]);
            if(std::uint64_t(shape[0])>SIZE_MAX/expert || tensor.nbytes!=std::uint64_t(shape[0])*expert ||
                    tensor.nbytes>SIZE_MAX-4095 || result->bytes>SIZE_MAX-((tensor.nbytes+4095)&~std::size_t(4095)))return {};
            if(!result->banks.emplace(tensor.name,Bank{result->bytes,std::size_t(tensor.nbytes),expert,expert,std::size_t(shape[0])}).second)return {};
            result->bytes+=(tensor.nbytes+4095)&~std::size_t(4095);
            result->payload+=tensor.nbytes;result->bounds.push_back(result->bytes);
        }
        if(result->banks.empty())return {};
        // The source stays in its canonical GGUF layout. Only this optional
        // RAM image places a routed expert's three unchanged matrices next to
        // one another, allowing a single DMA for Gate/Up/Down consumers.
        std::unordered_set<std::string> assigned;
        result->bytes=0;result->bounds={0};
        const auto aligned=[](std::size_t value,std::size_t alignment) {
            if(value>SIZE_MAX-(alignment-1))throw std::overflow_error("GGUF host layout overflow");
            return (value+alignment-1)&~(alignment-1);
        };
        std::vector<std::string> order;
        for(const auto& [name,bank]:result->banks)order.push_back(name);
        std::stable_sort(order.begin(),order.end(),[](const auto& a,const auto& b) {
            return a.ends_with(".mlp.experts.gate.weight")>b.ends_with(".mlp.experts.gate.weight");
        });
        for(const auto& name:order) {
            auto& bank=result->banks.at(name);
            if(assigned.count(name))continue;
            constexpr auto gate=".mlp.experts.gate.weight";
            std::array<std::string,3> names{name,"",""};
            bool joined=name.ends_with(gate);
            if(joined) {
                const auto stem=name.substr(0,name.size()-std::char_traits<char>::length(gate));
                names={name,stem+".mlp.experts.up.weight",stem+".mlp.experts.down.weight"};
                for(const auto& role:names) {
                    const auto found=result->banks.find(role);
                    joined&=found!=result->banks.end() && found->second.experts==bank.experts && !assigned.count(role);
                }
            }
            if(!joined)names={name,"",""};
            std::size_t stride=0;
            for(const auto& role:names)if(!role.empty()) {
                const auto image=aligned(result->banks.at(role).expert_bytes,16);
                if(stride>SIZE_MAX-image)throw std::overflow_error("GGUF expert image overflow");
                stride+=image;
            }
            // Standalone roles need no extra per-expert padding.
            if(!joined)stride=bank.expert_bytes;
            if(bank.experts>SIZE_MAX/stride)throw std::overflow_error("GGUF expert bank overflow");
            const auto size=aligned(bank.experts*stride,4096);
            if(result->bytes>SIZE_MAX-size)throw std::overflow_error("GGUF host arena overflow");
            std::size_t offset=result->bytes;
            for(const auto& role:names)if(!role.empty()) {
                auto& item=result->banks.at(role);item.offset=offset;item.stride=stride;
                offset+=aligned(item.expert_bytes,16);assigned.insert(role);
            }
            result->bytes+=size;result->bounds.push_back(result->bytes);
        }
        return result;
    }
    void load(const mfq::ModelSource& source,int device) {
        const auto started=std::chrono::steady_clock::now();
        const mfq::cuda::DeviceGuard guard(device);
        strata::core::pin_reserve_take();
        struct ReleaseReserve {~ReleaseReserve(){strata::core::pin_reserve_release_soft();strata::core::pin_reserve_release_hard();}} reserve;
        auto storage=std::make_unique<strata::core::PinnedArena>(bytes,bounds);
        if(!storage->valid())throw std::runtime_error("GGUF original host arena allocation failed");
        arena=std::move(storage);mfq::cuda::charge_tensor_host_bytes(bytes);
        // The ordinary constructor registers a prefix. Every registration is
        // tracked separately: adjacent independent registrations never merge.
        if(arena->registered_bytes==bytes && arena->slice_starts.empty())registrations.push_back({0,bytes,{}});
        else for(const auto first:arena->slice_starts) {
            const auto next=std::upper_bound(bounds.begin(),bounds.end(),first);
            if(next!=bounds.end())registrations.push_back({std::size_t(first),std::size_t(*next),{}});
        }
        prepare_device(device);
        std::vector<std::byte> staging;
        for(const auto& [name,bank]:banks) {
            const auto reader=source.tensor_reader(name);
            if(bank.stride==bank.expert_bytes) {
                reader(0,reinterpret_cast<std::byte*>(arena->data()+bank.offset),bank.bytes);continue;
            }
            const auto chunk=std::max(std::size_t(1),(std::size_t(16)<<20)/bank.expert_bytes);
            for(std::size_t first=0;first<bank.experts;first+=chunk) {
                const auto count=std::min(chunk,bank.experts-first);
                staging.resize(count*bank.expert_bytes);
                reader(first*bank.expert_bytes,staging.data(),staging.size());
                for(std::size_t expert=0;expert<count;++expert)
                    std::memcpy(arena->data()+bank.expert_offset(first+expert),
                        staging.data()+expert*bank.expert_bytes,bank.expert_bytes);
            }
        }
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::fprintf(stderr,"native_gguf_host_snapshot bytes=%zu registered_bytes=%llu locked_bytes=%llu seconds=%.3f gib_per_second=%.3f\n",
            bytes,static_cast<unsigned long long>(arena->registered_bytes),static_cast<unsigned long long>(arena->locked_bytes),
            seconds,bytes/1073741824./std::max(seconds,1e-9));
    }
    void prepare_device(int device) {
        const mfq::cuda::DeviceGuard guard(device);
        for(auto& region:registrations)if(!region.mapped.count(device)) {
            void* mapped=nullptr;
            TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,arena->data()+region.first,0));
            region.mapped.emplace(device,static_cast<std::uint8_t*>(mapped));
        }
    }
    const Registration* registration(std::size_t offset,std::size_t count) const {
        const auto next=std::upper_bound(registrations.begin(),registrations.end(),offset,
            [](auto key,const auto& region){return key<region.first;});
        if(next==registrations.begin())return nullptr;
        const auto& region=*std::prev(next);
        return offset>=region.first && offset<=region.last && count<=region.last-offset?&region:nullptr;
    }
    std::uint8_t* data(std::size_t offset) const {return arena->data()+offset;}
    std::uint8_t* mapped_data(std::size_t offset,std::size_t count,int device) const {
        const auto* region=registration(offset,count);
        if(!region)throw std::logic_error("GGUF host image is outside CUDA registration");
        return region->mapped.at(device)+offset-region->first;
    }
};
