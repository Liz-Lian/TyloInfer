#include "qsa_ram_session.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <limits>

namespace mfq::cuda::internal {
namespace tb=mfq_tensor_backend;
namespace qsa=tyloinfer::qsa;
namespace {
void require(bool condition,const char* message) {
    if(!condition)throw CudaSessionStateError(message);
}
size_t length(CudaPagedReader& r,const char* name) {
    const auto n=r.scalar<uint64_t>(name);
    require(n<=std::numeric_limits<size_t>::max(),"QSA prefix size overflow");return size_t(n);
}
std::vector<uint8_t> range(const CudaPagedPayload& p,size_t at,size_t n) {
    require(at<=p->size() && n<=p->size()-at,"invalid QSA prefix extent");
    return {p->begin()+at,p->begin()+at+n};
}
void write_tail(CudaPagedWriter& w,const tb::Tensor& t) {
    if(!t.defined()){w.scalar<uint64_t>(0);return;}
    const auto cpu=t.to(tb::kCPU,tb::kFloat32).contiguous();
    const auto bytes=size_t(cpu.numel())*sizeof(float);w.scalar<uint64_t>(bytes);w.raw(cpu.data_ptr(),bytes);
}
tb::Tensor read_tail(CudaPagedReader& r,const std::vector<int64_t>& shape,bool discard) {
    size_t expected=sizeof(float);for(auto dim:shape)expected*=size_t(dim);
    const auto bytes=length(r,"QSA raw tail size");
    require(bytes==expected && bytes<=r.remaining(),"incompatible QSA raw tail size");
    if(discard){r.skip(bytes);return {};}
    auto t=tb::empty(shape,tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
    r.raw(t.data_ptr(),bytes,"QSA raw tail");return t;
}
}
void write_ram_qsa_page(CudaPagedWriter& w,const QsaTqSession& s,size_t start,size_t count,bool exact) {
    require(s.tokens>0 && s.kv && s.index.size()==s.kv->blocks.size() &&
        s.kv->blocks.size()==size_t(s.tokens/4) && start<=size_t(s.tokens) && count<=size_t(s.tokens)-start,
        "invalid compressed QSA prefix boundary");
    for(int n:{s.heads,s.kv_heads,s.width,s.padded,s.index_width,s.ram_bits})w.scalar<uint32_t>(uint32_t(n));
    const auto first=start/4,last=(start+count)/4;
    w.scalar<uint32_t>(uint32_t(last-first));
    for(auto id=first;id<last;++id) {
        const auto& b=s.kv->blocks[id];const auto image=b.read();
        w.scalar<uint32_t>(uint32_t(b.tier));
        for(auto n:b.bytes)w.scalar<uint64_t>(n);
        w.scalar<uint64_t>(b.access);
        for(unsigned n:{image.key_bits,image.value_bits,image.key_state_bits,image.value_state_bits})w.scalar<uint32_t>(n);
        require(bool(s.index[id]),"missing QSA index archive");const auto index=s.index[id]->read();
        w.scalar<uint64_t>(image.key.size());w.scalar<uint64_t>(image.value.size());w.scalar<uint64_t>(index->size());
        w.raw(image.key.data(),image.key.size());w.raw(image.value.data(),image.value.size());w.raw(index->data(),index->size());
    }
    if(exact && s.tokens%4) {write_tail(w,s.tail_keys);write_tail(w,s.tail_values);write_tail(w,s.tail_index);}
}

void read_ram_qsa_page(CudaPagedReader& r,const CudaPagedPayload& payload,
        const std::shared_ptr<const cache::PagedPrefixReadLease>& lease,
        size_t start,size_t count,bool exact,std::shared_ptr<QsaTqSession>& state,bool bind) {
    std::array<uint32_t,6> geometry{};for(auto& n:geometry)n=r.scalar<uint32_t>("QSA geometry");
    const auto [heads,kv_heads,width,padded,iw,bits]=geometry;
    require(heads && kv_heads && heads%kv_heads==0 && width && width<=INT_MAX && padded>=width && padded%256==0 &&
        iw && iw<=INT_MAX && (bits==16 || bits==8 || bits==4) && uint64_t(kv_heads)*padded<=INT_MAX,
        "invalid compressed QSA prefix geometry");
    if(!state) {
        require(!bind,"missing live QSA snapshot");state=std::make_shared<QsaTqSession>();
        state->heads=heads;state->kv_heads=kv_heads;state->width=width;state->padded=padded;
        state->index_width=iw;state->ram_bits=bits;state->kv=std::make_shared<qsa::RamKvSnapshot>();
    }
    auto& s=*state;
    require(s.heads==heads && s.kv_heads==kv_heads && s.width==width && s.padded==padded && s.index_width==iw && s.ram_bits==bits &&
        s.kv,"inconsistent compressed QSA prefix layers");
    const size_t first=start/4,last=(start+count)/4,n=r.scalar<uint32_t>("QSA block count");
    require(n==last-first && n<=r.remaining()/76,"invalid compressed QSA prefix block count");
    if(!bind)require(s.kv->blocks.size()==first && s.index.size()==first,"incomplete compressed QSA prefix chain");
    else require(s.kv->blocks.size()>=last && s.index.size()>=last,"short live QSA prefix snapshot");
    const auto rows=uint64_t(kv_heads)*padded*4/256;
    const auto fp16=rows*512,affine=((rows*3+3)&~uint64_t(3))+rows*256;
    const auto tq4=((rows*2+(rows+1)/2+(rows*3+7)/8+3)&~uint64_t(3))+rows*128;
    const size_t index_bytes=uint64_t(iw)*10+4;
    for(auto id=first;id<last;++id) {
        qsa::RamKvSnapshotBlock b;const auto tier=r.scalar<uint32_t>("QSA RAM tier");
        require(tier<=3,"invalid QSA RAM tier");b.tier=qsa::RamKvTier(tier);
        for(auto& x:b.bytes)x=length(r,"QSA tier bytes");
        require(b.bytes==std::array<size_t,3>{size_t(2*fp16),size_t(2*affine),size_t(2*tq4)},
            "invalid QSA RAM pressure sizes");
        b.access=r.scalar<uint64_t>("QSA block access");
        auto backing=std::make_shared<qsa::RamKvBacking>();
        const auto kb=r.scalar<uint32_t>("QSA key bits"),vb=r.scalar<uint32_t>("QSA value bits");
        const auto ks=r.scalar<uint32_t>("QSA key states"),vs=r.scalar<uint32_t>("QSA value states");
        const auto tq_bits=[](unsigned bits){return (bits>=1 && bits<=8) || bits==255;};
        require(((kb==0 && vb==0) || (kb==qsa::affine8_tag && vb==qsa::affine8_tag) || (tq_bits(kb) && tq_bits(vb))) &&
            (ks==14 || ks==16) && (vs==14 || vs==16),"invalid QSA encoding flags");
        backing->key_bits=uint8_t(kb);backing->value_bits=uint8_t(vb);
        backing->key_state_bits=uint8_t(ks);backing->value_state_bits=uint8_t(vs);
        backing->key_bytes=length(r,"QSA key size");backing->value_bytes=length(r,"QSA value size");
        const auto ilen=length(r,"QSA index size");
        const auto key_bytes=backing->key_bytes,value_bytes=backing->value_bytes;
        require(key_bytes && value_bytes && key_bytes<=fp16 && value_bytes<=fp16 && ilen==index_bytes &&
            key_bytes<=r.remaining() && value_bytes<=r.remaining()-key_bytes && ilen<=r.remaining()-key_bytes-value_bytes,
            "invalid compressed QSA prefix extent");
        backing->tier=kb==0 && vb==0?qsa::RamKvTier::Fp16:
            kb==qsa::affine8_tag && vb==qsa::affine8_tag?qsa::RamKvTier::Affine8:qsa::RamKvTier::Tq4;
        require(tier==3 || unsigned(backing->tier)==tier,"QSA payload precision does not match its RAM tier");
        const auto at=payload->size()-r.remaining(),total=key_bytes+value_bytes;
        auto archive=std::make_shared<QsaIndexArchive>();
        if(lease) {
            backing->read=[lease,at,total]{return lease->read_range(at,total);};
            archive->reader=[lease,at,total,ilen]{return lease->read_range(at+total,ilen);};
            b.backing=backing;
        } else {
            auto data=std::make_shared<const std::vector<uint8_t>>(range(payload,at,total));
            auto image=std::make_shared<qsa::RamKvImage>();image->owner=data;
            image->key=data->data();image->value=data->data()+key_bytes;image->key_bytes=key_bytes;image->value_bytes=value_bytes;
            image->key_bits=backing->key_bits;image->value_bits=backing->value_bits;
            image->key_state_bits=backing->key_state_bits;image->value_state_bits=backing->value_state_bits;
            b.image=image;archive->resident=std::make_shared<const std::vector<uint8_t>>(range(payload,at+total,ilen));
        }
        r.skip(total+ilen);
        if(bind) {
            require(bool(lease),"QSA live prefix binding needs a file lease");
            const auto& old=s.kv->blocks[id];const auto* image=old.image.get();const auto* prev=old.backing.get();
            const bool same=(image?image->key_bytes:prev?prev->key_bytes:0)==key_bytes &&
                (image?image->value_bytes:prev?prev->value_bytes:0)==value_bytes &&
                (image?image->key_bits:prev?prev->key_bits:0)==kb &&
                (image?image->value_bits:prev?prev->value_bits:0)==vb &&
                (image?image->key_state_bits:prev?prev->key_state_bits:0)==ks &&
                (image?image->value_state_bits:prev?prev->value_state_bits:0)==vs;
            if(same)s.kv->bind_prefix(id,backing);
            // The exact index archive is independent of KV's current tier.
            require(bool(s.index[id]),"missing live QSA index archive");s.index[id]->bind(archive->reader);
        } else {s.kv->blocks.push_back(std::move(b));s.index.push_back(std::move(archive));}
    }
    if(!bind)s.tokens=int64_t(start+count);
    s.durable_prefix|=bool(lease);s.borrow_prefix|=bool(lease);
    if(exact && (start+count)%4) {
        const auto rem=int64_t((start+count)%4);
        auto k=read_tail(r,{1,rem,s.kv_heads,s.width},bind);
        auto v=read_tail(r,{1,rem,s.kv_heads,s.width},bind);
        auto i=read_tail(r,{1,rem,s.index_width},bind);
        if(!bind){s.tail_keys=std::move(k);s.tail_values=std::move(v);s.tail_index=std::move(i);}
    }
}
} // namespace mfq::cuda::internal
