#pragma once
#include "mfq_tensor_backend.h"
#include "tyloinfer/qsa_ram_store.h"
#include <stdexcept>
#include <cstring>

namespace mfq::cuda {
// A serialized model worker owns updates to these small immutable archives.
// Prefix persistence can replace their backing without retaining a RAM copy.
struct QsaIndexArchive {
    std::shared_ptr<const std::vector<std::uint8_t>> resident;
    std::shared_ptr<mfq::cache::QsaKvStore> store;
    mfq::cache::QsaKvStore::BlockPtr block;
    std::shared_ptr<const tyloinfer::qsa::RamKvReadSource> source;
    std::size_t offset=0,bytes=0;
    std::shared_ptr<const std::vector<std::uint8_t>> read()const {
        if(resident)return resident;
        if(source)return std::make_shared<const std::vector<std::uint8_t>>(source->read(offset,bytes));
        if(store && block)return store->read(block);
        throw std::runtime_error("QSA index archive has no retained backing");
    }
    // Restore the active INT16 suffix directly without copying its retained
    // FP32 companion. Validate the complete archive geometry before any read.
    void read_slice_into(std::size_t at,std::uint8_t* destination,std::size_t count,std::size_t expected_bytes)const {
        if(at>expected_bytes || count>expected_bytes-at)throw std::out_of_range("QSA index slice exceeds archive");
        if(count && !destination)throw std::invalid_argument("QSA index destination is null");
        if(source && !resident) {
            if(bytes!=expected_bytes)throw std::runtime_error("Incompatible QSA snapshot index archive length");
            source->validate(offset,bytes);source->read_into(offset+at,destination,count);return;
        }
        const auto data=read();
        if(data->size()!=expected_bytes)throw std::runtime_error("Incompatible QSA snapshot index archive length");
        if(count)std::memcpy(destination,data->data()+at,count);
    }
    void bind(std::shared_ptr<const tyloinfer::qsa::RamKvReadSource> reader,std::size_t at,std::size_t count) {
        if(!reader)throw std::invalid_argument("QSA index prefix reader is empty");
        reader->validate(at,count);
        source=std::move(reader);offset=at;bytes=count;
        resident.reset();block.reset();store.reset();
    }
};
struct QsaTqSession {
    std::int64_t tokens=0;
    int heads=0,kv_heads=0,width=0,padded=0,index_width=0,ram_bits=0;
    std::shared_ptr<tyloinfer::qsa::RamKvSnapshot> kv;
    std::vector<std::shared_ptr<QsaIndexArchive>> index;
    bool durable_prefix=false,borrow_prefix=false;
    mfq_tensor_backend::Tensor tail_keys,tail_values,tail_index;
    std::size_t resident_bytes()const {
        std::size_t bytes=sizeof(*this)+index.size()*sizeof(QsaIndexArchive);
        if(kv)for(const auto& b:kv->blocks){bytes+=sizeof(b);if(b.image)bytes+=b.image->bytes();}
        for(const auto& b:index)if(b && b->resident)bytes+=b->resident->size();
        for(const auto* t:{&tail_keys,&tail_values,&tail_index})if(t->defined())bytes+=std::size_t(t->numel())*t->element_size();
        return bytes;
    }
};
} // namespace mfq::cuda
