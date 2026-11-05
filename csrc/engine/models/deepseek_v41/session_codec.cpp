#include "engine/models/deepseek_v41/session_codec.h"
#include "engine/models/session.h"
#include "engine/models/deepseek_v41/snapshot.h"
#include "engine/models/deepseek_v41/hash.h"

#include <algorithm>
#include <array>

namespace tyloi::models::deepseek_v41 {
namespace {
constexpr std::array<uint8_t,8> magic{'M','F','Q','C','U','D','3',0};
struct Header { uint32_t start,count,layers,flags; };
Header header(CudaPagedReader& reader) {
    std::array<uint8_t,8> value{};reader.raw(value.data(),value.size(),"magic");
    if(value!=magic || reader.scalar<uint32_t>("version")!=1)
        throw CudaSessionStateError("unsupported DeepSeek-V4.1 prefix payload");
    Header result{reader.scalar<uint32_t>("start"),reader.scalar<uint32_t>("count"),
        reader.scalar<uint32_t>("layers"),reader.scalar<uint32_t>("flags")};
    if(!result.count || !result.layers || result.layers>4096 || result.flags>3 ||
            ((result.flags&2) && !(result.flags&1)) ||
            uint64_t(result.start)+result.count>INT64_MAX)
        throw CudaSessionStateError("invalid DeepSeek-V4.1 prefix header");
    return result;
}
void optional(CudaPagedWriter& writer,const ::tyloi::Tensor* value) {
    writer.scalar<uint32_t>(value && value->defined()?1:0);
    if(value && value->defined()) {
        const auto* dense=dynamic_cast<const ::tyloi::DenseTensor*>(value);
        if(!dense)throw std::runtime_error("inconsistent DeepSeek-V4.1 compression checkpoint");
        writer.tensor(*dense);
    }
}
::tyloi::DenseTensor optional(CudaPagedReader& reader,int expected_device) {
    const auto present=reader.scalar<uint32_t>("tensor presence");
    if(present>1)throw CudaSessionStateError("invalid DeepSeek-V4.1 tensor presence");
    if(!present)return {};
    auto [value,device]=reader.tensor();
    if(device!=expected_device)throw CudaSessionStateError("inconsistent DeepSeek-V4.1 prefix device");
    return value;
}
bool shape(const ::tyloi::DenseTensor& value,int64_t rows,int64_t width,::tyloi::ScalarType dtype) {
    return value.defined() && value.dim()==3 && value.size(0)==1 &&
        value.size(1)==rows && value.size(2)==width && value.scalar_type()==dtype;
}
struct Layout {
    int64_t ratio,kv_width,index_width;
    int device;
    bool operator==(const Layout&) const=default;
};
}

CudaPagedPayload encode_block(
        const TextSessionState& state,size_t block_size,size_t index) {
    const auto* snapshot=std::get_if<tyloi::models::deepseek_v41::Snapshot>(&state.payload);
    if(!block_size || !snapshot || snapshot->blocks.empty() || state.cache_pos<=0 ||
            state.tokens.size()!=size_t(state.cache_pos) || state.cache_pos>UINT32_MAX ||
            index>(state.tokens.size()-1)/block_size || !snapshot->engram ||
            snapshot->engram->batch!=1 ||
            snapshot->engram->position!=state.cache_pos || state.decode_position_delta!=0)
        throw std::runtime_error("invalid DeepSeek-V4.1 paged session");
    const auto start=index*block_size,count=std::min(block_size,state.tokens.size()-start),end=start+count;
    const bool exact=end==state.tokens.size();
    CudaPagedWriter writer;writer.raw(magic.data(),magic.size());writer.scalar<uint32_t>(1);
    for(const auto value:{start,count,snapshot->blocks.size(),size_t(exact?(state.mtp?3:1):0)})
        writer.scalar<uint32_t>(uint32_t(value));
    for(const auto& layer:snapshot->blocks) {
        const auto* compressed=dynamic_cast<const ::tyloi::DenseTensor*>(layer.compressed_kv.get());
        const auto* index=dynamic_cast<const ::tyloi::DenseTensor*>(layer.index_k.get());
        const int64_t ratio=compressed && compressed->defined()?(layer.partial_kv && layer.partial_kv->defined()?layer.partial_kv->size(1):1):0;
        const int64_t kv_width=ratio?compressed->size(2):0;
        const int64_t index_width=ratio && index && index->defined()?index->size(2):0;
        if((layer.compressed_kv && !compressed) || (layer.index_k && !index) ||
                !layer.local_kv || !layer.local_kv->is_cuda() || (ratio && (!index_width ||
                layer.compressed_length!=state.cache_pos/ratio ||
                layer.partial_length!=state.cache_pos%ratio ||
                !shape(*compressed,layer.compressed_length,kv_width,::tyloi::kFloat16) ||
                !shape(*index,layer.compressed_length,index_width,::tyloi::kFloat16))))
            throw std::runtime_error("inconsistent DeepSeek-V4.1 compression checkpoint");
        writer.scalar<int64_t>(ratio);writer.scalar<int64_t>(kv_width);writer.scalar<int64_t>(index_width);
        writer.scalar<int32_t>(layer.local_kv->get_device());
        const auto range=[&](const ::tyloi::DenseTensor* value) {
            if(ratio && end/ratio>start/ratio) {
                auto part=value->narrow(1,int64_t(start)/ratio,int64_t(end/ratio-start/ratio));
                optional(writer,&part);
            } else optional(writer,nullptr);
        };
        range(compressed);range(index);
        optional(writer,exact?layer.local_kv.get():nullptr);
        optional(writer,exact?layer.partial_kv.get():nullptr);
        optional(writer,exact?layer.partial_score.get():nullptr);
    }
    if(exact) {
        const auto& context=snapshot->engram->context;
        writer.scalar<uint32_t>(uint32_t(context.size()));
        for(auto token:context)writer.scalar<int64_t>(token);
        if(state.mtp) {
            const auto& mtp=*state.mtp;
            const auto* rings=std::get_if<tyloi::models::deepseek_v41::MtpSnapshot>(&mtp.payload);
            const auto* hidden=dynamic_cast<const ::tyloi::DenseTensor*>(mtp.last_target_hidden.get());
            if(mtp.cache_pos!=state.cache_pos || !rings || rings->rings.empty() ||
                    !hidden || !hidden->defined())
                throw std::runtime_error("incompatible DeepSeek-V4.1 DSpark checkpoint");
            writer.tensor(*hidden);
            writer.scalar<uint32_t>(uint32_t(rings->rings.size()));
            for(const auto& ring:rings->rings) {
                const auto* dense=dynamic_cast<const ::tyloi::DenseTensor*>(ring.get());
                if(!dense || !dense->defined())
                    throw std::runtime_error("incompatible DeepSeek-V4.1 DSpark checkpoint");
                writer.tensor(*dense);
            }
        }
    }
    return std::move(writer).finish();
}

bool exact(const CudaPagedPayload& payload) {
    if(!payload)throw CudaSessionStateError("null DeepSeek-V4.1 prefix payload");
    CudaPagedReader reader(*payload);return (header(reader).flags&1)!=0;
}
bool checkpoint_needs_update(const CudaPagedPayload& payload,const TextSessionState& state) {
    if(!payload)return true;
    CudaPagedReader reader(*payload);const auto flags=header(reader).flags;
    return !(flags&1) || (state.mtp && !(flags&2));
}

TextSessionState decode_session(
        const std::vector<CudaPagedPayload>& payloads,const std::vector<int64_t>& tokens,size_t block_size) {
    if(!block_size || tokens.empty() || tokens.size()>UINT32_MAX ||
            payloads.size()!=(tokens.size()-1)/block_size+1)
        throw CudaSessionStateError("invalid DeepSeek-V4.1 prefix chain length");
    TextSessionState state;state.tokens=tokens;state.cache_pos=int64_t(tokens.size());
    std::vector<Layout> layouts;
    std::vector<tyloi::models::deepseek_v41::BlockSnapshot> layers;
    std::vector<std::array<std::vector<::tyloi::DenseTensor>,2>> fragments;
    auto hash=std::make_shared<tyloi::models::deepseek_v41::EngramHashSnapshot>();
    hash->batch=1;hash->position=state.cache_pos;
    std::vector<std::pair<::tyloi::DenseTensor,int>> predictor;
    for(size_t i=0;i<payloads.size();++i) {
        if(!payloads[i])throw CudaSessionStateError("null DeepSeek-V4.1 prefix block");
        CudaPagedReader reader(*payloads[i]);const auto h=header(reader);
        const auto end=uint64_t(h.start)+h.count;
        const bool exact=(h.flags&1)!=0;
        if(h.start!=i*block_size || h.count!=std::min(block_size,tokens.size()-h.start) ||
                (i && h.layers!=layers.size()) || (i+1==payloads.size() && !exact))
            throw CudaSessionStateError("incompatible DeepSeek-V4.1 prefix chain");
        if(!i){layers.resize(h.layers);fragments.resize(h.layers);}
        for(size_t j=0;j<layers.size();++j) {
            Layout layout{reader.scalar<int64_t>("ratio"),reader.scalar<int64_t>("KV width"),
                reader.scalar<int64_t>("Indexer width"),reader.scalar<int32_t>("device")};
            if(layout.device<0 || layout.ratio<0 || layout.ratio>65536 ||
                    (layout.ratio?layout.kv_width<=0 || layout.index_width<=0:
                     layout.kv_width!=0 || layout.index_width!=0) || (i && layouts[j]!=layout))
                throw CudaSessionStateError("inconsistent DeepSeek-V4.1 prefix layout");
            if(!i)layouts.push_back(layout);
            const auto rows=layout.ratio?int64_t(end/layout.ratio-h.start/layout.ratio):0;
            for(size_t field=0;field<2;++field) {
                auto value=optional(reader,layout.device);
                if(value.defined()!=(rows>0) || (rows &&
                        !shape(value,rows,field?layout.index_width:layout.kv_width,::tyloi::kFloat16)))
                    throw CudaSessionStateError("invalid DeepSeek-V4.1 compressed prefix rows");
                if(rows)fragments[j][field].push_back(std::move(value));
            }
            auto local=optional(reader,layout.device),partial=optional(reader,layout.device),score=optional(reader,layout.device);
            if(local.defined()!=exact || (exact && (local.dim()!=3 || local.size(0)!=1 ||
                    local.size(1)<=0 || local.size(2)<=0 || local.scalar_type()!=::tyloi::kFloat16)) ||
                    partial.defined()!=(exact && layout.ratio>1) || score.defined()!=partial.defined() ||
                    (partial.defined() && (!shape(partial,layout.ratio,layout.kv_width,::tyloi::kFloat32) ||
                                          !shape(score,layout.ratio,layout.kv_width,::tyloi::kFloat32))))
                throw CudaSessionStateError("invalid DeepSeek-V4.1 exact checkpoint tensors");
            if(i+1==payloads.size()) {
                layers[j].local_kv=std::make_shared<::tyloi::DenseTensor>(std::move(local));
                if(partial.defined())layers[j].partial_kv=std::make_shared<::tyloi::DenseTensor>(std::move(partial));
                if(score.defined())layers[j].partial_score=std::make_shared<::tyloi::DenseTensor>(std::move(score));
            }
        }
        if(exact) {
            const auto size=reader.scalar<uint32_t>("Engram history size");
            if(size>65536)throw CudaSessionStateError("invalid DeepSeek-V4.1 Engram history size");
            hash->context.clear();
            for(uint32_t j=0;j<size;++j)hash->context.push_back(reader.scalar<int64_t>("Engram token"));
            predictor.clear();
            if(h.flags&2) {
                predictor.push_back(reader.tensor());
                const auto count=reader.scalar<uint32_t>("DSpark stages");
                if(!count || count>64)throw CudaSessionStateError("invalid DeepSeek-V4.1 DSpark stage count");
                for(uint32_t j=0;j<count;++j)predictor.push_back(reader.tensor());
            }
        }
        reader.expect_end();
    }
    // Validate the complete chain before allocating device state.
    for(size_t i=0;i<predictor.size();++i) {
        const auto& value=predictor[i].first;
        if(value.dim()!=3 || value.size(0)!=1 || (!i && value.size(1)!=1) || value.scalar_type()!=::tyloi::kFloat16)
            throw CudaSessionStateError("invalid DeepSeek-V4.1 DSpark tensor");
    }
    for(size_t j=0;j<layers.size();++j) {
        auto& layer=layers[j];const auto& layout=layouts[j];
        const auto device=::tyloi::Device{::tyloi::DeviceType::cuda,layout.device};
        layer.local_kv=layer.local_kv->relocated(device);
        if(layout.ratio) {
            layer.compressed_length=state.cache_pos/layout.ratio;layer.partial_length=state.cache_pos%layout.ratio;
            std::array<std::shared_ptr<const ::tyloi::Tensor>*,2> destination{&layer.compressed_kv,&layer.index_k};
            for(size_t field=0;field<2;++field) {
                const auto width=field?layout.index_width:layout.kv_width;
                auto& parts=fragments[j][field];
                auto value=parts.empty()?::tyloi::cuda::empty({1,0,width},::tyloi::TensorOptions().dtype(::tyloi::kFloat16).device(::tyloi::kCPU)):
                    parts.size()==1?parts.front(): ::tyloi::cuda::cat(parts,1);
                if(!shape(value,layer.compressed_length,width,::tyloi::kFloat16))
                    throw CudaSessionStateError("incomplete DeepSeek-V4.1 compressed prefix");
                *destination[field]=std::make_shared<::tyloi::DenseTensor>(value.to(device));
            }
            if(layout.ratio>1) {
                layer.partial_kv=layer.partial_kv->relocated(device);layer.partial_score=layer.partial_score->relocated(device);
            }
        }
        for(const auto* tensor:{&layer.local_kv,&layer.compressed_kv,&layer.index_k,&layer.partial_kv,&layer.partial_score})
            if(*tensor)state.bytes+=session_tensor_bytes(**tensor);
    }
    if(!predictor.empty()) {
        auto& mtp=state.mtp.emplace();mtp.cache_pos=state.cache_pos;
        auto& rings=mtp.payload.emplace<tyloi::models::deepseek_v41::MtpSnapshot>().rings;
        for(size_t i=0;i<predictor.size();++i) {
            auto& [value,device]=predictor[i];
            auto target=value.to(::tyloi::Device{::tyloi::DeviceType::cuda,device});mtp.bytes+=session_tensor_bytes(target);
            if(!i)mtp.last_target_hidden=std::make_shared<::tyloi::DenseTensor>(std::move(target));
            else rings.push_back(std::make_shared<::tyloi::DenseTensor>(std::move(target)));
        }
        state.bytes+=mtp.bytes;
    }
    state.bytes+=hash->context.size()*sizeof(int64_t);
    tyloi::models::deepseek_v41::Snapshot snapshot;
    snapshot.blocks=std::move(layers);snapshot.engram=std::move(hash);
    state.payload=std::move(snapshot);return state;
}

} // namespace tyloi::models::deepseek_v41
