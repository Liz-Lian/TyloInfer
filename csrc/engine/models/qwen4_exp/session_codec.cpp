#include "engine/models/qwen4_exp/session_codec.h"

#include "backend.h"

#include "engine/models/session/snapshot.h"
#include "engine/models/qwen4_exp/snapshot.h"

#include <algorithm>
#include <array>

namespace tyloi::models::qwen4_exp {
namespace {
constexpr std::array<uint8_t,8> qwen4_magic{'M','F','Q','C','U','D','2',0};
struct Qwen4PagedHeader { uint32_t start,count,layers;bool exact; };
Qwen4PagedHeader read_qwen4_header(CudaPagedReader& reader) {
    std::array<uint8_t,8> magic{};reader.raw(magic.data(),magic.size(),"magic");
    if(magic!=qwen4_magic || reader.scalar<uint32_t>("version")!=1)
        throw CudaSessionStateError("unsupported Qwen4 prefix cache payload");
    Qwen4PagedHeader header{reader.scalar<uint32_t>("start"),reader.scalar<uint32_t>("count"),
        reader.scalar<uint32_t>("layers"),reader.scalar<uint32_t>("exact boundary")!=0};
    if(!header.count || !header.layers || header.layers>4096)throw CudaSessionStateError("invalid Qwen4 prefix geometry");
    return header;
}
void write_optional(CudaPagedWriter& writer,const ::tyloi::Tensor& tensor) {
    if(tensor.defined() && !tensor.representation<::tyloi::DenseTensor>())throw CudaSessionStateError("incompatible Qwen4 session tensor");
    writer.scalar<uint32_t>(tensor.defined()?1:0);if(tensor.defined())writer.tensor(tensor);
}
::tyloi::Tensor read_optional(CudaPagedReader& reader) {
    if(!reader.scalar<uint32_t>("tensor presence"))return {};
    auto [tensor,device]=reader.tensor();
    return tensor.relocated(::tyloi::Device{::tyloi::DeviceType::cuda,device});
}
void write_recurrent(CudaPagedWriter& writer,const tyloi::models::qwen4_exp::LayerSnapshot& layer) {
    write_optional(writer,layer.convolution);write_optional(writer,layer.recurrent);write_optional(writer,layer.ple_convolution);
    writer.scalar<int64_t>(layer.ple_batch);writer.scalar<uint32_t>(uint32_t(layer.ple_tokens.size()));
    for(auto token:layer.ple_tokens)writer.scalar<int64_t>(token);
}
void read_recurrent(CudaPagedReader& reader,tyloi::models::qwen4_exp::LayerSnapshot& layer) {
    layer.convolution=read_optional(reader);layer.recurrent=read_optional(reader);layer.ple_convolution=read_optional(reader);
    layer.ple_batch=reader.scalar<int64_t>("PLE batch");
    const auto count=reader.scalar<uint32_t>("PLE history length");
    if(count>65536)throw CudaSessionStateError("invalid Qwen4 PLE history");
    for(uint32_t i=0;i<count;++i)layer.ple_tokens.push_back(reader.scalar<int64_t>("PLE token"));
}
void write_qwen4_mtp(CudaPagedWriter& writer,const MtpSessionState& state) {
    const auto* snapshot=std::get_if<tyloi::models::qwen4_exp::MtpSnapshot>(&state.payload);
    if(!snapshot)throw CudaSessionStateError("incompatible Qwen4 MTP session state");
    const auto* hidden=state.last_target_hidden.representation<::tyloi::DenseTensor>();
    if(state.last_target_hidden.defined() && !hidden)throw CudaSessionStateError("incompatible Qwen4 MTP session state");
    writer.scalar<int64_t>(state.cache_pos);
    write_optional(writer,state.last_target_hidden);
    writer.scalar<uint32_t>(uint32_t(snapshot->layers.size()));
    for(std::size_t i=0;i<snapshot->layers.size();++i) {
        const auto& layer=snapshot->layers[i];
        write_optional(writer,layer.keys);write_optional(writer,layer.values);write_optional(writer,layer.index);
        write_recurrent(writer,layer);write_optional(writer,snapshot->positions[i]);
        writer.scalar<int64_t>(snapshot->lengths[i]);
    }
}
std::size_t qwen4_layer_bytes(const tyloi::models::qwen4_exp::LayerSnapshot& layer) {
    std::size_t bytes=layer.ple_tokens.size()*sizeof(int64_t);
    for(const auto* tensor:{&layer.keys,&layer.values,&layer.index,&layer.convolution,&layer.recurrent,&layer.ple_convolution})
        if(*tensor)bytes+=session_tensor_bytes(*tensor);
    return bytes;
}
MtpSessionState read_qwen4_mtp(CudaPagedReader& reader) {
    MtpSessionState state;state.cache_pos=reader.scalar<int64_t>("MTP position");
    auto hidden=read_optional(reader);
    if(hidden)state.bytes=session_tensor_bytes(hidden);
    state.last_target_hidden=std::move(hidden);
    const auto count=reader.scalar<uint32_t>("MTP layers");
    if(!count || count>64)throw CudaSessionStateError("invalid Qwen4 MTP layers");
    auto& snapshot=state.payload.emplace<tyloi::models::qwen4_exp::MtpSnapshot>();
    for(uint32_t i=0;i<count;++i) {
        tyloi::models::qwen4_exp::LayerSnapshot layer;layer.keys=read_optional(reader);layer.values=read_optional(reader);layer.index=read_optional(reader);
        read_recurrent(reader,layer);state.bytes+=qwen4_layer_bytes(layer);snapshot.layers.push_back(std::move(layer));
        auto positions=read_optional(reader);if(positions)state.bytes+=session_tensor_bytes(positions);snapshot.positions.push_back(std::move(positions));
        snapshot.lengths.push_back(reader.scalar<int64_t>("MTP layer position"));
    }
    return state;
}
}

CudaPagedPayload encode_block(const TextSessionState& state,size_t block_size,size_t index) {
    const auto* snapshot=std::get_if<tyloi::models::qwen4_exp::Snapshot>(&state.payload);
    if(!block_size || !snapshot || snapshot->layers.empty() || state.cache_pos<=0 || state.tokens.size()!=std::size_t(state.cache_pos) ||
        index>state.tokens.size()/block_size || index*block_size>=state.tokens.size() || state.cache_pos>UINT32_MAX)
        throw std::runtime_error("invalid Qwen4 prefix block boundary");
    const auto start=index*block_size,count=std::min(block_size,state.tokens.size()-start);
    const bool exact=start+count==state.tokens.size();
    CudaPagedWriter writer;writer.raw(qwen4_magic.data(),qwen4_magic.size());writer.scalar<uint32_t>(1);
    writer.scalar<uint32_t>(uint32_t(start));writer.scalar<uint32_t>(uint32_t(count));
    writer.scalar<uint32_t>(uint32_t(snapshot->layers.size()));writer.scalar<uint32_t>(exact?1:0);
    for(const auto& layer:snapshot->layers) {
        for(const auto& tensor:{layer.keys,layer.values,layer.index}) {
            const auto* dense=tensor.representation<::tyloi::DenseTensor>();
            if(tensor.defined() && !dense)throw CudaSessionStateError("incompatible Qwen4 session tensor");
            if(dense && dense->defined()) {
                auto slice=dense->narrow(1,int64_t(start),int64_t(count));
                write_optional(writer,slice);
            } else write_optional(writer,::tyloi::Tensor{});
        }
        if(exact)write_recurrent(writer,layer);
    }
    if(exact) {
        write_optional(writer,snapshot->positions);writer.scalar<uint32_t>(state.mtp?1:0);
        if(state.mtp)write_qwen4_mtp(writer,*state.mtp);
    }
    return std::move(writer).finish();
}

bool exact(const CudaPagedPayload& payload) {
    CudaPagedReader reader(*payload);return read_qwen4_header(reader).exact;
}

TextSessionState decode_session(const std::vector<CudaPagedPayload>& payloads,
    const std::vector<int64_t>& tokens,size_t block_size) {
    TextSessionState state;state.tokens=tokens;state.cache_pos=int64_t(tokens.size());
    tyloi::models::qwen4_exp::Snapshot snapshot;
    auto& layers=snapshot.layers;
    std::vector<std::array<std::vector<::tyloi::DenseTensor>,3>> fragments;
    size_t position=0;
    for(size_t i=0;i<payloads.size();++i) {
        if(!payloads[i])throw CudaSessionStateError("null Qwen4 prefix payload");
        CudaPagedReader reader(*payloads[i]);const auto header=read_qwen4_header(reader);
        if(header.start!=position || header.count>block_size || (i+1<payloads.size() && header.count!=block_size) ||
            (i && header.layers!=layers.size()))throw CudaSessionStateError("incompatible Qwen4 prefix chain");
        if(!i){layers.resize(header.layers);fragments.resize(header.layers);}
        position+=header.count;
        for(size_t layer=0;layer<layers.size();++layer) {
            for(size_t field=0;field<3;++field) {
                auto tensor=read_optional(reader);
                if(tensor && tensor.defined()) {
                    if(tensor.dim()!=3 || tensor.size(0)!=1 || tensor.size(1)!=header.count)
                        throw CudaSessionStateError("invalid Qwen4 QSA prefix geometry");
                    fragments[layer][field].push_back(tensor.dense());
                }
            }
            if(header.exact) {
                layers[layer].convolution={};layers[layer].recurrent={};layers[layer].ple_convolution={};layers[layer].ple_tokens.clear();
                read_recurrent(reader,layers[layer]);
            }
        }
        if(header.exact) {
            snapshot.positions=read_optional(reader);
            state.mtp.reset();if(reader.scalar<uint32_t>("MTP presence"))state.mtp=read_qwen4_mtp(reader);
        }
        if(i+1==payloads.size() && !header.exact)throw CudaSessionStateError("Qwen4 prefix has no exact recurrent checkpoint");
        reader.expect_end();
    }
    if(payloads.empty() || position!=tokens.size() || !snapshot.positions || !snapshot.positions.defined() || snapshot.positions.size(-1)!=int64_t(position))
        throw CudaSessionStateError("Qwen4 prefix boundary mismatch");
    for(size_t i=0;i<layers.size();++i) {
        auto& layer=layers[i];std::array<::tyloi::Tensor*,3> fields{&layer.keys,&layer.values,&layer.index};
        for(size_t j=0;j<3;++j)if(!fragments[i][j].empty()) {
            if(fragments[i][j].size()!=payloads.size())throw CudaSessionStateError("incomplete Qwen4 QSA prefix chain");
            *fields[j]=std::make_shared<::tyloi::DenseTensor>(fragments[i][j].size()==1?fragments[i][j].front(): ::tyloi::cuda::cat(fragments[i][j],1));
        }
        state.bytes+=qwen4_layer_bytes(layer);
    }
    state.bytes+=session_tensor_bytes(snapshot.positions)+(state.mtp?state.mtp->bytes:0);
    state.payload=std::move(snapshot);return state;
}

} // namespace tyloi::models::qwen4_exp
