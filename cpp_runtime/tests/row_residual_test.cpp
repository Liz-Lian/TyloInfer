#include "tyloinfer/model_source.h"
#include "tyloinfer/gguf_model_source.h"
#include "tyloinfer/row_residual.h"
#include "row_residual_fixture.h"
#include "gguf.h"
#include "nlohmann/json.hpp"
#include <cstring>
#include <iostream>

namespace {
void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
struct Files {
    std::vector<std::filesystem::path> values;
    ~Files() {for(const auto& path:values){std::error_code error;std::filesystem::remove(path,error);}}
};
}
int main() try {
    RowResidualFixture residual(10,256);
    const auto base=std::filesystem::path(residual.path().string()+".base.gguf");
    const auto manifest=std::filesystem::path(residual.path().string()+".row-residual.json");
    const auto previous=std::filesystem::path(residual.path().string()+".previous");
    Files files{{base,manifest,previous}};
    auto* context=ggml_init({1<<20,nullptr,false});
    auto* header=gguf_init_empty();
    gguf_set_val_str(header,"general.architecture","qwen35");
    gguf_set_val_u32(header,"qwen35.embedding_length",256);
    gguf_set_val_u32(header,"qwen35.block_count",1);
    gguf_set_val_u32(header,"qwen35.attention.head_count",4);
    gguf_set_val_u32(header,"qwen35.attention.head_count_kv",2);
    gguf_set_val_u32(header,"qwen35.context_length",32768);
    auto* weight=ggml_new_tensor_2d(context,GGML_TYPE_Q3_K,256,10);
    ggml_set_name(weight,"blk.0.ffn_gate.weight");
    std::memset(weight->data,17,ggml_nbytes(weight));gguf_add_tensor(header,weight);
    gguf_write_to_file(header,base.string().c_str(),false);gguf_free(header);ggml_free(context);
    auto original=std::make_shared<mfq::GgufModelSource>(base);
    const std::string name="model.block.0.mlp.gate.weight";
    const auto frozen=original->read(name);
    nlohmann::json document{{"format","tyloinfer-row-residual-v1"},
        {"base",base.filename().string()},
        {"tensors",{{"blk.0.ffn_gate.weight",residual.path().filename().string()}}}};
    const auto write=[&]{std::ofstream output(manifest);output<<document.dump();};write();
    auto source=mfq::open_model_source(manifest);
    require(source->read(name)==frozen,"residual source changed original GGUF bytes");
    require(source->metadata()==original->metadata() && source->assets()==original->assets(),
        "residual source lost model metadata or assets");
    require(source->source_paths().size()==3,"residual source paths omitted dependencies");
    auto correction=source->row_residual(name);
    require(correction && correction->row_count()==10 && correction->columns()==256,
        "GGUF residual manifest did not attach canonical neuron rows");
    require(!source->row_residual("missing"),"nonexistent weight unexpectedly has a residual");
    bool rejected=false;
    try{correction->read_payload(correction->payload_bytes(),nullptr,1);}catch(const std::out_of_range&){rejected=true;}
    require(rejected,"residual reader accepted an unbounded read");
    // Atomic publication preserves already-open snapshots, including on Windows.
    const auto row=correction->read_row(7);
    std::filesystem::rename(residual.path(),previous);
    std::filesystem::copy_file(previous,residual.path());
    {
        std::fstream update(residual.path(),std::ios::in|std::ios::out|std::ios::binary);
        update.seekp(-1,std::ios::end);const char changed=char(row.back()^0xff);update.write(&changed,1);
    }
    require(correction->read_row(7)==row,"publishing a new residual mutated a loaded snapshot");
    mfq::RowResidualFile newer(residual.path());
    require(newer.read_row(7)!=row,"new residual reader did not see the published snapshot");
    document["tensors"][name]=residual.path().filename().string();write();
    rejected=false;try{mfq::open_model_source(manifest);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"duplicate residual alias was accepted");
    require(original->read(name)==frozen,"residual update changed the base checkpoint");
    std::cout<<"PASS: residual manifest, canonical names, immutable snapshots and bounded reads\n";
    return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
