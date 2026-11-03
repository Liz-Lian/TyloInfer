void qwen35_nextn_companion_source(const std::filesystem::path& directory) {
    const auto fixture=[&](const char* filename,bool head,int64_t hidden=32,bool complete=true,int64_t nextn=1) {
        const auto path=directory/filename;
        auto tensors=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({1<<20,nullptr,false}),ggml_free);
        auto file=std::unique_ptr<gguf_context,decltype(&gguf_free)>(gguf_init_empty(),gguf_free);
        gguf_set_val_str(file.get(),"general.architecture","qwen35");
        for(const auto& [suffix,value]:std::map<std::string,int64_t>{
                {"embedding_length",hidden},{"feed_forward_length",2*hidden},{"block_count",head?2:1},
                {"attention.head_count",2},{"attention.head_count_kv",1},{"attention.key_length",hidden/2},
                {"rope.dimension_count",hidden/2},{"context_length",128}})
            gguf_set_val_u32(file.get(),("qwen35."+suffix).c_str(),uint32_t(value));
        gguf_set_val_f32(file.get(),"qwen35.rope.freq_base",10000.f);
        gguf_set_val_f32(file.get(),"qwen35.attention.layer_norm_rms_epsilon",1e-6f);
        if(head)gguf_set_val_i64(file.get(),"qwen35.nextn_predict_layers",nextn);
        const auto add=[&](const std::string& name,int64_t columns,int64_t rows=0,ggml_type type=GGML_TYPE_F32) {
            auto* tensor=rows?ggml_new_tensor_2d(tensors.get(),type,columns,rows):
                ggml_new_tensor_1d(tensors.get(),type,columns);
            ggml_set_name(tensor,name.c_str());
            if(type==GGML_TYPE_F32)std::fill_n(static_cast<float*>(tensor->data),ggml_nelements(tensor),head?2.f:1.f);
            else std::memset(tensor->data,21,ggml_nbytes(tensor));
            gguf_add_tensor(file.get(),tensor);
        };
        add("token_embd.weight",hidden,7);add("output_norm.weight",hidden);add("blk.0.attn_norm.weight",hidden);
        if(head) {
            add("blk.1.nextn.eh_proj.weight",2*hidden,hidden,GGML_TYPE_Q8_0);
            if(complete)add("blk.1.nextn.enorm.weight",hidden);
            add("blk.1.nextn.hnorm.weight",hidden);add("blk.1.nextn.shared_head_norm.weight",hidden);
            add("blk.1.attn_norm.weight",hidden);add("blk.1.ffn_norm.weight",hidden);
            add("blk.1.attn_q.weight",hidden,2*hidden);add("blk.1.attn_k.weight",hidden,hidden/2);
            add("blk.1.attn_v.weight",hidden,hidden/2);add("blk.1.attn_output.weight",hidden,hidden);
            add("blk.1.attn_q_norm.weight",hidden/2);add("blk.1.attn_k_norm.weight",hidden/2);
            add("blk.1.ffn_gate.weight",hidden,2*hidden);add("blk.1.ffn_up.weight",hidden,2*hidden);
            add("blk.1.ffn_down.weight",2*hidden,hidden);
        }
        require(gguf_write_to_file(file.get(),path.string().c_str(),false),"Qwen nextn fixture write failed");
        return path;
    };
    const auto main=fixture("qwen35-backbone.gguf",false),head=fixture("qwen35-with-nextn.gguf",true);
    const mfq::GgufModelSource plain(main),standalone(head),combined(main,{head});
    const auto first_shard=directory/"qwen35-nextn-00001-of-00002.gguf";
    const auto last_shard=directory/"qwen35-nextn-00002-of-00002.gguf";
    {
        ggml_context* raw=nullptr;
        auto original=std::unique_ptr<gguf_context,decltype(&gguf_free)>(gguf_init_from_file(head.string().c_str(),{false,&raw}),gguf_free);
        auto tensors=std::unique_ptr<ggml_context,decltype(&ggml_free)>(raw,ggml_free);
        require(bool(original),"Qwen nextn shard fixture could not be read");
        auto first=std::unique_ptr<gguf_context,decltype(&gguf_free)>(gguf_init_empty(),gguf_free);
        auto last=std::unique_ptr<gguf_context,decltype(&gguf_free)>(gguf_init_empty(),gguf_free);
        gguf_set_kv(first.get(),original.get());
        gguf_set_val_u16(first.get(),"split.count",2);gguf_set_val_u16(first.get(),"split.no",0);
        gguf_set_val_u16(last.get(),"split.count",2);gguf_set_val_u16(last.get(),"split.no",1);
        for(int64_t i=0;i<gguf_get_n_tensors(original.get());++i) {
            const auto* name=gguf_get_tensor_name(original.get(),i);
            gguf_add_tensor(std::string_view(name).rfind("blk.1.",0)==0?last.get():first.get(),ggml_get_tensor(tensors.get(),name));
        }
        require(gguf_write_to_file(first.get(),first_shard.string().c_str(),false) &&
            gguf_write_to_file(last.get(),last_shard.string().c_str(),false),"Qwen nextn shard fixture write failed");
    }
    const mfq::GgufModelSource split(last_shard);
    require(nlohmann::json::parse(plain.model_config_json()).value("mtp_num_hidden_layers",0)==0,"plain Qwen GGUF gained a predictor");
    for(const auto* source:{&standalone,&combined,&split}) {
        const auto config=nlohmann::json::parse(source->model_config_json());
        require(config.at("num_hidden_layers")==1 && config.at("layer_types").size()==1 &&
            config.at("mtp_num_hidden_layers")==1 && !config.at("mtp_use_dedicated_embeddings").get<bool>(),
            "Qwen GGUF nextn changed backbone depth or failed predictor inference");
        require(source->find_tensor("predictor.fusion.weight") && source->find_tensor("predictor.embedding_norm.weight") &&
            source->find_tensor("predictor.block.0.attention.query.weight") &&
            source->find_tensor("predictor.block.0.mlp.down.weight") && !source->find_tensor("model.block.1.attention.query.weight"),
            "Qwen nextn tensor roles were not normalized");
        require(source->canonical_tensor_name("blk.1.nextn.eh_proj.weight")=="predictor.fusion.weight",
            "Qwen nextn stored tensor name was lost");
    }
    require(combined.tensors().size()==standalone.tensors().size(),"Qwen MTP companion duplicated backbone tensors");
    require(combined.read("model.token_embedding.weight")==plain.read("model.token_embedding.weight"),
        "Qwen MTP companion replaced target embeddings");
    require(combined.read("predictor.fusion.weight")==standalone.read("predictor.fusion.weight"),
        "Qwen MTP companion altered actual packed/dense tensor bytes");
    require(split.read("predictor.fusion.weight")==standalone.read("predictor.fusion.weight"),
        "Qwen nextn split metadata changed predictor bytes");
    const auto range=combined.tensor_file_range("predictor.fusion.weight");
    require(range && range->path==std::filesystem::absolute(head),"Qwen MTP companion lost its original file range");
    const auto rejects=[&](const std::filesystem::path& path,bool companion) {
        bool rejected=false;
        try {mfq::GgufModelSource source(companion?main:path,companion?std::vector<std::filesystem::path>{path}:std::vector<std::filesystem::path>{});}
        catch(const std::runtime_error&){rejected=true;}
        return rejected;
    };
    require(rejects(fixture("qwen35-bad-geometry.gguf",true,64),true),"incompatible Qwen MTP companion accepted");
    require(rejects(fixture("qwen35-incomplete-nextn.gguf",true,32,false),false),"incomplete Qwen nextn head accepted");
    require(rejects(fixture("qwen35-invalid-nextn-count.gguf",true,32,true,2),false),"invalid Qwen nextn block count accepted");
    bool duplicate=false;
    try {mfq::GgufModelSource invalid(head,{head});}catch(const std::runtime_error&){duplicate=true;}
    require(duplicate,"duplicate Qwen MTP source accepted");
    std::cout<<"Qwen GGUF native nextn and read-only head companion checks PASS\n";
}
