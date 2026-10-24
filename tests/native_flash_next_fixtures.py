"""Backend-neutral synthetic Flash-Next model fixtures."""
import json
import numpy as np


def synthetic_tq_experts(weights, *, experts=None, intermediate=None, prefix='', seed=2342):
    from bench.tq_smoke_fixture import matrix
    from tyloinfer.formats.tq import TQExpertTensor
    result = dict(weights)
    rng = np.random.default_rng(seed)
    for name, value in list(weights.items()):
        if not name.startswith(prefix) or '.experts.' not in name:
            continue
        parts = [(name, value)]
        if name.endswith('.gate_up.weight'):
            gate, up = np.split(value, 2, axis=1)
            parts = [(name.replace('gate_up.weight', role + '.weight'), part)
                for role, part in (('gate', gate), ('up', up))]
            del result[name]
        for key, part in parts:
            count, rows, width = part.shape
            if intermediate is not None:
                if key.endswith('.down.weight'):
                    width = intermediate
                else:
                    rows = intermediate
            result[key] = TQExpertTensor(tuple(matrix(rows, width, rng, True)
                for _ in range(experts or count)), width)
    return result


def glm_model_fixture(attention_types, sparse_ffn):
    from tyloinfer.architectures.flash_next import Glm5NextConfig
    rng=np.random.default_rng(5303)
    h,d,heads,rank,nope,vwidth,index=32,32,2,16,16,16,128
    layers=len(attention_types)
    def rand(shape,scale=.05): return rng.normal(scale=scale,size=shape).astype(np.float32)
    c=dict(model_type="glm5_next_text",vocab_size=32,hidden_size=h,intermediate_size=32,
        num_hidden_layers=layers,max_position_embeddings=32,rms_norm_eps=1e-5,
        layer_types=list(attention_types),mlp_layer_types=["sparse" if sparse_ffn else "dense"]*layers,
        mhc=True,mla_use_nope=True,hc_mult=2,hc_eps=1e-6,hc_sinkhorn_iters=5,
        linear_attn_config=dict(num_heads=heads,head_dim=d,short_conv_kernel_size=4,gate_lower_bound=-5,
            kda_layers=[i for i,t in enumerate(attention_types) if t=="linear_attention"],
            full_attn_layers=[i for i,t in enumerate(attention_types) if t!="linear_attention"]),
        q_lora_rank=rank,kv_lora_rank=rank,num_attention_heads=heads,qk_nope_head_dim=nope,qk_rope_head_dim=0,
        v_head_dim=vwidth,index_n_heads=heads,index_head_dim=index,index_topk=4,index_kpool=2,
        index_kpool_always_select_tail=True,indexer_types=["full"]*layers,n_routed_experts=3,
        num_experts_per_tok=2,moe_intermediate_size=16,n_shared_experts=1,routed_scaling_factor=2.5,
        swiglu_limit=10.0,scoring_func="sigmoid",norm_topk_prob=True,num_nextn_predict_layers=0,
        tie_word_embeddings=False,eos_token_id=[0])
    outer=dict(model_type="glm5_next",text_config=c)
    Glm5NextConfig.from_hf_config(outer)
    w={"model.token_embedding.weight":rand((32,h),.2),"model.output.weight":rand((32,h)),
       "model.output_norm.weight":np.ones(h,np.float32)}
    def dense_ffn(p,inner):
        w[p+".gate.weight"]=rand((inner,h));w[p+".up.weight"]=rand((inner,h));w[p+".down.weight"]=rand((h,inner))
    for i,kind in enumerate(attention_types):
        p=f"model.block.{i}"
        for site in ("attention","mlp"):
            prefix=p+f".{site}.mhc.pre"
            w[prefix+".function"]=rand((8,2*h),.02)
            w[prefix+".base"]=rand((8,),.02)
            w[prefix+".scale"]=np.array([.2,.3,.4],np.float32)
            w[p+f".{site}.norm.weight"]=np.ones(h,np.float32)
        if kind=="linear_attention":
            a=p+".linear_attention"
            for proj in ("query","key","value"):
                w[a+f".{proj}.weight"]=rand((heads*d,h))
                conv=np.zeros((heads*d,1,4),np.float32);conv[:,0,-1]=.8;conv[:,0,-2]=.1
                w[a+f".{proj}_conv.weight"]=conv
            for key,shape in {"forget_a":(d,h),"forget_b":(heads*d,d),"beta":(heads,h),
                              "gate_a":(d,h),"gate_b":(heads*d,d),"output":(h,heads*d)}.items():
                w[a+f".{key}.weight"]=rand(shape)
            w[a+".dt_bias"]=rand((heads*d,),.02);w[a+".a"]=rand((heads,),.02)
            w[a+".output_norm.weight"]=np.ones(d,np.float32)
        else:
            a=p+".attention"
            for key,shape in {"query_a":(rank,h),"key_value_a":(rank,h),"query_b":(heads*nope,rank),
                              "output":(h,heads*vwidth),"indexer.query":(heads*index,rank),
                              "indexer.key":(index,h),"indexer.score":(heads,h),
                              "latent.query_embedding":(heads,rank,nope),
                              "latent.output_unembedding":(heads,vwidth,rank)}.items():
                w[a+f".{key}.weight"]=rand(shape)
            for key,size in [("query_a_norm",rank),("key_value_a_norm",rank),("indexer.key_norm",index)]:
                w[a+f".{key}.weight"]=np.ones(size,np.float32)
            w[a+".indexer.key_norm.bias"]=np.zeros(index,np.float32)
            w[a+".indexer.pool.gate"]=rand((index,h))
            w[a+".indexer.pool.position"]=rand((2,index))
        if sparse_ffn:
            w[p+".mlp.experts.gate_up.weight"]=rand((3,32,h))
            w[p+".mlp.experts.down.weight"]=rand((3,h,16))
            w[p+".mlp.router.weight"]=rand((3,h));w[p+".mlp.router.bias"]=rand((3,),.02)
            dense_ffn(p+".mlp.shared_expert",16)
        else: dense_ffn(p+".mlp",32)
    return outer,w


def qwen_model_fixture(interval=2,silu_gate=False,ple=True,hidden=16):
    from tyloinfer.architectures.flash_next import Qwen4ExpConfig
    rng=np.random.default_rng(3805);rand=lambda shape,s=.05:rng.normal(scale=s,size=shape).astype(np.float32)
    h,streams,d,heads,kv,ih=hidden,2,32,2,1,8
    kinds=["full_attention" if (i+1)%interval==0 else "linear_attention" for i in range(2)]
    ple_layers=[i+1 for i,kind in enumerate(kinds) if kind=="linear_attention"] if ple else []
    c=dict(model_type="qwen4_exp_text",vocab_size=32,hidden_size=h,num_hidden_layers=2,
        max_position_embeddings=32,num_attention_heads=heads,num_key_value_heads=kv,head_dim=d,
        partial_rotary_factor=.25,rope_parameters=dict(rope_theta=1e7,mrope_section=[2,1,1],mrope_interleaved=silu_gate),
        rms_norm_eps=1e-6,layer_types=kinds,full_attention_interval=interval,hc_count=streams,hc_lowrank=4,
        linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=d,linear_value_head_dim=d,
        linear_conv_kernel_dim=4,num_experts=3,num_experts_per_tok=2,moe_intermediate_size=16,
        shared_expert_intermediate_size=16,norm_topk_prob=silu_gate,output_gate_type="silu" if silu_gate else "sigmoid",
        hidden_act="silu",attention_bias=False,indexer_n_heads=ih,indexer_kv_heads=1,indexer_head_dim=d,
        indexer_budget=4,indexer_compress_ratio=2,ple_layer_ids=ple_layers,ple_embed_dim=h,ple_conv_kernel_size=3,
        ngram_size=3,heads_per_ngram=2,ngram_vocab_size_base=8,split_ngram_parts=4,
        make_ngram_vocab_size_divisible_by=1,seed=3805,mtp_num_hidden_layers=0,tie_word_embeddings=False,eos_token_id=3)
    outer=dict(model_type="qwen4_exp",text_config=c);Qwen4ExpConfig.from_hf_config(outer)
    w={"model.token_embedding.weight":rand((32,h),.2),"model.output.weight":rand((32,h))}
    def gr(p,combine=True):
        w[p+".norm.weight"]=rand((streams*h,),.02);w[p+".down.weight"]=rand((4,streams*h));w[p+".up.weight"]=rand((streams*h,4))
        if combine:w[p.removesuffix(".pre")+".post.inject.weight"]=rand((streams,streams*h))
    gr("model.mhc.pre",False)
    for i,kind in enumerate(kinds):
        p=f"model.block.{i}"
        gr(p+".attention.mhc.pre");gr(p+".mlp.mhc.pre")
        m=p+".mlp"
        w[m+".experts.gate_up.weight"]=rand((3,32,h));w[m+".experts.down.weight"]=rand((3,h,16));w[m+".router.weight"]=rand((3,h))
        for key,shape in {"gate":(16,h),"up":(16,h),"down":(h,16),"router":(1,h)}.items():w[m+f".shared_expert.{key}.weight"]=rand(shape)
        if kind=="linear_attention":
            a=p+".linear_attention"
            for key,shape in {"qkv":(4*d,h),"gate":(2*d,h),"alpha":(2,h),"beta":(2,h),"output":(h,2*d)}.items():w[a+f".{key}.weight"]=rand(shape)
            conv=np.zeros((4*d,1,4),np.float32);conv[:,0,-1]=.8;conv[:,0,-2]=.1
            w[a+".conv.weight"]=conv;w[a+".dt_bias"]=rand((2,));w[a+".a"]=rand((2,));w[a+".norm.weight"]=np.ones(d,np.float32)
        else:
            a=p+".attention"
            for key,shape in {"query":(heads*2*d,h),"key":(kv*d,h),"value":(kv*d,h),"output":(h,heads*d),"indexer.query_key":((ih+1)*d,h)}.items():w[a+f".{key}.weight"]=rand(shape)
            for key in ["query_norm","key_norm","indexer.query_norm","indexer.key_norm"]:w[a+f".{key}.weight"]=rand((d,),.02)
        if i+1 in ple_layers:
            a=p+".position_embedding"
            for shard in range(4):w[a+f".ngram.shard.{shard}.weight"]=rand((8,h//4),.2)
            w[a+".ngram.layer_multipliers"]=np.array([-7046029254386353131,6364136223846793005,-4658895280553007687],np.int64)
            w[a+".ngram.head_offsets"]=np.array([0,8,16,24],np.int64);w[a+".ngram.head_vocab_sizes"]=np.array([5,7,5,7],np.int64)
            w[a+".key.weight"]=rand((streams*h,h));w[a+".value.weight"]=rand((h,h))
            for key in ["key_norm","query_norm","conv_norm"]:w[a+f".{key}.weight"]=rand((streams*h,),.02)
            w[a+".conv.weight"]=rand((3,streams*h)).T[:,None,:].copy()
    return outer,w


def independent_qwen_attention(config, weights):
    result = dict(weights)
    c = config['text_config']
    kw = c['linear_num_key_heads'] * c['linear_key_head_dim']
    heads, width = c['num_attention_heads'], c['head_dim']
    for name, value in weights.items():
        if name.endswith('.linear_attention.qkv.weight'):
            prefix = name.removesuffix('.qkv.weight')
            for key, matrix in zip(('query','key','value'), np.split(value, [kw, 2*kw], axis=0)):
                result[prefix+f'.{key}.weight'] = matrix.copy()
            del result[name]
        elif name.endswith('.attention.query.weight'):
            pair = value.reshape(heads, 2*width, -1)
            result[name] = pair[:, :width].reshape(heads*width, -1).copy()
            result[name.removesuffix('.query.weight')+'.gate.weight'] = pair[:, width:].reshape(heads*width, -1).copy()
    return result


def write_qwen_fixture(path,config,weights,predictor=False):
    from tyloinfer.formats.io import save
    from tyloinfer.formats.header import FileHeader
    from tyloinfer.formats.assets import MODEL_CONFIG_ASSET,MODEL_GRAPH_ASSET
    graph=dict(schema_version=1,architecture="qwen4_exp",canonical_naming=dict(namespace="mfq.tensor",version=1,component_roots=["model"]),
        topology=dict(text_layers=2),graph=dict(kind="causal_lm",backbone="qwen4_exp"),
        components=[dict(kind="text",tensor_root="model",implementation="qwen4_exp",policy="decoder")],capabilities=["text"])
    if predictor:
        graph["topology"]["predictor_layers"]=config["text_config"]["mtp_num_hidden_layers"]
        graph["components"].append(dict(kind="predictor",tensor_root="predictor",implementation="next_token_prediction",policy="optional"))
        graph["canonical_naming"]["component_roots"].append("predictor")
        graph["capabilities"].append("mtp")
    save(path,FileHeader(version=2,model_arch="qwen4_exp"),{**weights,MODEL_CONFIG_ASSET:json.dumps(config).encode(),MODEL_GRAPH_ASSET:json.dumps(graph).encode()})


def mtp_fixture(family,layers=2,*,hidden=16):
    rng=np.random.default_rng(3811 if family=="qwen" else 5311)
    rand=lambda shape:rng.normal(scale=.05,size=shape).astype(np.float32)
    if family=="qwen":
        config,w=qwen_model_fixture(2,True,True,hidden=hidden)
        _,head=qwen_model_fixture(1,True,False,hidden=hidden)
        h=config["text_config"]["hidden_size"];streams=config["text_config"]["hc_count"]
        config["text_config"]["mtp_num_hidden_layers"]=layers
        w["predictor.embedding_norm.weight"]=rand((h,));w["predictor.hidden_norm.weight"]=rand((streams*h,))
        w["predictor.fusion.embedding.weight"]=rand((h,h));w["predictor.fusion.hidden.weight"]=rand((h,h))
        for name,value in head.items():
            if name.startswith("model.mhc.pre."):w[name.replace("model.","predictor.",1)]=value.copy()
    else:
        config,w=glm_model_fixture(("linear_attention","deepseek_sparse_attention"),True)
        _,head=glm_model_fixture(("deepseek_sparse_attention",)*2,True)
        h=config["text_config"]["hidden_size"]
        config["text_config"]["num_nextn_predict_layers"]=layers
        for name in ("embedding_norm","hidden_norm","output_norm"):w[f"predictor.{name}.weight"]=1+rand((h,))
        w["predictor.fusion.weight"]=rand((h,2*h))
    for layer in range(layers):
        prefix=f"model.block.{layer}."
        for name,value in head.items():
            if name.startswith(prefix):w[name.replace("model.","predictor.",1)]=value.copy()
    return config,w
