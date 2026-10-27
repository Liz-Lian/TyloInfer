from __future__ import annotations


def source_tensor_name(name: str) -> str:
    if name.startswith("language_model.model."):
        name = "model.language_model." + name[len("language_model.model."):]
    elif name.startswith("language_model.lm_head."):
        name = "lm_head." + name[len("language_model.lm_head."):]
    return name.replace(".mlp.switch_mlp.", ".mlp.experts.").replace(
        ".ngram_embedding.shards.", ".ngram_embedding.shard_"
    )
