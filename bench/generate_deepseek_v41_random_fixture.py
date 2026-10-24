#!/usr/bin/env python3
"""Generate a graph-faithful, reduced-depth DeepSeek-V4.1 HF fixture.

The fixture keeps the production tensor widths, expert counts, native MXFP4 /
MXFP8 encodings, Engram table geometry, Vision block geometry, and all three
DSpark stages.  Only repeated backbone and Vision depth is reduced.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    fill: str = "random"

    @property
    def nbytes(self) -> int:
        item_size = {
            "BF16": 2,
            "F32": 4,
            "I8": 1,
            "F8_E4M3": 1,
            "F8_E8M0": 1,
        }[self.dtype]
        result = item_size
        for dimension in self.shape:
            result *= dimension
        return result


class Fixture:
    def __init__(self) -> None:
        self.shards: dict[str, list[Tensor]] = {}

    def add(
        self,
        shard: str,
        name: str,
        dtype: str,
        shape: Iterable[int],
        fill: str = "random",
    ) -> None:
        self.shards.setdefault(shard, []).append(
            Tensor(name, dtype, tuple(int(value) for value in shape), fill)
        )


def config() -> dict[str, object]:
    return {
        "architectures": ["DeepseekV41ForConditionalGeneration"],
        "model_type": "deepseek_v41",
        "eos_token_id": 1,
        "image_token_id": 129264,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": 129280,
            "hidden_size": 5120,
            "moe_intermediate_size": 2304,
            "num_hidden_layers": 3,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "hidden_act": "silu",
            "attention_bias": False,
            "rms_norm_eps": 1e-20,
            "max_position_embeddings": 1048576,
            "rope_theta": 10000,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
            },
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "sliding_window": 128,
            # Preserve all three production attention regimes while dropping
            # only repeated decoder depth.  DSpark ratios must remain zero.
            "compress_ratios": [0, 2, 1, 0, 0, 0],
            "compress_rope_theta": 160000,
            "kv_source_layer_ids": [1, 2],
            "index_source_layer_ids": [1, 2],
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 512,
            "candidate_source_layer_id": 2,
            "candidate_topk_blocks": 2048,
            "candidate_block_size": 8,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "engram_layer_ids": [1],
            "engram_num_embeddings": [384006168],
            "engram_max_ngram_size": 4,
            "engram_vocab_size": 16000000,
            "engram_n_heads": 8,
            "engram_head_dim": 256,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 99092,
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [0, 1, 2],
            "dspark_markov_rank": 256,
            "dspark_n_routed_experts": 128,
            "dspark_num_experts_per_tok": 3,
        },
        "vision_config": {
            "model_type": "deepseek_v41_vision",
            "num_hidden_layers": 1,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 2816,
            "patch_size": 14,
            "rope_theta": 10000,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": None,
        },
    }


def add_mxfp8(
    fixture: Fixture,
    shard: str,
    name: str,
    output: int,
    input_: int,
) -> None:
    if input_ % 32:
        raise ValueError(f"MXFP8 input is not divisible by 32: {name}")
    fixture.add(shard, name, "F8_E4M3", (output, input_))
    fixture.add(
        shard,
        name.removesuffix(".weight") + ".scale",
        "F8_E8M0",
        ((output + 31) // 32, input_ // 32),
        "scale",
    )


def add_mxfp4_expert(
    fixture: Fixture,
    shard: str,
    prefix: str,
    projection: int,
    output: int,
    input_: int,
) -> None:
    if input_ % 32:
        raise ValueError(f"MXFP4 input is not divisible by 32: {prefix}")
    name = f"{prefix}.w{projection}.weight"
    fixture.add(shard, name, "I8", (output, input_ // 2))
    fixture.add(
        shard,
        f"{prefix}.w{projection}.scale",
        "F8_E8M0",
        (output, input_ // 32),
        "scale",
    )


def add_mhc(fixture: Fixture, shard: str, prefix: str) -> None:
    fixture.add(shard, f"{prefix}_fn", "BF16", (24, 4 * 5120), "zero")
    fixture.add(shard, f"{prefix}_base", "F32", (24,), "zero")
    fixture.add(shard, f"{prefix}_scale", "F32", (3,), "zero")


def add_attention(
    fixture: Fixture,
    shard: str,
    prefix: str,
    *,
    ratio: int,
    source: bool,
    index_source: bool,
) -> None:
    hidden = 5120
    head_dim = 512
    add_mxfp8(fixture, shard, f"{prefix}.attn.wq_a.weight", 1280, hidden)
    fixture.add(shard, f"{prefix}.attn.q_norm.weight", "BF16", (1280,), "one")
    add_mxfp8(fixture, shard, f"{prefix}.attn.wq_b.weight", 64 * head_dim, 1280)
    add_mxfp8(fixture, shard, f"{prefix}.attn.wkv.weight", head_dim, hidden)
    fixture.add(shard, f"{prefix}.attn.kv_norm.weight", "BF16", (head_dim,), "one")
    fixture.add(shard, f"{prefix}.attn.attn_sink", "F32", (64,), "zero")
    add_mxfp8(fixture, shard, f"{prefix}.attn.wo_a.weight", 8 * 1024, 4096)
    add_mxfp8(fixture, shard, f"{prefix}.attn.wo_b.weight", hidden, 8 * 1024)
    if source:
        fixture.add(
            shard,
            f"{prefix}.attn.compressor.wkv.weight",
            "BF16",
            (head_dim, hidden),
        )
        if ratio > 1:
            fixture.add(
                shard,
                f"{prefix}.attn.compressor.wgate.weight",
                "BF16",
                (head_dim, hidden),
            )
        fixture.add(
            shard,
            f"{prefix}.attn.compressor.norm.weight",
            "BF16",
            (head_dim,),
            "one",
        )
        fixture.add(
            shard,
            f"{prefix}.attn.indexer.wk.weight",
            "BF16",
            (128, head_dim),
        )
        fixture.add(
            shard,
            f"{prefix}.attn.indexer.k_norm.weight",
            "BF16",
            (128,),
            "one",
        )
    if index_source:
        add_mxfp8(
            fixture,
            shard,
            f"{prefix}.attn.indexer.wq_b.weight",
            32 * 128,
            1280,
        )
        fixture.add(
            shard,
            f"{prefix}.attn.indexer.weights_proj.weight",
            "BF16",
            (32, hidden),
        )


def add_moe(
    fixture: Fixture,
    dense_shard: str,
    expert_shard: str,
    prefix: str,
    experts: int,
) -> None:
    hidden = 5120
    intermediate = 2304
    fixture.add(
        dense_shard,
        f"{prefix}.ffn.gate.weight",
        "BF16",
        (experts, hidden),
    )
    fixture.add(
        dense_shard,
        f"{prefix}.ffn.gate.bias",
        "F32",
        (experts,),
        "zero",
    )
    fixture.add(
        dense_shard,
        f"{prefix}.ffn.gate.bias_vl",
        "F32",
        (experts,),
        "zero",
    )
    add_mxfp8(
        fixture,
        dense_shard,
        f"{prefix}.ffn.shared_experts.w1.weight",
        intermediate,
        hidden,
    )
    add_mxfp8(
        fixture,
        dense_shard,
        f"{prefix}.ffn.shared_experts.w2.weight",
        hidden,
        intermediate,
    )
    add_mxfp8(
        fixture,
        dense_shard,
        f"{prefix}.ffn.shared_experts.w3.weight",
        intermediate,
        hidden,
    )
    for expert in range(experts):
        expert_prefix = f"{prefix}.ffn.experts.{expert}"
        add_mxfp4_expert(
            fixture,
            expert_shard,
            expert_prefix,
            1,
            intermediate,
            hidden,
        )
        add_mxfp4_expert(
            fixture,
            expert_shard,
            expert_prefix,
            2,
            hidden,
            intermediate,
        )
        add_mxfp4_expert(
            fixture,
            expert_shard,
            expert_prefix,
            3,
            intermediate,
            hidden,
        )


def add_block(
    fixture: Fixture,
    dense_shard: str,
    expert_shard: str,
    prefix: str,
    *,
    experts: int,
    ratio: int = 0,
    source: bool = False,
    index_source: bool = False,
) -> None:
    fixture.add(dense_shard, f"{prefix}.attn_norm.weight", "BF16", (5120,), "one")
    fixture.add(dense_shard, f"{prefix}.ffn_norm.weight", "BF16", (5120,), "one")
    add_mhc(fixture, dense_shard, f"{prefix}.hc_attn")
    add_mhc(fixture, dense_shard, f"{prefix}.hc_ffn")
    add_attention(
        fixture,
        dense_shard,
        prefix,
        ratio=ratio,
        source=source,
        index_source=index_source,
    )
    add_moe(fixture, dense_shard, expert_shard, prefix, experts)


def build_fixture() -> Fixture:
    fixture = Fixture()
    dense = "model-00001-dense.safetensors"
    fixture.add(dense, "embed.weight", "BF16", (129280, 5120))
    fixture.add(dense, "norm.weight", "BF16", (5120,), "one")
    fixture.add(dense, "head.weight", "BF16", (129280, 5120))

    ratios = (0, 2, 1)
    for layer, ratio in enumerate(ratios):
        add_block(
            fixture,
            dense,
            f"model-backbone-experts-{layer:02d}.safetensors",
            f"layers.{layer}",
            experts=384,
            ratio=ratio,
            source=layer in (1, 2),
            index_source=layer in (1, 2),
        )

    engram = "model-engram.safetensors"
    fixture.add(engram, "layers.1.engram.embed.weight", "F8_E4M3", (384006168, 256))
    fixture.add(
        engram,
        "layers.1.engram.embed.scale",
        "F8_E8M0",
        (384006168, 8),
        "scale",
    )
    fixture.add(dense, "layers.1.engram.q_weight", "BF16", (4, 5120))
    fixture.add(dense, "layers.1.engram.k_weight", "BF16", (4, 5120))
    add_mxfp8(fixture, dense, "layers.1.engram.wkv.weight", 5 * 5120, 24 * 256)

    for stage in range(3):
        add_block(
            fixture,
            dense,
            f"model-dspark-experts-{stage:02d}.safetensors",
            f"mtp.{stage}",
            experts=128,
        )
    fixture.add(dense, "mtp.0.main_norm.weight", "BF16", (5120,), "one")
    add_mxfp8(fixture, dense, "mtp.0.main_proj.weight", 5120, 3 * 5120)
    fixture.add(dense, "mtp.2.norm.weight", "BF16", (5120,), "one")
    fixture.add(dense, "mtp.2.markov_head.embed.weight", "BF16", (129280, 256))
    fixture.add(dense, "mtp.2.markov_head.head.weight", "BF16", (129280, 256))
    fixture.add(
        dense,
        "mtp.2.confidence_head.proj.weight",
        "BF16",
        (1, 5120 + 256),
    )

    fixture.add(dense, "vision.patch_embed.proj.weight", "BF16", (1024, 3 * 14 * 14))
    fixture.add(dense, "vision.patch_embed.proj.bias", "BF16", (1024,), "zero")
    fixture.add(dense, "vision.norm.weight", "BF16", (1024,), "one")
    fixture.add(dense, "aligner.w1.weight", "BF16", (5120, 1024 * 3 * 3))
    fixture.add(dense, "aligner.w1.bias", "BF16", (5120,), "zero")
    fixture.add(dense, "aligner.w2.weight", "BF16", (5120, 5120))
    fixture.add(dense, "aligner.w2.bias", "BF16", (5120,), "zero")
    fixture.add(dense, "image_start", "BF16", (5120,))
    fixture.add(dense, "image_newline", "BF16", (5120,))
    fixture.add(dense, "image_end", "BF16", (5120,))
    prefix = "vision.blocks.0"
    fixture.add(dense, f"{prefix}.norm1.weight", "BF16", (1024,), "one")
    fixture.add(dense, f"{prefix}.norm2.weight", "BF16", (1024,), "one")
    fixture.add(dense, f"{prefix}.attn.wqkv.weight", "BF16", (3072, 1024))
    fixture.add(dense, f"{prefix}.attn.wqkv.bias", "BF16", (3072,), "zero")
    fixture.add(dense, f"{prefix}.attn.wo.weight", "BF16", (1024, 1024))
    fixture.add(dense, f"{prefix}.attn.wo.bias", "BF16", (1024,), "zero")
    fixture.add(dense, f"{prefix}.mlp.w1.weight", "BF16", (2 * 2816, 1024))
    fixture.add(dense, f"{prefix}.mlp.w2.weight", "BF16", (1024, 2816))
    return fixture


def pattern(dtype: str, fill: str, rng: np.random.Generator) -> bytes:
    elements = 4 * 1024 * 1024
    if dtype == "F8_E8M0":
        return bytes([119]) * elements
    if dtype in {"I8", "F8_E4M3"}:
        if dtype == "I8":
            values = rng.integers(0, 256, elements, dtype=np.uint8)
        else:
            magnitude = rng.integers(0, 64, elements, dtype=np.uint8)
            sign = rng.integers(0, 2, elements, dtype=np.uint8) << 7
            values = magnitude | sign
        return values.tobytes()
    if dtype == "BF16":
        if fill == "zero":
            return b"\0\0" * elements
        if fill == "one":
            return struct.pack("<H", 0x3F80) * elements
        # Small finite BF16 values: random sign, exponent around 2^-8, and
        # random mantissa.  This is cheap to generate and avoids NaNs/Infs.
        exponent = rng.integers(117, 122, elements, dtype=np.uint16) << 7
        mantissa = rng.integers(0, 128, elements, dtype=np.uint16)
        sign = rng.integers(0, 2, elements, dtype=np.uint16) << 15
        return (sign | exponent | mantissa).astype("<u2", copy=False).tobytes()
    if dtype == "F32":
        value = 1.0 if fill == "one" else 0.0
        return np.full(elements, value, dtype="<f4").tobytes()
    raise ValueError(dtype)


def write_repeated(stream, data: bytes, count: int) -> None:
    view = memoryview(data)
    while count >= len(view):
        stream.write(view)
        count -= len(view)
    if count:
        stream.write(view[:count])


def write_shard(
    output: Path,
    tensors: list[Tensor],
    seed: int,
) -> None:
    offset = 0
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for tensor in tensors:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    rng = np.random.default_rng(seed)
    patterns: dict[tuple[str, str], bytes] = {}
    with output.open("wb", buffering=16 * 1024 * 1024) as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for tensor in tensors:
            key = (tensor.dtype, tensor.fill)
            if key not in patterns:
                patterns[key] = pattern(tensor.dtype, tensor.fill, rng)
            write_repeated(stream, patterns[key], tensor.nbytes)


def copy_tokenizer(source: Path, output: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
    ):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, output / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer-source", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output}")
    output.mkdir(parents=True)
    fixture = build_fixture()
    total = sum(
        tensor.nbytes
        for tensors in fixture.shards.values()
        for tensor in tensors
    )
    print(
        f"fixture={output} shards={len(fixture.shards)} "
        f"tensors={sum(map(len, fixture.shards.values()))} "
        f"payload_gib={total / 2**30:.2f}",
        flush=True,
    )
    weight_map: dict[str, str] = {}
    shard_count = len(fixture.shards)
    for index, (name, tensors) in enumerate(fixture.shards.items(), 1):
        size = sum(tensor.nbytes for tensor in tensors)
        print(
            f"[{index}/{shard_count}] writing {name}: "
            f"{len(tensors)} tensors, {size / 2**30:.2f} GiB",
            flush=True,
        )
        write_shard(output / name, tensors, 41000 + index)
        weight_map.update((tensor.name, name) for tensor in tensors)
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "config.json").write_text(
        json.dumps(config(), indent=2) + "\n",
        encoding="utf-8",
    )
    copy_tokenizer(args.tokenizer_source.expanduser().resolve(), output)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
