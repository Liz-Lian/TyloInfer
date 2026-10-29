from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

import pytest

from tyloinfer.formats.gguf import gguf_is_table_component, parse_gguf_shard_path
from tyloinfer.server.services.model_deletion import model_deletion_plan
from tyloinfer.server.services.model_memory import checkpoint_cache_profile
from tyloinfer.server.services.checkpoint_analysis import analyze_checkpoint
from tyloinfer.server.runtime.prefix_cache_text import decode_prefix_cache_tokens
from tyloinfer.server.runtime.qsa_kv_policy import qsa_index_requirement
from tyloinfer.server.state.catalog import ModelCatalog


def header(path: Path, tensors: int, name: bytes = b'per_layer_token_embd.weight') -> None:
    path.write_bytes(struct.pack('<4sIQQ', b'GGUF', 3, tensors, 0) + struct.pack('<Q', len(name)) + name)


def test_gguf_component_and_shard_headers(tmp_path):
    path = tmp_path / 'table.gguf'
    header(path, 1)
    assert gguf_is_table_component(path)
    header(path, 3)
    assert not gguf_is_table_component(path)
    assert parse_gguf_shard_path(tmp_path / 'model-00002-of-00003.gguf') == (tmp_path / 'model.gguf', 2, 3)
    with pytest.raises(ValueError):
        parse_gguf_shard_path(tmp_path / 'model-00000-of-00003.gguf')


def test_gguf_prefix_text_decodes_embedded_byte_vocabulary(tmp_path):
    from gguf import GGUFWriter
    path = tmp_path / 'tokenizer.gguf'
    writer = GGUFWriter(str(path), 'qwen35')
    writer.add_tokenizer_model('gpt2')
    writer.add_token_list(['hello', 'Ġworld', 'Ċ', '<|im_end|>'])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert decode_prefix_cache_tokens(path, [0, 1, 2, 3]) == 'hello world\n<|im_end|>'
    content = bytearray(path.read_bytes())
    struct.pack_into('<Q', content, 8, 1)
    content.extend(struct.pack('<Q', 6) + b'weight' + struct.pack('<IQIQ', 1, 64, 42, 0))
    path.write_bytes(content)
    assert decode_prefix_cache_tokens(path, [0, 1]) == 'hello world'


def test_gguf_catalog_groups_shards_and_shared_table(tmp_path):
    async def check():
        models, shared = tmp_path / 'models', tmp_path / 'shared'
        models.mkdir(); shared.mkdir()
        table = shared / 'table-00002-of-00002.gguf'
        first, second = (models / f'checkpoint-0000{index}-of-00002.gguf' for index in (1, 2))
        header(first, 4); header(table, 1); second.symlink_to(table)
        calls = []

        def inspect(path):
            calls.append(path)
            return {'architecture': 'qwen4exp', 'config': {
                'model_type': 'qwen4_exp', 'max_position_embeddings': 16384, 'num_hidden_layers': 1,
                'layer_types': ['full_attention'], 'num_key_value_heads': 2, 'head_dim': 256,
                'indexer_head_dim': 128, 'indexer_compress_ratio': 4, 'indexer_budget': 2048,
            }, 'tensors': {
                'model.token_embedding.weight': {'dtype': 'GGUF_Q8_0', 'nbytes': 500, 'shape': [32, 32]},
                'model.block.0.mlp.experts.gate.weight': {'dtype': 'GGUF_IQ3_XXS', 'nbytes': 2000, 'shape': [4, 256, 256]},
                'model.block.0.position_embedding.ngram.shard.0.weight': {'dtype': 'GGUF_IQ4_NL', 'nbytes': 9000, 'shape': [128, 160]},
            }}

        catalog = ModelCatalog([models, shared], cache_seconds=0, native_inspector=inspect)
        listing = await catalog.list(refresh=True)
        assert len(listing.data) == 1
        resource = listing.data[0]
        assert resource.format == 'gguf' and resource.complete and resource.loadable
        assert resource.architecture == 'qwen4_exp'
        assert resource.shard_count == 2 and resource.ssd_ple_bytes == 9000
        assert resource.estimated_resident_weight_bytes == 2500
        artifact = await catalog.get(resource.id)
        assert artifact.routed_expert_bytes == 2000
        assert checkpoint_cache_profile(artifact).components[0].group == 'QSA'
        assert qsa_index_requirement(artifact, 8192, 8 << 20) == 128 * ((4 + 5) * 2 + 2048 * 4) + 2
        analysis = analyze_checkpoint(artifact)
        assert analysis.format == 'gguf' and analysis.expert_count == 4
        assert len(analysis.experts) == 4
        assert {expert.format for expert in analysis.experts} == {'IQ3_XXS'}
        assert analysis.experts[0].parameters == 256 * 256
        assert analysis.experts[0].stored_bytes == 500
        assert catalog._immediate_model_count(shared) == 0
        assert len(model_deletion_plan(artifact).files) == 2
        assert model_deletion_plan(artifact).total_bytes == first.lstat().st_size + second.lstat().st_size
        await catalog.list(refresh=True)
        assert calls == [first]
        second.unlink()
        incomplete = (await catalog.list(refresh=True)).data
        assert len(incomplete) == 1 and incomplete[0].missing_shards == 1
        assert not incomplete[0].complete and not incomplete[0].loadable

    asyncio.run(check())


def test_mlx_native_metadata_reports_logical_shapes_and_refreshes(tmp_path):
    async def check():
        model = tmp_path / 'mlx-model'
        model.mkdir()
        config = {'model_type': 'qwen4_exp', 'num_hidden_layers': 1, 'num_experts': 4}
        (model / 'config.json').write_text(json.dumps(config))
        tensors = {'model.layers.0.mlp.experts.gate_proj.weight': {
            'dtype': 'U32', 'shape': [4, 64, 32], 'data_offsets': [0, 4],
        }}
        shard = model / 'model.safetensors'
        encoded = json.dumps(tensors).encode()
        shard.write_bytes(struct.pack('<Q', len(encoded)) + encoded + b'\0' * 4)
        calls = []

        def inspect(path):
            calls.append(path)
            return {'architecture': 'qwen4_exp', 'config': config, 'tensors': {
                'model.block.0.mlp.experts.gate_up.weight': {'dtype': 'MLX-AFFINE',
                    'nbytes': 16384, 'shape': [4, 128, 256]},
                'model.block.0.position_embedding.ngram.shard.0.weight': {
                    'dtype': 'MLX-AFFINE', 'nbytes': 9000, 'shape': [512, 160]},
            }}

        catalog = ModelCatalog([tmp_path], cache_seconds=0, native_inspector=inspect)
        resources = (await catalog.list(refresh=True)).data
        assert len(resources) == 1 and resources[0].loadable
        artifact = await catalog.get(resources[0].id)
        assert artifact.resource.estimated_resident_weight_bytes == 16384
        assert artifact.resource.ssd_ple_bytes == 9000
        analysis = analyze_checkpoint(artifact)
        assert analysis.parameters == 4 * 128 * 256 + 512 * 160
        assert len(analysis.experts) == 8
        assert {item.projection for item in analysis.experts} == {'gate', 'up'}
        assert all(item.bpw == 1 for item in analysis.experts)
        await catalog.list(refresh=True)
        assert calls == [model]
        (model / 'config.json').write_text(json.dumps({**config, 'num_experts_per_tok': 1}))
        await catalog.list(refresh=True)
        assert calls == [model, model]

        unavailable = ModelCatalog([tmp_path], cache_seconds=0)
        resource = (await unavailable.list()).data[0]
        assert resource.complete and not resource.loadable
        assert 'inspection is unavailable' in resource.error

    asyncio.run(check())
