from __future__ import annotations

import asyncio
import errno
import json
import os
import struct
from pathlib import Path

import pytest

from tyloinfer.formats.gguf import (
    gguf_is_table_component, gguf_part_paths, gguf_vocabulary_decoder, is_gguf_path,
    matching_gguf_shard_paths, parse_gguf_shard_path,
)
from tyloinfer.server.services.model_deletion import model_deletion_plan
from tyloinfer.server.services.model_memory import checkpoint_cache_profile
from tyloinfer.server.services.checkpoint_analysis import analyze_checkpoint
from tyloinfer.server.runtime.prefix_cache_text import decode_prefix_cache_tokens
from tyloinfer.server.runtime.qsa_kv_policy import qsa_index_requirement
from tyloinfer.server.state.catalog import ModelCatalog
from tyloinfer.server.runtime.native_metadata import NativeModelInspector


def header(path: Path, tensors: int, name: bytes = b'per_layer_token_embd.weight') -> None:
    if tensors > 1 and name == b'per_layer_token_embd.weight':
        name = b'weight'
    path.write_bytes(struct.pack('<4sIQQ', b'GGUF', 3, tensors, 0) + b''.join(
        struct.pack('<Q', len(name)) + name + struct.pack('<IQIQ', 1, 32, 0, 0) for _ in range(tensors)))


def test_gguf_component_and_shard_headers(tmp_path):
    path = tmp_path / 'table.gguf'
    header(path, 1)
    assert gguf_is_table_component(path)
    header(path, 3)
    assert not gguf_is_table_component(path)
    assert parse_gguf_shard_path(tmp_path / 'model-00002-of-00003.gguf') == (tmp_path / 'model.gguf', 2, 3)
    with pytest.raises(ValueError):
        parse_gguf_shard_path(tmp_path / 'model-00000-of-00003.gguf')


@pytest.mark.parametrize('backend', ['metal', 'cuda'])
def test_native_gguf_inspection_accepts_supported_backends(monkeypatch, tmp_path, backend):
    import subprocess
    import tyloinfer.server.runtime.native_metadata as native_metadata

    metadata = {'architecture': 'qwen4_exp', 'config': {'model_type': 'qwen4_exp'},
        'tensors': {'model.embed_tokens.weight': {'dtype': 'GGUF_IQ3_S', 'shape': [128, 256], 'nbytes': 12000}}}
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, json.dumps(metadata).encode(), b'')

    monkeypatch.setattr(native_metadata, 'native_runtime_environment', lambda *args: {})
    monkeypatch.setattr(native_metadata.subprocess, 'run', run)
    executable = tmp_path / 'mfq-runtime'
    model = tmp_path / 'model.gguf'
    assert NativeModelInspector(executable, backend)(model) == metadata
    assert calls == [[str(executable), '--model', str(model), '--inspect-model-json']]


def write_raw_parts(directory: Path, name: str, payload: bytes) -> tuple[Path, ...]:
    cuts = (0, 2, 37, len(payload) - 3, len(payload))
    paths = tuple(directory / f'{name}.gguf.part-{index:05d}-of-00004' for index in range(1, 5))
    for index, path in enumerate(paths):
        path.write_bytes(payload[cuts[index]:cuts[index + 1]])
    return paths


def test_raw_gguf_parts_cross_headers_without_reconstruction(tmp_path):
    source = tmp_path / 'source.gguf'
    header(source, 1)
    parts = write_raw_parts(tmp_path, 'model', source.read_bytes())
    for index, part in enumerate(parts, 1):
        assert is_gguf_path(part)
        assert parse_gguf_shard_path(part) == (tmp_path / 'model.gguf', index, 4)
        assert gguf_part_paths(part) == parts
        assert matching_gguf_shard_paths(part) == list(parts)
        assert gguf_is_table_component(part)
    assert not (tmp_path / 'model.gguf').exists()
    standard = tmp_path / 'model-00001-of-00002.gguf'
    header(standard, 3)
    assert matching_gguf_shard_paths(standard) == [standard]
    assert matching_gguf_shard_paths(parts[-1]) == list(parts)
    parts[-1].unlink()
    assert matching_gguf_shard_paths(parts[0]) == list(parts[:-1])
    with pytest.raises(FileNotFoundError, match='missing GGUF part'):
        gguf_is_table_component(parts[0])
    with pytest.raises(ValueError, match='invalid GGUF part'):
        gguf_part_paths(tmp_path / 'model.gguf.part-00000-of-00004')


def test_raw_gguf_empty_and_truncated_parts(tmp_path):
    source = tmp_path / 'source.gguf'
    header(source, 1)
    parts = write_raw_parts(tmp_path, 'table', source.read_bytes())
    parts[1].write_bytes(b'')
    with pytest.raises(ValueError, match='empty GGUF part'):
        gguf_is_table_component(parts[0])
    parts = write_raw_parts(tmp_path, 'table', source.read_bytes()[:-1])
    with pytest.raises(ValueError, match='truncated'):
        gguf_is_table_component(parts[-1])


@pytest.mark.parametrize('width', [0, 2, 5])
def test_raw_gguf_sixteen_parts_from_any_index(tmp_path, width):
    source = tmp_path / 'source.gguf'
    header(source, 1)
    payload = source.read_bytes()
    parts = tuple(tmp_path / f'model.gguf.part-{index:0{width}d}-of-{16:0{width}d}' for index in range(1, 17))
    for index, part in enumerate(parts):
        part.write_bytes(payload[len(payload) * index // 16:len(payload) * (index + 1) // 16])
    for part in parts:
        assert gguf_part_paths(part) == parts
        assert matching_gguf_shard_paths(part) == list(parts)
        assert gguf_is_table_component(part)


def test_raw_gguf_vocabulary_reads_across_parts(tmp_path):
    from gguf import GGUFWriter
    source = tmp_path / 'tokenizer.gguf'
    writer = GGUFWriter(str(source), 'qwen35')
    writer.add_tokenizer_model('gpt2')
    writer.add_token_list(['hello', 'Ġworld', 'Ċ', '<|im_end|>'])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    parts = write_raw_parts(tmp_path, 'raw-tokenizer', source.read_bytes())
    assert gguf_vocabulary_decoder(parts[-1]).decode([0, 1, 2, 3], skip_special_tokens=False) == 'hello world\n<|im_end|>'
    assert decode_prefix_cache_tokens(parts[-1], [0, 1]) == 'hello world'
    unchanged = (parts[0].stat().st_mtime_ns, parts[-1].stat().st_mtime_ns)
    payload = parts[2].read_bytes()
    assert b'world' in payload
    parts[2].write_bytes(payload.replace(b'world', b'there'))
    assert unchanged == (parts[0].stat().st_mtime_ns, parts[-1].stat().st_mtime_ns)
    assert decode_prefix_cache_tokens(parts[-1], [0, 1]) == 'hello there'


def test_raw_gguf_catalog_groups_resolves_and_reports_missing_parts(tmp_path):
    async def check():
        models, tables = tmp_path / 'models', tmp_path / 'tables'
        models.mkdir(); tables.mkdir()
        source = tmp_path / 'source.gguf'
        header(source, 3)
        parts = write_raw_parts(models, 'xq-model', source.read_bytes())
        header(source, 1)
        write_raw_parts(tables, 'shared-table', source.read_bytes())
        source.unlink()
        calls = []
        def inspect(path):
            calls.append(path)
            return {'architecture': 'deepseek_v41', 'config': {'model_type': 'deepseek_v41',
                'max_position_embeddings': 16384}, 'tensors': {
                'model.block.0.mlp.experts.gate.weight': {'dtype': 'GGUF_Q2_0', 'nbytes': 4096, 'shape': [4, 256, 256]},
            }}
        catalog = ModelCatalog([models, tables], cache_seconds=0, native_inspector=inspect)
        resource, = (await catalog.list(refresh=True)).data
        assert resource.name == 'xq-model' and resource.format == 'gguf'
        assert resource.complete and resource.loadable and resource.shard_count == 4
        assert resource.total_bytes == sum(path.stat().st_size for path in parts)
        artifact = await catalog.resolve_path(parts[-1])
        assert artifact.path == parts[0] and artifact.resource.id == resource.id
        assert calls == [parts[0]]
        assert catalog._immediate_model_count(models) == 1
        assert catalog._immediate_model_count(tables) == 0
        assert catalog._registration_paths(models) == [parts[0]]
        assert catalog._registration_paths(tables) == []
        assert len(catalog._browse_directories(None, models).files) == 4
        standard = models / 'xq-model-00001-of-00004.gguf'
        header(standard, 3)
        assert set(model_deletion_plan(artifact).files) == {str(path) for path in parts}
        standard.unlink()
        parts[2].unlink()
        incomplete, = (await catalog.list(refresh=True)).data
        assert not incomplete.complete and not incomplete.loadable and incomplete.missing_shards == 1
        assert calls == [parts[0]]
        parts[0].unlink()
        missing_first = await catalog.resolve_path(parts[-1])
        assert not missing_first.resource.complete and missing_first.resource.missing_shards == 2
        assert calls == [parts[0]]
    asyncio.run(check())


@pytest.mark.parametrize('suffix', ['part-00000-of-00004', 'part-00005-of-00004', 'part-00001-of-00000'])
def test_invalid_raw_gguf_part_does_not_break_catalog(tmp_path, suffix):
    async def check():
        valid = tmp_path / 'valid.gguf'
        invalid = tmp_path / ('invalid.gguf.' + suffix)
        header(valid, 3)
        header(invalid, 3)
        calls = []
        def inspect(path):
            calls.append(path)
            return {'architecture': 'qwen4exp', 'config': {}, 'tensors': {}}
        catalog = ModelCatalog([tmp_path], cache_seconds=0, native_inspector=inspect)
        resources = (await catalog.list(refresh=True)).data
        assert len(resources) == 2
        good, = (resource for resource in resources if resource.name == 'valid')
        bad, = (resource for resource in resources if resource.name != 'valid')
        assert good.loadable and good.error is None
        assert not bad.loadable and 'invalid GGUF shard number' in bad.error
        assert calls == [valid]
        artifact = await catalog.get(bad.id)
        assert model_deletion_plan(artifact).files == [str(invalid)]
    asyncio.run(check())


def test_gguf_engram_component_with_multiple_tables(tmp_path):
    path = tmp_path / 'engram.gguf'
    payload = struct.pack('<4sIQQ', b'GGUF', 3, 2, 0)
    for layer in (1, 14):
        name = f'blk.{layer}.engram_embd.weight'.encode()
        payload += struct.pack('<Q', len(name)) + name + struct.pack('<IQQIQ', 2, 264, 8, 24, 0)
    path.write_bytes(payload)
    assert gguf_is_table_component(path)
    path.write_bytes(payload[:-1])
    with pytest.raises(ValueError, match='truncated'):
        gguf_is_table_component(path)


@pytest.mark.parametrize('dtype', ['PTQ1_0', 'MXFP4', 'TQ2_T', 'TQK4', 'TQK5', 'TQK6', 'TQK7', 'TQP5', 'TQP6', 'TQP7', 'MIXED'])
def test_catalog_accepts_native_gyro_and_mixed_experts(tmp_path, dtype):
    async def check():
        path = tmp_path / 'native.gguf'
        header(path, 3)
        def inspect(_):
            return {'architecture': 'deepseek_v41', 'config': {
                'model_type': 'deepseek_v41', 'text_config': {'model_type': 'deepseek_v41_text', 'max_position_embeddings': 16384},
            }, 'tensors': {
                'model.block.0.mlp.experts.gate.weight': {'dtype': 'GGUF_' + dtype, 'nbytes': 4096, 'shape': [4, 256, 256]},
                'model.block.1.associative_memory.embedding.weight': {'dtype': 'GGUF_ENGRAM_MXFP8', 'nbytes': 2640, 'shape': [10, 256]},
            }}
        catalog = ModelCatalog([tmp_path], cache_seconds=0, native_inspector=inspect)
        resource, = (await catalog.list(refresh=True)).data
        assert resource.loadable and resource.error is None
        assert resource.estimated_resident_weight_bytes == 4096
        assert resource.ssd_ple_bytes == 2640
    asyncio.run(check())


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


@pytest.mark.parametrize('expert_format', [
    'IQ3_XXS', 'PTQ1_0', 'TQ2_T', 'TQK6', 'TQK7', 'TQP5', 'TQP6', 'TQP7', 'TQK5', 'TQK4',
])
def test_gguf_catalog_groups_shards_and_shared_table(tmp_path, expert_format):
    async def check():
        models, shared = tmp_path / 'models', tmp_path / 'shared'
        models.mkdir(); shared.mkdir()
        table = shared / 'table-00002-of-00002.gguf'
        first, second = (models / f'checkpoint-0000{index}-of-00002.gguf' for index in (1, 2))
        header(first, 4); header(table, 1)
        try:
            second.symlink_to(table)
        except OSError as error:
            if error.errno not in {errno.EPERM, errno.EACCES, errno.ENOTSUP} and getattr(error, 'winerror', None) != 1314:
                raise
            # Shared-file catalog behavior does not require symbolic links;
            # hard links work without the optional symlink privilege too.
            os.link(table, second)
        calls = []

        def inspect(path):
            calls.append(path)
            return {'architecture': 'qwen4exp', 'config': {
                'model_type': 'qwen4_exp', 'max_position_embeddings': 16384, 'num_hidden_layers': 1,
                'layer_types': ['full_attention'], 'num_key_value_heads': 2, 'head_dim': 256,
                'indexer_head_dim': 128, 'indexer_compress_ratio': 4, 'indexer_budget': 2048,
            }, 'tensors': {
                'model.token_embedding.weight': {'dtype': 'GGUF_Q8_0', 'nbytes': 500, 'shape': [32, 32]},
                'model.block.0.mlp.experts.gate.weight': {'dtype': f'GGUF_{expert_format}', 'nbytes': 2000, 'shape': [4, 256, 256]},
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
        assert {expert.format for expert in analysis.experts} == {expert_format}
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
