"""TQ uses MFQ1 shards, including shared Engram dependencies."""

import asyncio

import numpy as np
import pytest

from bench.tq_random import random_tensor
from tyloinfer.formats.companions import with_engram_shards
from tyloinfer.formats.header import FileHeader
from tyloinfer.formats.io import open_mmap, save
from tyloinfer.formats.shards import format_shard_path
from tyloinfer.server.protocol.models import HubModelFile, HubSystemProfile
from tyloinfer.server.services.hub import HubCatalog, _model_variants
from tyloinfer.server.state.catalog import ModelCatalog
from tyloinfer.tools.split_mfq import split_mfq


def test_official_root_file_and_tinfer_assets(tmp_path):
    import json
    from tyloinfer.formats.assets import MODEL_CONFIG_ASSET

    first = tmp_path / 'Q3/model-Q3.tq'
    last = tmp_path / 'Q3/model-Q3-00003-of-00003.tq'
    shared = tmp_path / 'Engram/model-Engram-MXFP8-00002-of-00003.tq'
    first.parent.mkdir()
    shared.parent.mkdir()
    table = 'model.block.1.associative_memory.embedding.weight'
    config = b'{"model_type":"test"}'
    for index, path, records in [
        (0, first, {'weight.0': np.ones(4, dtype=np.float16), '__tinfer_asset__/model_config.json': config}),
        (1, shared, {table: np.ones(4, dtype=np.float16)}),
        (2, last, {'weight.2': np.ones(4, dtype=np.float16)}),
    ]:
        save(path, FileHeader(version=2, model_arch='test', extra={
            'split.no': index, 'split.count': 3,
            'split.tensors.count': 3, 'split.records.count': 4,
            'runtime_assets': {'version': 1, 'assets': {'model_config.json': {
                'record': '__tinfer_asset__/model_config.json', 'bytes': len(config),
            }}},
        }), records)
    for entry in (first, last):
        with open_mmap(entry) as model:
            assert set(model.records) == {'weight.0', table, 'weight.2', MODEL_CONFIG_ASSET}
            assert model[MODEL_CONFIG_ASSET] == config
            manifest = model.header.extra['runtime_assets']['assets']['model_config.json']
            assert manifest['record'] == MODEL_CONFIG_ASSET
            assert json.loads(model[manifest['record']])['model_type'] == 'test'


def test_root_entry_rejects_an_ambiguous_first_shard(tmp_path):
    first = tmp_path / 'model.tq'
    for index, path in [(0, first), (0, format_shard_path(first, 1, 2)), (1, format_shard_path(first, 2, 2))]:
        save(path, FileHeader(version=2, model_arch='test', extra={
            'split.no': index, 'split.count': 2,
        }), {f'weight.{index}': np.ones(4, dtype=np.float16)})
    with pytest.raises(ValueError, match='duplicate MFQ shard index 1'):
        open_mmap(first)


def test_unindexed_entry_must_be_the_first_shard(tmp_path):
    path = tmp_path / 'model.tq'
    save(path, FileHeader(version=2, model_arch='test', extra={
        'split.no': 1, 'split.count': 2,
    }), {'weight': np.ones(4, dtype=np.float16)})
    with pytest.raises(ValueError, match='lacks -00001-of-00000 suffix'):
        open_mmap(path)


def test_asset_prefix_normalization_rejects_collisions(tmp_path):
    path = tmp_path / 'model.tq'
    save(path, FileHeader(version=2, model_arch='test'), {
        '__tinfer_asset__/model_config.json': b'{}',
        '__mfq_asset__/model_config.json': b'{}',
    })
    with pytest.raises(ValueError, match='duplicate MFQ tensor record'):
        open_mmap(path)


def test_tq_shards_preserve_canonical_shapes_and_row_manifest(tmp_path):
    table = 'model.block.1.associative_memory.embedding.weight'
    names = [f'{table}.tq_row_shard.{index:05d}' for index in range(2)]
    tensors = {name: random_tensor(3, 37, 300, seed=index, groupsize=16)
               for index, name in enumerate(names)}
    tensors['model.output.weight'] = random_tensor(5, 37, 500, seed=3)
    metadata = {'tq_shapes': {table: [6, 37], 'model.output.weight': [5, 37]},
                'tq_row_shards': {table: names}}
    source = tmp_path / 'source.tq'
    save(source, FileHeader(model_arch='test', extra=metadata), tensors)
    paths = split_mfq(source, tmp_path / 'split/model.tq', split_max_tensors=1)
    assert len(paths) == 3
    for path in paths:
        with open_mmap(path) as model:
            assert model.header.extra['tq_shapes'] == metadata['tq_shapes']
            assert model.header.extra['tq_row_shards'] == metadata['tq_row_shards']
            assert set(model.records) == set(tensors)
            for name, tensor in tensors.items():
                assert model[name].to_bytes() == tensor.to_bytes()


@pytest.mark.parametrize('suffix', ['.tq', '.mfq'])
def test_tq_model_is_discovered_and_missing_shard_stays_incomplete(tmp_path, suffix):
    for index in (1, 2):
        save(format_shard_path(tmp_path / f'model{suffix}', index, 2),
             FileHeader(version=2, model_arch='test', extra={
                 'split.no': index - 1, 'split.count': 2,
                 'split.tensors.count': 2, 'split.records.count': 2,
             }), {f'weight.{index}': np.ones((2, 4), dtype=np.float16)})

    async def run():
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        complete = await catalog.list()
        assert len(complete.data) == 1
        assert complete.data[0].complete
        format_shard_path(tmp_path / f'model{suffix}', 1, 2).unlink()
        incomplete = await catalog.list(refresh=True)
        assert len(incomplete.data) == 1
        assert not incomplete.data[0].complete
    asyncio.run(run())


def test_tq_hub_variant_includes_shared_engram_and_has_header_candidates():
    names = ['Q3/model-Q3-00004-of-00004.tq', 'Q3/model-Q3-00001-of-00004.tq',
             'Engram/model-Engram-MXFP8-00002-of-00004.tq',
             'Engram/model-Engram-MXFP8-00003-of-00004.tq']
    files = [HubModelFile(name=name, byte_size=100) for name in names]
    variant, = _model_variants(files, HubSystemProfile(platform='test', machine='test', backend='unknown'))
    assert set(variant.files) == set(names)
    assert variant.format == 'tq'  # The serialized container remains MFQ1.
    from tyloinfer.server.protocol.models import HubModelInfo
    info = HubModelInfo(provider='huggingface', repo_id='Tylogi/model-TQ', revision='main', files=files)
    assert len(HubCatalog._missing_mfq_files(info)) == 4
    assert HubCatalog._is_official_repo(info.repo_id)


@pytest.mark.parametrize('suffix', ['.tq', '.mfq'])
def test_shard_families_do_not_mix_container_suffixes(tmp_path, suffix):
    other = '.mfq' if suffix == '.tq' else '.tq'
    primary = f'Q3/model-Q3-00001-of-00002{suffix}'
    wrong = f'Engram/model-Engram-MXFP8-00002-of-00002{other}'
    assert with_engram_shards([primary], [primary, wrong]) == [primary]
    for index, extension in ((1, suffix), (2, other)):
        save(format_shard_path(tmp_path / f'model{extension}', index, 2),
             FileHeader(version=2, model_arch='test', extra={
                 'split.no': index - 1, 'split.count': 2,
             }), {f'weight.{index}': np.ones(4, dtype=np.float16)})
    with pytest.raises(FileNotFoundError, match='missing MFQ shard'):
        open_mmap(format_shard_path(tmp_path / f'model{suffix}', 1, 2))


def test_hub_keeps_tq_container_suffix_families_separate():
    files = [HubModelFile(name=f'Q3/model-Q3-00001-of-00002{suffix}', byte_size=100)
             for suffix in ('.tq', '.mfq')]
    files.extend(HubModelFile(name=f'Engram/model-Engram-MXFP8-00002-of-00002{suffix}', byte_size=200)
                 for suffix in ('.tq', '.mfq'))
    variants = _model_variants(files, HubSystemProfile(platform='test', machine='test', backend='unknown'))
    assert len(variants) == 2
    for variant in variants:
        assert len(variant.files) == 2
        suffix = '.mfq' if variant.id.startswith('mfq:') else '.tq'
        assert all(name.endswith(suffix) for name in variant.files)
