from __future__ import annotations

import asyncio
import json

import httpx
import numpy as np
import pytest
from safetensors.numpy import save_file

from tyloinfer.formats.header import FileHeader
from tyloinfer.formats.io import save
from tyloinfer.server.api import create_app
from tyloinfer.server.services.model_deletion import ModelDeletionError, model_deletion_plan
from tyloinfer.server.services.service import ServerService
from tyloinfer.server.state.catalog import ModelCatalog
from tyloinfer.server.state.storage import SessionStore
from tyloinfer.server.state.table_components import inspect_table_component
from tests.test_engram_companions import TABLE, checkpoint
from tests.test_gguf_catalog import header, write_raw_parts
from tests.test_server_models import IdleBackend


PLE = 'model.block.2.position_embedding.ngram.shard.0.weight'


def save_tables(path, *names, extra=None):
    save(path, FileHeader(version=2, model_arch='qwen4_exp_text', extra=extra or {}),
         {name: np.ones((2, 4), dtype=np.float16) for name in names})


@pytest.mark.parametrize('name, kind', [(TABLE, 'engram'), (PLE, 'ple')])
def test_only_independent_tables_are_included_on_request(tmp_path, name, kind):
    pure, packed = tmp_path / 'table.tq', tmp_path / 'packed.tq'
    save_tables(pure, name)
    save_tables(packed, name, 'model.output.weight')

    async def check():
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        models = (await catalog.list(refresh=True)).data
        assert len(models) == 1 and models[0].name == 'packed'
        all_items = (await catalog.list(include_components=True)).data
        component, = [item for item in all_items if item.asset_kind == kind]
        assert component.complete and not component.loadable
        assert component.total_bytes == pure.stat().st_size
        assert component.shard_count == component.tensor_count == 1
        assert component.owned_file_bytes == component.total_bytes
        resolved = await catalog.get(component.id)
        assert resolved.path == pure
        with pytest.raises(ModelDeletionError):
            model_deletion_plan(resolved)
        assert inspect_table_component(packed) is None
        pure.unlink()
        assert len((await catalog.list(refresh=True, include_components=True)).data) == 1
    asyncio.run(check())


def test_shared_engram_keeps_original_split_indices_and_is_counted_once(tmp_path):
    first, table = checkpoint(tmp_path, 'Q3')
    checkpoint(tmp_path, 'Q2.5')

    async def check():
        items = (await ModelCatalog([tmp_path]).list(include_components=True)).data
        models = [item for item in items if item.asset_kind == 'model']
        component, = [item for item in items if item.asset_kind == 'engram']
        assert len(models) == 2 and all(item.complete for item in models)
        assert component.shard_count == 1 and component.complete
        assert component.total_bytes == table.stat().st_size
        assert all(item.total_bytes - item.owned_file_bytes == component.total_bytes for item in models)
        assert sum(item.owned_file_bytes for item in items) == sum(path.stat().st_size for path in tmp_path.rglob('*.tq'))
        for path in first:
            artifact = await ModelCatalog([tmp_path]).resolve_path(path)
            assert str(table) not in model_deletion_plan(artifact).files
    asyncio.run(check())


def test_pure_table_shards_are_one_asset_not_incomplete_models(tmp_path):
    for index in (2, 3):
        save_tables(tmp_path / f'Qwen3.8-Flash-Next-Engram-{index:05d}-of-00043.tq',
                    TABLE.replace('.1.', f'.{index}.'), extra={'split.no': index - 1, 'split.count': 43})

    async def check():
        catalog = ModelCatalog([tmp_path])
        assert not (await catalog.list()).data
        component, = (await catalog.list(include_components=True)).data
        assert component.asset_kind == 'engram'
        assert component.shard_count == component.tensor_count == 2
        assert component.complete and component.missing_shards == 0
    asyncio.run(check())


def test_embedded_tables_and_table_projections_are_not_separate_assets(tmp_path):
    save_tables(tmp_path / 'complete.tq', TABLE, PLE, 'model.token_embedding.weight')
    save_tables(tmp_path / 'mixed-shard.tq', TABLE, 'model.block.1.associative_memory.projection.weight')
    assert inspect_table_component(tmp_path / 'complete.tq') is None
    assert inspect_table_component(tmp_path / 'mixed-shard.tq') is None


@pytest.mark.parametrize('name, kind', [(b'per_layer_token_embd.weight', 'ple'), (b'blk.3.engram_embd.weight', 'engram')])
def test_raw_gguf_table_parts_are_aggregated_without_duplicate_tensors(tmp_path, name, kind):
    source = tmp_path / 'source.gguf'
    header(source, 1, name)
    parts = write_raw_parts(tmp_path, 'shared-table', source.read_bytes())
    source.unlink()

    async def check():
        catalog = ModelCatalog([tmp_path])
        assert not (await catalog.list()).data
        component, = (await catalog.list(include_components=True)).data
        assert component.asset_kind == kind and component.shard_count == 4
        assert component.tensor_count == 1
        assert component.total_bytes == sum(part.stat().st_size for part in parts)
    asyncio.run(check())


def test_component_metadata_cache_is_invalidated_when_file_becomes_mixed(tmp_path):
    path = tmp_path / 'shared.tq'
    save_tables(path, TABLE)

    async def check():
        catalog = ModelCatalog([tmp_path])
        item, = (await catalog.list(include_components=True)).data
        assert item.asset_kind == 'engram'
        save_tables(path, TABLE, 'model.output.weight')
        item, = (await catalog.list(refresh=True, include_components=True)).data
        assert item.asset_kind == 'model'
    asyncio.run(check())


def test_safetensors_lists_only_physical_table_shards_and_counts_files_once(tmp_path):
    root = tmp_path / 'Qwen3.8-Flash-Next-oQ4e'
    root.mkdir()
    (root / 'config.json').write_text(json.dumps({'model_type': 'test'}))
    save_file({PLE: np.ones((2, 4), dtype=np.float16)}, root / 'model-00001-of-00002.safetensors')
    save_file({TABLE: np.ones((2, 4), dtype=np.float16), 'model.output.weight': np.ones((2, 4), dtype=np.float16)},
              root / 'model-00002-of-00002.safetensors')

    async def check():
        catalog = ModelCatalog([tmp_path])
        items = (await catalog.list(include_components=True)).data
        component, = [item for item in items if item.asset_kind != 'model']
        model, = [item for item in items if item.asset_kind == 'model']
        assert component.asset_kind == 'ple' and component.format == 'hf'
        assert component.name == root.name + '-PLE'
        assert component.shard_count == 1 and not component.loadable
        assert model.owned_file_bytes + component.total_bytes == model.total_bytes
    asyncio.run(check())


@pytest.mark.parametrize('format', ['tq', 'hf'])
def test_table_spread_between_pure_and_mixed_shards_is_not_independently_usable(tmp_path, format):
    first = PLE
    second = PLE.replace('shard.0.', 'shard.1.')
    if format == 'hf':
        (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'test'}))
        save_file({first: np.ones((2, 4), dtype=np.float16)}, tmp_path / 'model-00001-of-00002.safetensors')
        save_file({second: np.ones((2, 4), dtype=np.float16), 'model.output.weight': np.ones((2, 4), dtype=np.float16)},
                  tmp_path / 'model-00002-of-00002.safetensors')
    else:
        save_tables(tmp_path / 'model-00001-of-00002.tq', first,
                    extra={'split.no': 0, 'split.count': 2, 'split.tensors.count': 3, 'split.records.count': 3})
        save_tables(tmp_path / 'model-00002-of-00002.tq', second, 'model.output.weight',
                    extra={'split.no': 1, 'split.count': 2, 'split.tensors.count': 3, 'split.records.count': 3})

    async def check():
        items = (await ModelCatalog([tmp_path]).list(include_components=True)).data
        assert items and all(item.asset_kind == 'model' for item in items)
    asyncio.run(check())


def test_api_components_are_opt_in_and_files_can_be_browsed(tmp_path):
    root, external = tmp_path / 'models', tmp_path / 'external'
    root.mkdir(); external.mkdir()
    save_tables(external / 'Qwen3.8-Flash-Next-Engram.tq', TABLE)

    async def check():
        catalog = ModelCatalog([root])
        registered, = (await catalog.register_directory(path=external)).data
        assert registered.asset_kind == 'engram'
        service = ServerService(SessionStore(tmp_path / 'state'), IdleBackend(), catalog=catalog)
        app = create_app(service)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                assert (await client.get('/api/v1/models')).json()['data'] == []
                response = await client.get('/api/v1/models', params={'include_components': 'true'})
                assert response.status_code == 200
                asset, = response.json()['data']
                assert asset['asset_kind'] == 'engram' and not asset['loadable']
                listing = await client.get(f"/api/v1/models/{asset['id']}/directory")
                assert listing.status_code == 200
                assert listing.json()['files'][0]['name'] == 'Qwen3.8-Flash-Next-Engram.tq'
    asyncio.run(check())
