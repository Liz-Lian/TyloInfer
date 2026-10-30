import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from tests.test_gguf_catalog import header, write_raw_parts
from tyloinfer.server.protocol.models import ModelLoadRequest
from tyloinfer.server.runtime.native import (
    NativeRuntime, NativeRuntimeError, native_gguf_companion_arguments, native_tokenizer_arguments,
)
from tyloinfer.server.runtime.native_metadata import NativeModelInspector
from tyloinfer.server.runtime.runtime_pool import RuntimePool
from tyloinfer.server.services.jobs import JobExecutionError
from tyloinfer.server.state.catalog import ModelCatalog


def test_raw_parts_use_the_tokenizer_asset_collected_by_native_loader(tmp_path):
    source = tmp_path / 'source.gguf'
    header(source, 3)
    parts = write_raw_parts(tmp_path, 'xq', source.read_bytes())
    for part in parts:
        assert native_tokenizer_arguments(part, 'metal') == []
    parts[1].unlink()
    with pytest.raises(FileNotFoundError, match='missing GGUF part'):
        native_tokenizer_arguments(parts[-1], 'metal')


def test_companions_are_explicit_deduplicated_and_complete(tmp_path):
    source = tmp_path / 'source.gguf'
    header(source, 1)
    parts = write_raw_parts(tmp_path, 'tables', source.read_bytes())
    folder = tmp_path / 'engram'
    folder.mkdir()
    (folder / 'weights.safetensors').write_bytes(b'')
    paths = [str(parts[-1]), str(parts[0]), str(folder)]
    expected = ['--gguf-companion', str(parts[0]), '--gguf-companion', str(folder)]
    assert native_gguf_companion_arguments(tmp_path / 'model.gguf', paths, 'metal') == expected
    assert native_gguf_companion_arguments(tmp_path / 'model.gguf', [], 'metal') == []
    parts[2].unlink()
    with pytest.raises(NativeRuntimeError, match='missing GGUF part'):
        native_gguf_companion_arguments(tmp_path / 'model.gguf', paths, 'metal')


@pytest.mark.parametrize('backend,model', [('cuda', 'model.gguf'), ('metal', 'model.tq')])
def test_unsupported_companion_combination_is_explicit(tmp_path, backend, model):
    with pytest.raises(NativeRuntimeError, match='GGUF model on Metal'):
        native_gguf_companion_arguments(tmp_path / model, ['tables.gguf'], backend)


def test_unrelated_files_and_folders_are_not_silently_selected(tmp_path):
    model = tmp_path / 'model.gguf'
    header(model, 3)
    for companion in [tmp_path / 'missing.gguf', tmp_path, model]:
        with pytest.raises(NativeRuntimeError, match='Invalid GGUF companion'):
            native_gguf_companion_arguments(model, [str(companion)], 'metal')


def test_worker_and_inspector_forward_the_same_explicit_companion(tmp_path, monkeypatch):
    model, table = tmp_path / 'model.gguf', tmp_path / 'table.gguf'
    header(model, 3)
    header(table, 1)
    runtime = NativeRuntime(tmp_path / 'worker', model, 'model', 'metal', gguf_companions=[str(table)])
    arguments = runtime.command(12345)
    assert arguments[-2:] == ['--gguf-companion', str(table)]
    run = Mock(return_value=Mock(returncode=0, stdout=b'{"config":{},"tensors":{}}'))
    monkeypatch.setattr('tyloinfer.server.runtime.native_metadata.subprocess.run', run)
    NativeModelInspector(runtime.executable, 'metal')(model, [str(table)])
    assert run.call_args.args[0][-2:] == arguments[-2:]


def test_managed_load_and_profile_roundtrip_retain_companion_paths(tmp_path):
    model, table = tmp_path / 'model.gguf', tmp_path / 'table.gguf'
    header(model, 3)
    header(table, 1)
    inspector = lambda _: {'architecture': 'deepseek_v41', 'config': {'max_position_embeddings': 8192}, 'tensors': {}}
    catalog = ModelCatalog([tmp_path], native_inspector=inspector)
    artifact = asyncio.run(catalog.resolve('model'))
    request = ModelLoadRequest(model='model', gguf_companions=[str(table)])
    restored = ModelLoadRequest.model_validate_json(request.model_dump_json())
    assert restored.gguf_companions == [str(table)]
    pool = RuntimePool(catalog, tmp_path / 'worker', backend='metal')
    command, _ = pool._launch_configuration(artifact, restored, port=12345)
    assert command[command.index('--gguf-companion') + 1] == str(table)
    with pytest.raises(ValidationError):
        ModelLoadRequest(model='model', gguf_companions=[''])


def test_missing_companion_is_rejected_before_runtime_eviction(tmp_path):
    async def check():
        model = tmp_path / 'model.gguf'
        header(model, 3)
        catalog = ModelCatalog([tmp_path], native_inspector=lambda _: {
            'architecture': 'deepseek_v41', 'config': {}, 'tensors': {}})
        pool = RuntimePool(catalog, tmp_path / 'worker', backend='metal')
        pool._maintain_prefix_disk_budget = AsyncMock()
        pool._rebalance_resident_budget = AsyncMock()
        with pytest.raises(JobExecutionError, match='Invalid GGUF companion'):
            await pool._load(Mock(), {'model': 'model', 'gguf_companions': [str(tmp_path / 'missing.gguf')]})
        pool._rebalance_resident_budget.assert_not_called()
        assert not pool._loading_model_names
    asyncio.run(check())


def test_incompatible_companion_is_rejected_by_native_before_eviction(tmp_path, monkeypatch):
    async def check():
        model, table = tmp_path / 'model.gguf', tmp_path / 'table.gguf'
        header(model, 3); header(table, 1)
        catalog = ModelCatalog([tmp_path], native_inspector=lambda _: {
            'architecture': 'deepseek_v41', 'config': {}, 'tensors': {}})
        pool = RuntimePool(catalog, tmp_path / 'worker', backend='metal')
        pool._maintain_prefix_disk_budget = AsyncMock()
        pool._rebalance_resident_budget = AsyncMock()
        inspector = Mock(side_effect=ValueError('Engram geometry mismatch'))
        monkeypatch.setattr(NativeModelInspector, '__call__', inspector)
        with pytest.raises(JobExecutionError, match='Engram geometry mismatch'):
            await pool._load(Mock(), {'model': 'model', 'gguf_companions': [str(table)]})
        inspector.assert_called_once_with(model, [str(table)])
        pool._rebalance_resident_budget.assert_not_called()
        assert not pool._loading_model_names
    asyncio.run(check())


@pytest.mark.parametrize('embedded', [False, True])
def test_external_table_requirement_is_derived_from_metadata(tmp_path, embedded):
    async def check():
        model = tmp_path / 'model.gguf'
        header(model, 3)
        tensors = {f'model.block.{layer}.associative_memory.embedding.weight': {
            'dtype': 'GGUF_ENGRAM_MXFP8', 'shape': [8, 256], 'nbytes': 2112} for layer in [1, 14]} if embedded else {}
        catalog = ModelCatalog([tmp_path], native_inspector=lambda _: {
            'architecture': 'deepseek_v41', 'config': {'text_config': {'engram_layer_ids': [1, 14]}}, 'tensors': tensors})
        artifact = await catalog.resolve('model')
        assert artifact.resource.gguf_companion_required is not embedded
        if not embedded:
            pool = RuntimePool(catalog, tmp_path / 'worker', backend='metal')
            pool._maintain_prefix_disk_budget = AsyncMock()
            pool._rebalance_resident_budget = AsyncMock()
            with pytest.raises(JobExecutionError, match='external Engram'):
                await pool._load(Mock(), {'model': 'model'})
            pool._rebalance_resident_budget.assert_not_called()
    asyncio.run(check())
