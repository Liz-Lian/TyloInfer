from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from tyloinfer.server.api import create_app
from tyloinfer.server.protocol.models import HubModelFile
from tyloinfer.server.services.service import ServerService
from tyloinfer.server.state.storage import SessionStore
from tests.test_download_directory import make_handlers
from tests.test_server_service import FakeBackend


@pytest.mark.parametrize('provider', ['huggingface', 'modelscope'])
def test_complete_download_is_rejected_before_a_job_is_created(tmp_path, provider):
    async def run():
        handlers = make_handlers(tmp_path)
        request = {'repo_id': 'owner/model', 'destination': 'models/owner/model/S4',
            'include': ['S4/*.tq', '*.json'], 'expected_bytes': 5}
        destination = handlers._download_output(request['destination'], None)
        (destination / 'S4').mkdir()
        (destination / 'S4/model.tq').write_bytes(b'model')
        (destination / 'config.json').write_bytes(b'{}')
        files = [HubModelFile(name='S4/model.tq', byte_size=5), HubModelFile(name='config.json', byte_size=2),
            HubModelFile(name='S5/model.tq', byte_size=8)]
        hub = SimpleNamespace(download_files=AsyncMock(return_value=SimpleNamespace(files=files)))
        store = SessionStore(tmp_path / 'state.sqlite3')
        service = ServerService(store, FakeBackend(), hub_catalog=hub, tool_handlers=handlers)
        service.jobs.submit = AsyncMock()
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/api/v1/jobs', json={'kind': f'download.{provider}', 'payload': request})
        assert response.status_code == 409
        assert response.json()['error']['code'] == 'download_already_exists'
        service.jobs.submit.assert_not_called()
        assert not (tmp_path / 'owned').exists()
    asyncio.run(run())


@pytest.mark.parametrize('state', ['missing', 'wrong_size', 'partial', 'different_folder'])
def test_incomplete_downloads_and_other_destinations_are_not_blocked(tmp_path, state):
    async def run():
        handlers = make_handlers(tmp_path)
        payload = {'repo_id': 'owner/model', 'destination': 'models/owner/model/S4', 'include': ['*.tq']}
        request, destination = handlers.download_request('download.modelscope', payload)
        assert not destination.exists()
        if state != 'different_folder':
            destination.mkdir(parents=True)
            (destination / 'first.tq').write_bytes(b'first')
        if state == 'wrong_size':
            (destination / 'second.tq').write_bytes(b'bad')
        if state == 'partial':
            (destination / 'second.tyloinfer.incomplete').write_bytes(b'second')
        files = [HubModelFile(name='first.tq', byte_size=5), HubModelFile(name='second.tq', byte_size=6)]
        hub = SimpleNamespace(download_files=AsyncMock(return_value=SimpleNamespace(files=files)))
        service = ServerService(SessionStore(tmp_path / 'state.sqlite3'), FakeBackend(), hub_catalog=hub, tool_handlers=handlers)
        from tyloinfer.server.protocol.models import CreateJobRequest
        await service._check_download_job(CreateJobRequest(kind='download.modelscope', payload=payload))
        if state == 'different_folder':
            hub.download_files.assert_not_called()
            assert not destination.exists()
    asyncio.run(run())


def test_concurrent_submissions_reuse_one_download_task(tmp_path):
    async def run():
        from tyloinfer.server.protocol.models import CreateJobRequest
        from tyloinfer.server.services.jobs import JobManager
        handlers = make_handlers(tmp_path)
        store = SessionStore(tmp_path / 'state.sqlite3')
        waiting = asyncio.Event()
        async def download(context, payload):
            await waiting.wait()
            return {}
        jobs = JobManager(store, {'download.modelscope': download})
        service = ServerService(store, FakeBackend(), jobs=jobs, tool_handlers=handlers)
        request = CreateJobRequest(kind='download.modelscope', payload={
            'repo_id': 'owner/model', 'destination': 'models/owner/model/S4', 'include': ['*.tq']})
        try:
            first, second = await asyncio.gather(service.create_job(request), service.create_job(request))
            assert first.id == second.id
            assert len(store.list_jobs()) == 1
        finally:
            await jobs.close()
    asyncio.run(run())


def test_complete_mfq_on_the_other_hub_blocks_a_second_download(tmp_path):
    async def run():
        handlers = make_handlers(tmp_path)
        destination = handlers._download_output('models/modelscope/owner/model/S4', None)
        (destination / 'model.tq').write_bytes(b'model')
        files = [HubModelFile(name='model.tq', byte_size=5), HubModelFile(name='configuration.json', byte_size=2)]
        hub = SimpleNamespace(download_files=AsyncMock(return_value=SimpleNamespace(files=files)))
        store = SessionStore(tmp_path / 'state.sqlite3')
        service = ServerService(store, FakeBackend(), hub_catalog=hub, tool_handlers=handlers)
        service.jobs.submit = AsyncMock()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url='http://test') as client:
            response = await client.post('/api/v1/jobs', json={'kind': 'download.huggingface', 'payload': {
                'repo_id': 'owner/model', 'destination': 'models/huggingface/owner/model/S4', 'include': ['*.tq', '*.json']}})
        assert response.status_code == 409 and response.json()['error']['code'] == 'download_already_exists'
        service.jobs.submit.assert_not_called()
        assert not (tmp_path / 'models/huggingface').exists()
    asyncio.run(run())
