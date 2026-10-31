from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from tests.test_server_hub import FakeHub
from tests.test_server_service import FakeBackend
from tyloinfer.server.api import create_app
from tyloinfer.server.services import hub
from tyloinfer.server.services.hub import HubCatalog, HubError, _search_formats
from tyloinfer.server.services.service import ServerService
from tyloinfer.server.state.storage import SessionStore


def test_discovery_and_filter_parameters_are_forwarded_and_validated(tmp_path):
    calls = []

    class BrowserHub(FakeHub):
        async def search(self, provider, query, *, limit, format="all", sort="downloads"):
            calls.append((provider, query, limit, format, sort))
            return await super().search(provider, query, limit=limit)

        async def readme(self, provider, repo_id, revision):
            calls.append((provider, repo_id, revision))
            return "# Model card"

    async def run():
        service = ServerService(SessionStore(tmp_path / "server.sqlite3"), FakeBackend(), hub_catalog=BrowserHub())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
            response = await client.get('/api/v1/hub/models', params={'provider': 'huggingface', 'format': 'gguf', 'sort': 'updated', 'limit': 40})
            assert response.status_code == 200
            assert calls[-1] == ('huggingface', '', 40, 'gguf', 'updated')
            for field, value in [('format', 'unknown'), ('sort', 'unknown'), ('limit', '101')]:
                response = await client.get('/api/v1/hub/models', params={'provider': 'huggingface', field: value})
                assert response.status_code == 422
            response = await client.get('/api/v1/hub/models/huggingface/owner/model/readme', params={'revision': 'branch/name'})
            assert response.status_code == 200
            assert response.json() == {'content': '# Model card'}
            assert calls[-1] == ('huggingface', 'owner/model', 'branch/name')
            response = await client.get('/api/v1/hub/models/unknown/owner/model/readme')
            assert response.status_code == 422
        await service.aclose()

    asyncio.run(run())


def test_huggingface_discovery_uses_native_format_and_sort_filters(monkeypatch):
    import huggingface_hub

    calls = []

    class Api:
        def list_models(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(id='owner/model-GGUF', tags=['gguf'], downloads=42, likes=2, last_modified=None)]

    monkeypatch.delenv('HF_TOKEN', raising=False)
    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    result = HubCatalog._search_huggingface('', 40, 'gguf', 'updated')
    assert calls[-1]['search'] == ''
    assert calls[-1]['filter'] == 'gguf'
    assert calls[-1]['sort'] == 'lastModified'
    assert 'tags' in calls[-1]['expand']
    assert result.data[0].formats == ['gguf']


@pytest.mark.parametrize(('name', 'tags', 'formats'), [
    ('owner/model-TQ', [], ['tq']),
    ('owner/model-TQ4', [], ['tq']),
    ('owner/model-Gyro-TQ3_0-GGUF', [], ['gguf']),
    ('owner/model', ['mlx'], ['mlx']),
    ('owner/model', ['gguf'], ['gguf']),
    ('owner/model', ['gguf', 'mlx'], ['gguf', 'mlx']),
    ('owner/model', ['safetensors'], ['hf']),
    ('owner/unknown', [], []),
])
def test_format_badges_do_not_mislabel_gyro_as_native_tq(name, tags, formats):
    assert _search_formats(name, tags) == formats


def test_modelscope_search_uses_shared_transport_and_remote_filters(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={'data': {'models': [
            {'id': 'owner/newer-GGUF', 'downloads': 5, 'last_modified': '2026-10-30T00:00:00Z'},
            {'id': 'owner/older-GGUF', 'downloads': 50, 'last_modified': '2026-10-01T00:00:00Z'},
        ]}})

    monkeypatch.setattr(hub, '_metadata_client', lambda endpoint, **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    result = asyncio.run(HubCatalog._search_modelscope('model', 40, 'gguf', 'updated'))
    assert [item.repo_id for item in result.data] == ['owner/newer-GGUF', 'owner/older-GGUF']
    assert requests[0].url.host == 'modelscope.cn'
    assert requests[0].url.path == '/openapi/v1/models'
    assert dict(requests[0].url.params) == {'search': 'model', 'sort': 'last_modified', 'filter.library': 'gguf', 'page_size': '40', 'page_number': '1'}


@pytest.mark.parametrize('provider', ['huggingface', 'modelscope'])
def test_model_card_download_is_pinned_and_not_arbitrary_url(monkeypatch, provider):
    import huggingface_hub
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, text='# Model card')

    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: None)
    monkeypatch.setattr(hub, '_metadata_client', lambda endpoint, **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    assert asyncio.run(HubCatalog.readme(provider, 'owner/model', 'branch/name')) == '# Model card'
    request = requests[0]
    if provider == 'huggingface':
        assert request.url.host == 'huggingface.co'
        assert '/resolve/branch%2Fname/README.md' in str(request.url)
    else:
        assert request.url.host == 'modelscope.cn'
        assert request.url.params['Revision'] == 'branch/name'
        assert request.url.params['FilePath'] == 'README.md'
    with pytest.raises(HubError):
        asyncio.run(HubCatalog.readme(provider, 'https://invalid.test/model', None))
    assert len(requests) == 1


def test_model_card_missing_and_oversized_responses(monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: None)
    response = httpx.Response(404)
    monkeypatch.setattr(hub, '_metadata_client', lambda endpoint, **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response), **kwargs))
    assert asyncio.run(HubCatalog.readme('huggingface', 'owner/model', None)) is None
    response = httpx.Response(200, content=b'x' * (512 * 1024 + 1))
    with pytest.raises(HubError, match='size limit'):
        asyncio.run(HubCatalog.readme('huggingface', 'owner/model', None))
