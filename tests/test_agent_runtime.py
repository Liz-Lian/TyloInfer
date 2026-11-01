from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, Request

from tyloinfer.server.api import create_app
from tyloinfer.server.api.auth import ApiKeyManager
from tyloinfer.server.services.agent_runtime import AgentRuntime, AgentRuntimeError
from tyloinfer.server.state.storage import SessionStore


def test_agent_routes_require_admin_and_restrict_model_credential(tmp_path: Path) -> None:
    async def run() -> None:
        app = create_app(api_keys=ApiKeyManager(SessionStore(tmp_path / 'state.sqlite3'), 'root-secret'))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            viewer = await client.post('/api/v1/auth/keys', headers={'Authorization': 'Bearer root-secret'},
                json={'name': 'viewer', 'role': 'viewer'})
            headers = {'Authorization': f"Bearer {viewer.json()['token']}"}
            assert (await client.get('/api/v1/agent/state', headers=headers)).status_code == 403
            assert (await client.get('/api/v1/agent/search?query=test', headers=headers)).status_code == 403
            assert (await client.post('/api/v1/agent/start', headers=headers)).status_code == 403
            assert (await client.post('/api/v1/agent/workspaces/test/session', headers=headers)).status_code == 403
            assert (await client.get('/api/v1/agent/sessions/test/controls', headers=headers)).status_code == 403
            assert (await client.post('/api/v1/agent/sessions/test/commands', headers=headers, json={'line': '/plan'})).status_code == 403
            assert (await client.post('/api/v1/agent/sessions/test/files', headers=headers, content=b'test')).status_code == 403
            assert (await client.get('/api/v1/agent/sessions/test/export', headers=headers)).status_code == 403
            assert (await client.get('/api/v1/agent/state')).status_code == 401
            headers = {'Authorization': f'Bearer {app.state.agent_token}'}
            assert (await client.get('/v1/models', headers=headers)).status_code == 501
            assert (await client.get('/api/v1/agent/state', headers=headers)).status_code == 403
            assert (await client.post('/api/v1/jobs', headers=headers, json={})).status_code == 403
            assert (await client.get('/api/v1/agent/state', headers={'Authorization': 'Bearer root-secret'})).status_code == 501
    asyncio.run(run())


def test_agent_controls_and_commands_use_actual_dsh_scope(tmp_path: Path) -> None:
    async def run() -> None:
        runtime = AgentRuntime(tmp_path)
        calls = []
        async def rpc(method, args):
            calls.append((method, args))
            return {'session/projections': {'values': {'permissions': {'currentValue': 'read-only'}}},
                'commands/list': [{'name': 'plan'}], 'permissionPresets/catalog': {'options': [{'value': 'read-only'}]},
                'commands/execute': {'result': {'kind': 'error', 'text': 'Permission change refused'}}}[method]
        runtime.rpc = rpc
        controls = await runtime.controls('session')
        assert controls['commands'] == [{'name': 'plan'}]
        assert controls['values']['permissions']['currentValue'] == 'read-only'
        assert ('commands/list', {'agentId': 'session'}) in calls
        with pytest.raises(AgentRuntimeError, match='Permission change refused'):
            await runtime.command('session', '/permission read-only', [])
        assert calls[-1] == ('commands/execute', {'agentId': 'session', 'line': '/permission read-only', 'submittedAttachments': []})
    asyncio.run(run())


def test_agent_search_is_scoped_bounded_and_exposes_no_arbitrary_rpc(tmp_path: Path) -> None:
    async def run() -> None:
        calls = []
        async def rpc(method, args):
            calls.append((method, args))
            return {'items': [{'sessionId': 'match', 'snippet': 'matching history'}], 'hasMore': True}
        app = create_app(service=SimpleNamespace(agent_runtime=SimpleNamespace(rpc=rpc)))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            result = await client.get('/api/v1/agent/search', params={'query': '  history  '})
            assert result.status_code == 200
            assert result.json()['hasMore'] is True
            assert calls == [('session/search', {'request': {'query': 'history'}})]
            for query in ('', 'x' * 501):
                assert (await client.get('/api/v1/agent/search', params={'query': query})).status_code == 422
    asyncio.run(run())


def test_agent_workspace_stream_tracks_archive_snapshots_and_increments(tmp_path: Path) -> None:
    async def run() -> None:
        runtime = AgentRuntime(tmp_path)
        async def frames(endpoint, args):
            yield {'type': 'baseline', 'value': {'items': [], 'archivedSessionIds': ['first']}}
            yield {'type': 'archived', 'archivedSessionIds': ['second']}
        runtime.stream = frames
        await runtime._workspace_changes()
        assert runtime.archived_session_ids == ['second']
        await runtime.close()
        assert runtime.archived_session_ids == []
    asyncio.run(run())


def test_agent_model_routes_advertise_actual_capabilities_and_global_mtp_policy() -> None:
    async def run() -> None:
        async def instances():
            return SimpleNamespace(data=[SimpleNamespace(id=uuid4(), model='test-model', state='ready', context_size=65536, mtp_available=True)])
        async def status(instance_id):
            return {'mtp_service_enabled': False, 'chat_template_capabilities': {
                'thinking': {'supported': True}, 'reasoning_effort': {'values': ['low', 'high']}}}
        async def capabilities(instance_id):
            return SimpleNamespace(model_capabilities=SimpleNamespace(features=SimpleNamespace(image_input=False)))
        captured = []
        async def start(url, token, models):
            captured.extend(models)
        async def state():
            return {'running': True, 'models': captured}
        app = create_app(service=SimpleNamespace(runtime_instances=instances, runtime_status=status,
            runtime_capabilities=capabilities, agent_runtime=SimpleNamespace(start=start, state=state)))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            result = await client.post('/api/v1/agent/start')
            assert result.status_code == 200
            model = result.json()['models'][0]
            assert model['input'] == ['text']
            assert model['reasoningEfforts'] == {'off': 'none', 'low': 'low', 'high': 'high'}
            assert model['mtpAvailable'] is False
    asyncio.run(run())


def test_agent_rpc_whitelist_and_error_boundary(tmp_path: Path) -> None:
    async def run() -> None:
        runtime = AgentRuntime(tmp_path)
        app = create_app(service=SimpleNamespace(agent_runtime=runtime))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.get('/api/v1/agent/state')
            assert response.json()['running'] is False
            for method in ('system/exec', '$events/result', 'session/delete'):
                assert (await client.post('/api/v1/agent/rpc', json={'method': method, 'args': {}})).status_code == 422
            response = await client.post('/api/v1/agent/rpc', json={'method': 'session/create', 'args': {}})
            assert response.status_code == 400
            assert 'not running' in response.json()['error']['message']
            assert (await client.post('/api/v1/agent/approvals', json={'event_id': 'expired', 'allow': True})).status_code == 409
    asyncio.run(run())


def test_agent_early_exit_is_reported_without_startup_timeout(tmp_path: Path, monkeypatch) -> None:
    import shutil
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is not installed')
    cli = tmp_path / 'exit.cjs'
    cli.write_text('process.exit(23)', encoding='utf-8')
    monkeypatch.setattr(AgentRuntime, 'component', staticmethod(lambda: (node, cli)))
    async def run() -> None:
        runtime = AgentRuntime(tmp_path / 'home')
        with pytest.raises(AgentRuntimeError, match='status 23'):
            await asyncio.wait_for(runtime.start('http://127.0.0.1:1/v1', 'private-token', []), 5)
        assert runtime.process is None
    asyncio.run(run())


def test_agent_approval_and_question_responses_are_bound_to_pending_events(tmp_path: Path) -> None:
    async def run() -> None:
        runtime = AgentRuntime(tmp_path)
        calls = []
        async def rpc(method, args):
            calls.append((method, args))
        runtime.rpc = rpc
        runtime.approvals['approval-1'] = {'clientId': 'private-client', 'request': {'toolName': 'write'}}
        await runtime.approve('approval-1', False)
        assert calls[-1][1]['outcome'] == {'kind': 'result', 'value': 'rejected'}
        with pytest.raises(AgentRuntimeError, match='no longer pending'):
            await runtime.approve('approval-1', True)
        runtime.questions['question-1'] = {'clientId': 'private-client', 'request': {'questions': [
            {'id': 'choice', 'question': 'Continue?', 'options': [{'label': 'Yes'}]}]}}
        with pytest.raises(AgentRuntimeError, match='not a valid option'):
            await runtime.answer('question-1', [{'id': 'choice', 'selected': ['Fake']}])
        await runtime.answer('question-1', [{'id': 'choice', 'selected': ['Yes']}])
        assert calls[-1][1]['outcome']['value'] == {'answers': [{'id': 'choice', 'selected': ['Yes']}]}
        assert not runtime.questions
    asyncio.run(run())


def test_workspace_connect_reuses_only_unarchived_blanks_and_serializes_creation(tmp_path: Path) -> None:
    async def run() -> None:
        runtime = AgentRuntime(tmp_path)
        workspace = {'workspaceId': 'workspace', 'path': str(tmp_path), 'sessionIds': []}
        sessions = []
        archived = []
        created = []
        async def stream(endpoint, args):
            yield {'type': 'baseline', 'value': {'items': [workspace], 'archivedSessionIds': archived}}
        async def rpc(method, args):
            if method == 'session/list':
                return {'items': sessions}
            selection = args['request']
            if selection.get('sessionId'):
                return {'sessionId': selection['sessionId']}
            await asyncio.sleep(0)
            session_id = f'session-{len(created)}'
            created.append(session_id)
            sessions.append({'sessionId': session_id, 'blank': True, 'cwd': str(tmp_path)})
            workspace['sessionIds'].append(session_id)
            return {'sessionId': session_id}
        runtime.stream, runtime.rpc = stream, rpc
        results = await asyncio.gather(*(runtime.connect_workspace('workspace') for _ in range(8)))
        assert {result['sessionId'] for result in results} == {'session-0'}
        assert len(created) == 1
        sessions[0]['blank'] = False
        assert (await runtime.connect_workspace('workspace'))['sessionId'] == 'session-1'
        archived.append('session-1')
        assert (await runtime.connect_workspace('workspace'))['sessionId'] == 'session-2'
        sessions[-1]['cwd'] = str(tmp_path / 'different-workspace')
        assert (await runtime.connect_workspace('workspace'))['sessionId'] == 'session-3'
        with pytest.raises(AgentRuntimeError, match='no longer available'):
            await runtime.connect_workspace('missing')
        assert len(created) == 4
    asyncio.run(run())


@pytest.mark.skipif(os.environ.get('TYLOINFER_TEST_DSH') != '1', reason='requires the staged DSH runtime')
def test_real_dsh_tool_loop_persistence_and_parent_pipe(tmp_path: Path) -> None:
    import socket
    import uvicorn
    from fastapi.responses import StreamingResponse

    async def run() -> None:
        api = FastAPI()
        calls: list[dict] = []
        @api.post('/v1/chat/completions')
        async def completion(request: Request):
            body = await request.json()
            if not body.get('tools'):
                return {'id': 'session-title', 'object': 'chat.completion', 'created': 0, 'model': 'agent-test',
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Marker review'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3}}
            calls.append(body)
            assert request.headers['authorization'] == 'Bearer test-inference-token'
            assert request.headers['x-tyloinfer-mtp'] == 'off'
            assert body['reasoning_effort'] == 'none'
            tool_count = sum(message['role'] == 'tool' for message in body['messages'])
            tool_result = tool_count > 0
            if tool_result:
                assert 'workspace-marker-123' in json.dumps(body['messages'])
            if tool_count > 1:
                delta = {'content': 'Agent integration OK'}
            elif tool_count == 1:
                delta = {'tool_calls': [{'index': 0, 'id': 'confirm', 'type': 'function',
                    'function': {'name': 'ask_user_question', 'arguments': json.dumps({'questions': [
                        {'id': 'continue', 'question': 'Continue?', 'options': [{'label': 'Yes'}, {'label': 'No'}]}]})}}]}
            else:
                assert any(tool['function']['name'] == 'read' for tool in body['tools'])
                delta = {'tool_calls': [{'index': 0, 'id': 'read-marker', 'type': 'function',
                    'function': {'name': 'read', 'arguments': json.dumps({'file_path': 'marker.txt'})}}]}
            async def chunks():
                for value, reason in (({'role': 'assistant', **delta}, None), ({}, 'stop' if tool_count > 1 else 'tool_calls')):
                    yield 'data: ' + json.dumps({'id': 'test-completion', 'object': 'chat.completion.chunk',
                        'created': 0, 'model': 'agent-test', 'choices': [{'index': 0, 'delta': value, 'finish_reason': reason}]}) + '\n\n'
                yield 'data: [DONE]\n\n'
            return StreamingResponse(chunks(), media_type='text/event-stream')

        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        (workspace / 'marker.txt').write_text('workspace-marker-123', encoding='utf-8')
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(api, log_level='error', lifespan='off'))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        runtime = AgentRuntime(tmp_path / 'agent')
        models = [{'id': 'agent-test', 'name': 'Agent test', 'contextWindow': 65536, 'maxTokens': 4096, 'input': ['text'],
            'reasoningEfforts': {'off': 'none', 'low': 'low', 'high': 'high'}, 'mtpAvailable': True}]
        try:
            await asyncio.gather(*(runtime.start(f'http://127.0.0.1:{port}/v1', 'test-inference-token', models) for _ in range(2)))
            pid = runtime.process.pid
            created = await runtime.rpc('workspace/create', {'request': {'path': str(workspace)}})
            workspace_id = created['workspace']['workspaceId']
            connections = await asyncio.gather(*(runtime.connect_workspace(workspace_id) for _ in range(4)))
            session_id = connections[0]['sessionId']
            assert all(item['sessionId'] == session_id for item in connections)
            assert len((await runtime.state())['sessions']) == 1
            controls = await runtime.controls(session_id)
            assert {'plan', 'permission', 'goal'} <= {item['name'] for item in controls['commands']}
            assert controls['values']['permissions']['currentValue'] == 'workspace-write'
            await runtime.command(session_id, '/plan', [])
            plan = (await runtime.controls(session_id))['values']['plan']
            assert plan.get('pending') or plan['active']
            await runtime.command(session_id, '/plan off', [])
            await runtime.command(session_id, '/permission read-only', [])
            assert (await runtime.controls(session_id))['values']['permissions']['currentValue'] == 'read-only'
            await runtime.command(session_id, '/permission workspace-write', [])
            async def chunks():
                yield b'staged-file-content'
            upload = await runtime.upload(session_id, 'note.txt', chunks())
            assert upload['receiptId'] and upload['file']['bytes'] == len(b'staged-file-content')
            await runtime.rpc('session/selectModel', {'request': {'sessionId': session_id, 'provider': 'tyloinfer-mtp-off', 'model': 'agent-test', 'reasoningEffort': 'off'}})
            await runtime.rpc('session/prompt', {'request': {'sessionId': session_id, 'requestId': str(uuid4()),
                'mode': 'queue', 'content': [{'type': 'text', 'text': 'Read marker.txt and report the marker.'},
                    {'type': 'file', 'receiptId': upload['receiptId']}]}})
            records = []
            async def answer_question():
                async with asyncio.timeout(12):
                    while not runtime.questions:
                        await asyncio.sleep(0.01)
                    with pytest.raises(AgentRuntimeError, match='running agent turn'):
                        await runtime.start(f'http://127.0.0.1:{port}/v1', 'test-inference-token', [])
                    await runtime.answer(next(iter(runtime.questions)), [{'id': 'continue', 'selected': ['Yes']}])
            answering = asyncio.create_task(answer_question())
            try:
                async with asyncio.timeout(15):
                    async for frame in runtime.stream('session/follow', {'request': {
                            'address': {'kind': 'session', 'sessionId': session_id}, 'assistantStream': True}}):
                        records.extend(frame.get('records', [frame]))
                        if 'Agent integration OK' in json.dumps(records):
                            break
                        assert not runtime.approvals, 'A read inside the workspace should not need approval'
            except TimeoutError:
                pytest.fail(json.dumps({'calls': len(calls), 'events': [item.get('event', {}).get('type', item.get('type')) for item in records], 'logs': list(runtime.logs)}, ensure_ascii=False))
            await answering
            assert len(calls) == 3
            assert 'tool/result' in json.dumps(records)
            exported = await runtime.export(session_id)
            try:
                archive = await exported.aread()
                assert archive.startswith(b'PK')
            finally:
                await exported.aclose()
            assert runtime.process.pid == pid
            state = await runtime.state()
            assert state['workspaces'][0]['workspaceId'] == workspace_id
            assert session_id in state['workspaces'][0]['sessionIds']
            assert 'test-inference-token' not in json.dumps(state)
            async with asyncio.timeout(5):
                while True:
                    matches = await runtime.rpc('session/search', {'request': {'query': 'report the marker'}})
                    if session_id in {item['sessionId'] for item in matches['items']}:
                        break
                    await asyncio.sleep(0.02)
            async with asyncio.timeout(5):
                while any(item['running'] for item in (await runtime.state())['sessions']):
                    await asyncio.sleep(0.02)
            await runtime.rpc('workspace/archiveSession', {'request': {'sessionId': session_id}})
            async with asyncio.timeout(5):
                while session_id not in (await runtime.state())['archivedSessionIds']:
                    await asyncio.sleep(0.02)
            await runtime.rpc('workspace/unarchiveSession', {'request': {'sessionId': session_id}})
            async with asyncio.timeout(5):
                while session_id in (await runtime.state())['archivedSessionIds']:
                    await asyncio.sleep(0.02)
            new_blank = (await runtime.connect_workspace(workspace_id))['sessionId']
            assert new_blank != session_id
            assert (await runtime.connect_workspace(workspace_id))['sessionId'] == new_blank
            await runtime.close()
            await runtime.start(f'http://127.0.0.1:{port}/v1', 'test-inference-token', models)
            restored = await runtime.state()
            assert session_id in [item['sessionId'] for item in restored['sessions']]
            restored_blank = (await runtime.connect_workspace(workspace_id))['sessionId']
            assert restored_blank == new_blank, json.dumps({'sessions': [{key: value for key, value in item.items() if key in {'sessionId', 'blank', 'cwd'}} for item in restored['sessions']], 'workspace': restored['workspaces']})
            selection = (await runtime.controls(session_id))['values']['modelSelection']['next']
            assert selection['provider'] == 'tyloinfer-mtp-off' and selection['reasoningEffort'] == 'off'
            async for snapshot in runtime.stream('session/follow', {'request': {'address': {'kind': 'session', 'sessionId': session_id}, 'assistantStream': True}}):
                assert 'Agent integration OK' in json.dumps(snapshot)
                break
            runtime.process.stdin.close()
            await asyncio.wait_for(runtime.process.wait(), 8)
        finally:
            await runtime.close()
            server.should_exit = True
            await task
            sock.close()
    asyncio.run(run())
