from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import sys
from uuid import uuid4
from collections import deque
from contextlib import aclosing
from pathlib import Path
from typing import Any

import httpx
from websockets.asyncio.client import connect


class AgentRuntimeError(RuntimeError):
    pass


class AgentRuntime:
    def __init__(self, home: Path) -> None:
        self.home = home.resolve()
        self.process: asyncio.subprocess.Process | None = None
        self.client: httpx.AsyncClient | None = None
        self.cookie = ''
        self.url = ''
        self.error: str | None = None
        self.logs: deque[str] = deque(maxlen=30)
        self.workspaces: list[dict[str, Any]] = []
        self.archived_session_ids: list[str] = []
        self.approvals: dict[str, dict[str, Any]] = {}
        self.questions: dict[str, dict[str, Any]] = {}
        self.tasks: set[asyncio.Task] = set()
        self.lock = asyncio.Lock()
        self.signature = ''
        self.models: list[dict[str, Any]] = []

    @staticmethod
    def component() -> tuple[str, Path]:
        root = Path(__file__).resolve().parents[3]
        directories = [Path(os.environ['TYLOINFER_AGENT_RUNTIME'])] if os.environ.get('TYLOINFER_AGENT_RUNTIME') else []
        directories += [root / 'agent-runtime', root / 'build' / 'agent-runtime']
        if getattr(sys, 'frozen', False):
            directories.insert(0, Path(sys.executable).parent.parent / 'agent-runtime')
        for directory in directories:
            cli = directory / 'node_modules' / '@deepseek-ai' / 'dsh' / 'lib' / 'bin.js'
            node = directory / ('node.exe' if os.name == 'nt' else 'node')
            executable = str(node) if node.is_file() else os.environ.get('TYLOINFER_AGENT_NODE') or shutil.which('node')
            if executable and cli.is_file():
                return executable, cli
        raise AgentRuntimeError('The bundled DeepSeek Harness runtime is not installed.')

    def _track(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _drain(self, stream, startup: asyncio.Future) -> None:
        while line := await stream.readline():
            text = line.decode('utf-8', errors='replace').strip()
            match = re.search(r'http://127\.0\.0\.1:(\d+)/\?token=([A-Za-z0-9_-]+)', text)
            if match and not startup.done():
                startup.set_result((f'http://127.0.0.1:{match[1]}', match[2]))
            elif 'token=' not in text:
                self.logs.append(text)

    async def _watch_exit(self, process: asyncio.subprocess.Process, startup: asyncio.Future) -> None:
        status = await process.wait()
        message = f'DeepSeek Harness exited with status {status}.'
        if not startup.done():
            startup.set_exception(AgentRuntimeError(message))
        elif self.process is process:
            self.error = message

    async def start(self, base_url: str, token: str, models: list[dict[str, Any]]) -> None:
        async with self.lock:
            signature = json.dumps([base_url, models], sort_keys=True)
            if self.process and self.process.returncode is None and signature == self.signature:
                return
            if self.client and self.process and self.process.returncode is None:
                sessions = await self.rpc('session/list', {'_request': {}})
                if any(item.get('running') for item in sessions.get('items', [])):
                    raise AgentRuntimeError('Wait for the running agent turn before changing model routes.')
            await self.close()
            executable, cli = self.component()
            self.home.mkdir(parents=True, exist_ok=True)
            run = self.home / 'run'
            run.mkdir(exist_ok=True)
            provider = {'displayName': 'TyloInfer', 'api': 'openai-completions',
                'baseURL': base_url, 'apiKeyEnv': 'TYLOINFER_AGENT_TOKEN',
                'compat': {'supportsStore': False}, 'models': [{key: value for key, value in model.items()
                    if key in {'id', 'name', 'contextWindow', 'maxTokens', 'input', 'reasoningEfforts'}} for model in models]}
            providers = {name: {**provider, **options} for name, options in {
                'tyloinfer': {}, 'tyloinfer-mtp-on': {'headers': {'X-TyloInfer-MTP': 'on'}},
                'tyloinfer-mtp-off': {'headers': {'X-TyloInfer-MTP': 'off'}}}.items()} if models else {}
            patch = [{'id': 'llm-pi-ai', 'config': {'providers': providers}},
                {'id': 'session-query-sqlite', 'config': {'path': str(self.home / 'session-search.sqlite3'), 'openAt': 'first-search'}}]
            if models:
                patch.append({'id': 'agent-default-model', 'config': {'provider': 'tyloinfer', 'model': models[0]['id']}})
            profile = self.home / 'tyloinfer.patch.yml'
            profile.write_text(json.dumps(patch), encoding='utf-8')
            profile.chmod(0o600)
            env = {key: value for key, value in os.environ.items() if not re.search(r'KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH', key, re.I)}
            env.update(DSH_HOME=str(self.home / 'dsh'), DSH_TELEMETRY_DISABLED='1',
                TYLOINFER_AGENT_TOKEN=token, NODE_USE_SYSTEM_CA='1')
            startup = asyncio.get_running_loop().create_future()
            try:
                launcher = "process.stdin.resume();process.stdin.on('end',()=>process.kill(process.platform==='win32'?process.pid:-process.pid,'SIGTERM'));import(require('node:url').pathToFileURL(process.argv[1]).href).then(({runCli})=>runCli()).catch(error=>{console.error(error);process.exit(1)})"
                self.process = await asyncio.create_subprocess_exec(executable, '--eval', launcher, str(cli), '--profile', 'web',
                    '--patch', str(profile), '--port', '0', '--no-open', cwd=run, env=env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    start_new_session=os.name != 'nt')
                self._track(self._drain(self.process.stdout, startup))
                self._track(self._drain(self.process.stderr, startup))
                self._track(self._watch_exit(self.process, startup))
                self.url, launch_token = await asyncio.wait_for(startup, 90)
                self.client = httpx.AsyncClient(base_url=self.url, timeout=30, trust_env=False)
                response = await self.client.get('/', params={'token': launch_token}, follow_redirects=False)
                cookies = [f'{name}={value}' for name, value in response.cookies.items() if name.startswith('dsh-auth-')]
                if not cookies:
                    raise AgentRuntimeError('DeepSeek Harness refused its launch credential.')
                self.cookie = '; '.join(cookies)
                self.client.headers['Cookie'] = self.cookie
                self.models, self.signature, self.error = models, signature, None
                async with aclosing(self.stream('workspace/follow', {})) as frames:
                    baseline = await anext(frames)
                self.workspaces = baseline['value']['items']
                self.archived_session_ids = baseline['value'].get('archivedSessionIds', [])
                self._track(self._follow(self._workspace_changes))
                self._track(self._follow(self._host_events))
            except Exception as error:
                await self.close()
                self.error = str(error) if isinstance(error, AgentRuntimeError) else 'DeepSeek Harness could not start. Check the agent runtime installation.'
                if any('No usable native binding' in line for line in self.logs):
                    self.error = 'DeepSeek Harness could not load its native components. Reinstall the agent runtime.'
                raise AgentRuntimeError(self.error) from error

    async def rpc(self, method: str, args: dict[str, Any]) -> Any:
        if not self.client or not self.process or self.process.returncode is not None:
            raise AgentRuntimeError('DeepSeek Harness is not running.')
        try:
            response = await self.client.post(f'/api/{method}', json={
                'type': 'client-request', 'rpcId': str(uuid4()), 'method': method, 'payload': {'args': args}})
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise AgentRuntimeError('The agent connection is unavailable. Reconnect to continue.') from error
        try:
            result = response.json()['result']
        except (ValueError, KeyError) as error:
            raise AgentRuntimeError('DeepSeek Harness returned an invalid response.') from error
        if not result.get('ok'):
            raise AgentRuntimeError(result.get('error', {}).get('message', 'The agent operation failed.'))
        return result.get('value')

    async def stream(self, endpoint: str, args: dict[str, Any]):
        if not self.client:
            raise AgentRuntimeError('DeepSeek Harness is not running.')
        async with connect(self.url.replace('http:', 'ws:') + '/api/remote.mux',
                additional_headers={'Cookie': self.cookie}, max_size=32 << 20, proxy=None) as socket:
            await socket.send(json.dumps({'type': 'open', 'streamId': 'agent', 'endpoint': endpoint, 'payload': {'args': args}}))
            async for raw in socket:
                frame = json.loads(raw)
                if frame.get('type') == 'item':
                    yield frame.get('value')
                elif frame.get('type') == 'error':
                    raise AgentRuntimeError(frame.get('error', {}).get('message', 'The agent stream failed.'))
                elif frame.get('type') == 'end':
                    break

    async def _follow(self, listener) -> None:
        while self.process and self.process.returncode is None:
            try:
                await listener()
            except Exception:
                self.error = 'The agent connection was interrupted. Reconnecting…'
            await asyncio.sleep(2)

    async def _workspace_changes(self) -> None:
        async for frame in self.stream('workspace/follow', {}):
            self.error = None
            kind = frame.get('type')
            if kind in {'baseline', 'replace'}:
                self.workspaces = frame['value']['items']
                self.archived_session_ids = frame['value'].get('archivedSessionIds', [])
            elif kind == 'upsert':
                item = frame['workspace']
                self.workspaces = [value for value in self.workspaces if value['workspaceId'] != item['workspaceId']] + [item]
            elif kind == 'remove':
                self.workspaces = [value for value in self.workspaces if value['workspaceId'] != frame['workspaceId']]
            elif kind == 'order':
                order = {value: index for index, value in enumerate(frame['workspaceIds'])}
                self.workspaces.sort(key=lambda item: order.get(item['workspaceId'], len(order)))
            elif kind == 'archived':
                self.archived_session_ids = frame['archivedSessionIds']

    async def _host_events(self) -> None:
        client_id = None
        self.approvals.clear()
        self.questions.clear()
        async for event in self.stream('$events', {}):
            if event.get('type') == 'ready':
                client_id = event['clientId']
            elif event.get('type') == 'waterfall':
                if event.get('event') == 'approval/request':
                    self.approvals[event['eventId']] = dict(event, clientId=client_id)
                elif event.get('event') == 'user-questions/request':
                    self.questions[event['eventId']] = dict(event, clientId=client_id)
                else:
                    await self.rpc('$events/result', {'clientId': client_id, 'eventId': event['eventId'], 'outcome': {'kind': 'next'}})
            elif event.get('type') == 'cancel':
                self.approvals.pop(event.get('eventId'), None)
                self.questions.pop(event.get('eventId'), None)

    async def state(self) -> dict[str, Any]:
        running = bool(self.client and self.process and self.process.returncode is None)
        sessions = (await self.rpc('session/list', {'_request': {}})).get('items', []) if running else []
        return {'running': running, 'error': self.error, 'workspaces': self.workspaces if running else [],
            'sessions': sessions, 'models': self.models if running else [],
            'archivedSessionIds': self.archived_session_ids if running else [],
            'questions': [{key: value for key, value in item.items() if key != 'clientId'} for item in self.questions.values()],
            'approvals': [{key: value for key, value in item.items() if key != 'clientId'} for item in self.approvals.values()]}

    async def controls(self, session_id: str) -> dict[str, Any]:
        projections, commands, permissions = await asyncio.gather(
            self.rpc('session/projections', {'request': {'sessionId': session_id}}),
            self.rpc('commands/list', {'agentId': session_id}),
            self.rpc('permissionPresets/catalog', {}))
        return {'values': projections['values'], 'commands': commands, 'permissions': permissions['options']}

    async def command(self, session_id: str, line: str, attachments: list[dict[str, Any]]) -> dict[str, Any]:
        result = await self.rpc('commands/execute', {'agentId': session_id, 'line': line, 'submittedAttachments': attachments})
        if result.get('result', {}).get('kind') == 'error':
            raise AgentRuntimeError(result['result']['text'])
        return result

    async def upload(self, session_id: str, name: str, chunks) -> dict[str, Any]:
        if not self.client:
            raise AgentRuntimeError('DeepSeek Harness is not running.')
        received = 0
        async def bounded():
            nonlocal received
            async for chunk in chunks:
                received += len(chunk)
                if received > 64 << 20:
                    raise AgentRuntimeError('Each file must be no larger than 64 MiB.')
                yield chunk
        try:
            response = await self.client.post('/api/session/uploadFileBinary', params={'sessionId': session_id, 'name': name},
                content=bounded(), headers={'Content-Type': 'application/octet-stream'})
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise AgentRuntimeError('The file could not be uploaded. Please retry.') from error
        if not result.get('ok'):
            raise AgentRuntimeError(result.get('error', {}).get('message', 'The file could not be uploaded.'))
        return result['value']

    async def export(self, session_id: str) -> httpx.Response:
        if not self.client:
            raise AgentRuntimeError('DeepSeek Harness is not running.')
        try:
            response = await self.client.send(self.client.build_request('GET', '/api/session.export', params={'sessionId': session_id}), stream=True)
        except httpx.HTTPError as error:
            raise AgentRuntimeError('This conversation could not be exported. Please retry.') from error
        if response.is_error:
            await response.aclose()
            raise AgentRuntimeError('This conversation could not be exported. Please retry.')
        return response

    async def connect_workspace(self, workspace_id: str) -> dict[str, Any]:
        async with self.lock:
            try:
                async with asyncio.timeout(15):
                    async with aclosing(self.stream('workspace/follow', {})) as frames:
                        snapshot = await anext(frames)
                    listing = snapshot['value']
                    workspace = next((item for item in listing['items'] if item['workspaceId'] == workspace_id), None)
                    if workspace is None:
                        raise AgentRuntimeError('This workspace is no longer available.')
                    sessions = (await self.rpc('session/list', {'_request': {}}))['items']
                    archived = set(listing.get('archivedSessionIds', []))
                    blank = None
                    for item in sessions:
                        if item.get('cwd') != workspace['path'] or item['sessionId'] not in workspace['sessionIds'] or item['sessionId'] in archived:
                            continue
                        is_blank = item.get('blank', False)
                        metadata = item.get('projections', {}).get('values', {}).get('sessionListMetadata')
                        if not is_blank and not item.get('agentAvailable', True) and metadata is None:
                            projections = await self.rpc('session/projections', {'request': {'sessionId': item['sessionId']}})
                            is_blank = (projections or {}).get('values', {}).get('sessionListMetadata', {}).get('blank', False)
                        if is_blank:
                            blank = item
                            break
                    request = {'workspaceId': workspace_id}
                    if blank:
                        request['sessionId'] = blank['sessionId']
                    result = await self.rpc('session/create', {'request': request})
                    async with aclosing(self.stream('workspace/follow', {})) as frames:
                        snapshot = await anext(frames)
                    self.workspaces = snapshot['value']['items']
                    self.archived_session_ids = snapshot['value'].get('archivedSessionIds', [])
                    return result
            except TimeoutError as error:
                raise AgentRuntimeError('The workspace could not be opened in time. Please retry.') from error

    async def approve(self, event_id: str, allow: bool) -> None:
        item = self.approvals.get(event_id)
        if not item:
            raise AgentRuntimeError('This approval is no longer pending.')
        await self.rpc('$events/result', {'clientId': item['clientId'], 'eventId': event_id,
            'outcome': {'kind': 'result', 'value': 'allowed-once' if allow else 'rejected'}})
        self.approvals.pop(event_id, None)

    async def answer(self, event_id: str, answers: list[dict[str, Any]]) -> None:
        item = self.questions.get(event_id)
        if not item:
            raise AgentRuntimeError('This question is no longer pending.')
        expected = {question['id']: question for question in item['request']['questions']}
        if len(answers) != len(expected) or {answer['id'] for answer in answers} != set(expected):
            raise AgentRuntimeError('Answer each question exactly once.')
        for answer in answers:
            question = expected[answer['id']]
            labels = {option['label'] for option in question.get('options', [])}
            if not set(answer['selected']) <= labels or (not question.get('multiSelect') and len(answer['selected']) > 1):
                raise AgentRuntimeError('The selected answer is not a valid option.')
            if not answer['selected'] and not answer.get('custom', '').strip():
                raise AgentRuntimeError('Choose an option or enter an answer.')
        await self.rpc('$events/result', {'clientId': item['clientId'], 'eventId': event_id,
            'outcome': {'kind': 'result', 'value': {'answers': answers}}})
        self.questions.pop(event_id, None)

    async def close(self) -> None:
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        if self.client:
            await self.client.aclose()
        self.client = None
        if self.process and self.process.returncode is None:
            self._signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), 8)
            except TimeoutError:
                self._signal(9)
                await self.process.wait()
        self.process = None
        self.approvals.clear()
        self.questions.clear()
        self.archived_session_ids.clear()

    def _signal(self, sig: int) -> None:
        try:
            if os.name != 'nt':
                os.killpg(self.process.pid, sig)
            elif sig == signal.SIGTERM:
                self.process.terminate()
            else:
                self.process.kill()
        except ProcessLookupError:
            pass
