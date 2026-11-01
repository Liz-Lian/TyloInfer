from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from tyloinfer.server.api.dependencies import ServiceDependency
from tyloinfer.server.services.agent_runtime import AgentRuntimeError
from tyloinfer.server.services.service import ServiceError

router = APIRouter(prefix='/api/v1/agent', tags=['agent'])


class AgentRpcRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    method: Literal['workspace/create', 'workspace/rename', 'workspace/delete', 'workspace/archiveSession', 'workspace/unarchiveSession',
        'session/create', 'session/rename', 'session/selectModel', 'session/prompt', 'session/cancel', 'session/page']
    args: dict[str, Any] = Field(default_factory=dict)


class AgentApprovalRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    event_id: str = Field(min_length=1, max_length=255)
    allow: bool


class AgentAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=255)
    selected: list[str] = Field(default_factory=list, max_length=100)
    custom: str = Field(default='', max_length=32000)


class AgentAnswerRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    event_id: str = Field(min_length=1, max_length=255)
    answers: list[AgentAnswer] = Field(min_length=1, max_length=100)


class AgentCommandRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    line: str = Field(min_length=1, max_length=32000, pattern=r'^/')
    attachments: list[dict[str, Any]] = Field(default_factory=list, max_length=20)


@router.get('/state')
async def agent_state(service: ServiceDependency) -> dict[str, Any]:
    try:
        return await service.agent_runtime.state()
    except AgentRuntimeError as error:
        raise ServiceError(503, 'agent_unavailable', str(error), retryable=True) from error


@router.get('/search')
async def search_agent_sessions(service: ServiceDependency, query: str = Query(min_length=1, max_length=500)) -> dict[str, Any]:
    try:
        return await service.agent_runtime.rpc('session/search', {'request': {'query': query.strip()}})
    except AgentRuntimeError as error:
        raise ServiceError(400, 'agent_search_failed', str(error), retryable=True) from error


@router.post('/start')
async def start_agent(request: Request, service: ServiceDependency) -> dict[str, Any]:
    instances = (await service.runtime_instances()).data
    async def describe(item):
        try:
            status = await service.runtime_status(item.id)
        except ServiceError:
            status = {}
        try:
            model_capabilities = await service.runtime_capabilities(item.id)
            vision = model_capabilities.model_capabilities.features.image_input
        except ServiceError:
            vision = False
        capabilities = status.get('chat_template_capabilities', {})
        efforts = {value: value for value in capabilities.get('reasoning_effort', {}).get('values', [])
            if value in {'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}}
        if capabilities.get('thinking', {}).get('supported'):
            efforts = {'off': 'none', **(efforts or {'high': 'high'})}
        return {'id': item.model, 'name': item.model, 'contextWindow': item.context_size or 32768,
            'maxTokens': min(8192, max(512, (item.context_size or 32768) // 16)), 'input': ['text', 'image'] if vision else ['text'],
            'reasoningEfforts': efforts or False, 'mtpAvailable': item.mtp_available and status.get('mtp_service_enabled', True),
            'thinkingToggleOnly': len(efforts) == 2 and set(efforts) == {'off', 'high'}}
    models = await asyncio.gather(*(describe(item) for item in instances if item.state == 'ready'))
    port = request.scope.get('server', ('127.0.0.1', 8090))[1]
    try:
        await service.agent_runtime.start(f'http://127.0.0.1:{port}/v1', request.app.state.agent_token, models)
        return await service.agent_runtime.state()
    except AgentRuntimeError as error:
        raise ServiceError(503, 'agent_start_failed', str(error), retryable=True) from error


@router.post('/rpc')
async def agent_rpc(payload: AgentRpcRequest, service: ServiceDependency) -> Any:
    try:
        return await service.agent_runtime.rpc(payload.method, payload.args)
    except AgentRuntimeError as error:
        raise ServiceError(400, 'agent_operation_failed', str(error), retryable=True) from error


@router.post('/workspaces/{workspace_id}/session')
async def connect_workspace(workspace_id: str, service: ServiceDependency) -> dict[str, Any]:
    try:
        return await service.agent_runtime.connect_workspace(workspace_id)
    except AgentRuntimeError as error:
        raise ServiceError(400, 'agent_workspace_unavailable', str(error), retryable=True) from error


@router.get('/sessions/{session_id}/controls')
async def agent_controls(session_id: str, service: ServiceDependency) -> dict[str, Any]:
    try:
        return await service.agent_runtime.controls(session_id)
    except AgentRuntimeError as error:
        raise ServiceError(400, 'agent_controls_unavailable', str(error), retryable=True) from error


@router.post('/sessions/{session_id}/commands')
async def agent_command(session_id: str, payload: AgentCommandRequest, service: ServiceDependency) -> dict[str, Any]:
    try:
        return await service.agent_runtime.command(session_id, payload.line, payload.attachments)
    except AgentRuntimeError as error:
        raise ServiceError(400, 'agent_command_failed', str(error), retryable=True) from error


@router.post('/sessions/{session_id}/files')
async def upload_agent_file(session_id: str, request: Request, service: ServiceDependency, name: str = 'attachment') -> dict[str, Any]:
    try:
        return await service.agent_runtime.upload(session_id, name, request.stream())
    except AgentRuntimeError as error:
        raise ServiceError(400, 'agent_upload_failed', str(error), retryable=True) from error


@router.get('/sessions/{session_id}/export')
async def export_agent_session(session_id: str, service: ServiceDependency) -> StreamingResponse:
    try:
        response = await service.agent_runtime.export(session_id)
    except (AgentRuntimeError, OSError) as error:
        raise ServiceError(400, 'agent_export_failed', 'The conversation could not be exported.', retryable=True) from error
    async def chunks():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
    return StreamingResponse(chunks(), media_type='application/zip', headers={'Content-Disposition': 'attachment; filename="conversation.zip"', 'Cache-Control': 'no-store'})


@router.post('/approvals')
async def agent_approval(payload: AgentApprovalRequest, service: ServiceDependency) -> dict[str, bool]:
    try:
        await service.agent_runtime.approve(payload.event_id, payload.allow)
        return {'accepted': True}
    except AgentRuntimeError as error:
        raise ServiceError(409, 'agent_approval_expired', str(error)) from error


@router.get('/sessions/{session_id}/events')
async def agent_events(session_id: str, service: ServiceDependency) -> StreamingResponse:
    if not service.agent_runtime.client:
        raise ServiceError(503, 'agent_unavailable', 'DeepSeek Harness is not running.', retryable=True)

    async def events():
        try:
            async for item in service.agent_runtime.stream('session/follow', {
                    'request': {'address': {'kind': 'session', 'sessionId': session_id}, 'assistantStream': True}}):
                yield json.dumps(item, ensure_ascii=False) + '\n'
        except (AgentRuntimeError, OSError):
            yield json.dumps({'type': 'connection-error', 'message': 'The agent connection was interrupted.'}) + '\n'

    return StreamingResponse(events(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})


@router.post('/answers')
async def answer_agent(payload: AgentAnswerRequest, service: ServiceDependency) -> dict[str, bool]:
    try:
        await service.agent_runtime.answer(payload.event_id, [answer.model_dump() for answer in payload.answers])
        return {'accepted': True}
    except AgentRuntimeError as error:
        raise ServiceError(409, 'agent_answer_rejected', str(error)) from error
