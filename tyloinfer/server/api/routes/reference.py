from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Query

from tyloinfer.server.api.dependencies import ServiceDependency
from tyloinfer.server.api.routes import ERROR_RESPONSES
from tyloinfer.server.protocol.reference import (
    ReferenceFile, ReferenceFiles, ReferenceSource, ReferenceSourceRequest,
    SourceLoadingRequest, SourceLoadingPlan,
)
from tyloinfer.server.services.jobs import JobExecutionError
from tyloinfer.server.services.service import ServiceError

router = APIRouter(tags=['evaluations'], responses=ERROR_RESPONSES)


def resolve(service, value):
    if service.tool_handlers is None:
        raise ServiceError(501, 'reference_unavailable', 'Reference generation is unavailable on this server')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else service.tool_handlers.root / path).resolve()


async def invoke(function, *args, **kwargs):
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except JobExecutionError as error:
        raise ServiceError(422, error.detail.code, error.detail.message, details=error.detail.details) from error
    except (ValueError, OSError, KeyError) as error:
        raise ServiceError(422, 'reference_validation_failed', str(error)) from error


@router.get('/api/v1/evaluations/reference/workspace')
async def workspace(service: ServiceDependency):
    resolve(service, '.')
    return {'import_directory': str(service.tool_handlers.model_root)}


@router.get('/api/v1/evaluations/reference/files', response_model=ReferenceFiles)
async def files(service: ServiceDependency, path: str | None = Query(default=None, max_length=4096)):
    directory = resolve(service, path or str(service.tool_handlers.model_root) if service.tool_handlers else '.')

    def listing():
        if not directory.is_dir():
            raise ValueError('Directory does not exist')
        items = []
        for item in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold())):
            if item.name.startswith('.'):
                continue
            try:
                is_directory = item.is_dir()
                items.append(ReferenceFile(name=item.name, path=str(item), directory=is_directory,
                    byte_size=None if is_directory else item.stat().st_size))
            except OSError:
                continue
            if len(items) >= 4096:
                break
        return ReferenceFiles(path=str(directory), parent=str(directory.parent) if directory.parent != directory else None, data=items)

    return await invoke(listing)


@router.post('/api/v1/evaluations/reference/source', response_model=ReferenceSource)
async def source(service: ServiceDependency, body: ReferenceSourceRequest):
    from tyloinfer.evaluation.source import source_summary
    return await invoke(source_summary, resolve(service, body.path))


@router.post('/api/v1/evaluations/reference/loading-plan', response_model=SourceLoadingPlan)
async def loading_plan(service: ServiceDependency, body: SourceLoadingRequest):
    from tyloinfer.evaluation.source import loading_plan as plan
    return await invoke(plan, resolve(service, body.path), **body.model_dump(exclude={'path'}))
