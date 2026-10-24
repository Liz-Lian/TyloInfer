from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from tyloinfer.server.protocol.models import ProtocolModel


class ReferenceSourceRequest(ProtocolModel):
    path: str = Field(min_length=1, max_length=4096)


class SourceLoadingRequest(ReferenceSourceRequest):
    purpose: Literal['wt2'] = 'wt2'
    backend: Literal['auto', 'metal', 'cuda'] = 'auto'
    context_size: int = Field(default=512, ge=2, le=1048576)
    batch_size: int = Field(default=1, ge=1, le=64)


class SourceLoadingPlan(ProtocolModel):
    backend: str
    resident_required_bytes: int
    available_bytes: int
    resident_allowed: bool
    weights_bytes: int
    workspace_bytes: int


class Wt2ReferencePayload(ProtocolModel):
    model: str = Field(min_length=1, max_length=4096)
    dataset_id: UUID
    output: str = Field(min_length=1, max_length=4096)
    backend: Literal['auto', 'metal', 'cuda'] = 'auto'
    context_size: int = Field(default=512, ge=32, le=1048576)
    chunks: int = Field(default=8, ge=1, le=10000)
    layerwise: bool = True


class ReferenceSource(ProtocolModel):
    path: str
    architecture: str
    tensors: int
    parameters: int
    format: Literal['hf']
    full_precision: bool
    source_precisions: list[str] = Field(default_factory=list)
    reference_supported: bool = False


class ReferenceFile(ProtocolModel):
    name: str
    path: str
    directory: bool
    byte_size: int | None = None


class ReferenceFiles(ProtocolModel):
    path: str
    parent: str | None
    data: list[ReferenceFile]
