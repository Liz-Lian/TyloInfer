from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tyloinfer.formats.assets import is_asset_record
from tyloinfer.formats.companions import is_engram_table
from tyloinfer.formats.gguf import gguf_part_paths, gguf_table_metadata, is_gguf_path, parse_gguf_shard_path
from tyloinfer.formats.io import _open_single_mmap
from tyloinfer.formats.shards import parse_shard_path
from tyloinfer.server.protocol.models import ModelArtifactResource


@dataclass(frozen=True)
class TableComponent:
    path: Path
    files: tuple[Path, ...]
    kind: str
    architecture: str
    tensor_count: int
    dtypes: tuple[str, ...]
    mixed_kinds: tuple[str, ...] = ()


def table_record_kind(name: str) -> str | None:
    if is_engram_table(name):
        return 'engram'
    if re.search(r'\.position_embedding\.ngram\.shard\.\d+\.(?:weight|scales|biases)(?:\.tq_row_shard\.\d+)?$', name):
        return 'ple'
    return None


def inspect_table_component(path: Path, *, include_mixed: bool = False) -> TableComponent | None:
    if is_gguf_path(path):
        metadata = gguf_table_metadata(path, include_mixed=include_mixed)
        if metadata is None:
            return None
        return TableComponent(path, gguf_part_paths(path), metadata['kind'], metadata['architecture'], metadata['tensor_count'], (), metadata['mixed_kinds'])
    header, records, file, mapping = _open_single_mmap(path)
    try:
        tensors = [record for record in records.values() if not is_asset_record(record.name)]
        kinds = {table_record_kind(record.name) for record in tensors}
        table_kinds = kinds - {None}
        if not tensors or None in kinds or len(kinds) != 1:
            return TableComponent(path, (path,), 'mixed', header.model_arch or 'unknown', len(tensors), (),
                tuple(sorted(table_kinds))) if include_mixed and table_kinds else None
        return TableComponent(path, (path,), kinds.pop(), header.model_arch or 'unknown', len(tensors), tuple(sorted({record.dtype for record in tensors})))
    finally:
        mapping.close()
        file.close()


def component_base(path: Path) -> Path:
    if path.suffix.casefold() == '.safetensors':
        return path.parent
    parsed = parse_gguf_shard_path(path) if is_gguf_path(path) else parse_shard_path(path)
    return parsed[0] if parsed else path


def component_resource(parts: list[TableComponent]) -> tuple[ModelArtifactResource, tuple[Path, ...]]:
    first = parts[0]
    base = component_base(first.path)
    files = tuple(sorted({path.resolve() for part in parts for path in part.files}))
    stats = [path.stat() for path in files]
    identity = '\0'.join(f'{path}:{info.st_size}:{info.st_mtime_ns}' for path, info in zip(files, stats, strict=True))
    identifier = hashlib.sha256(f'component\0{first.kind}\0{identity}'.encode()).hexdigest()[:32]
    label = 'Engram' if first.kind == 'engram' else 'PLE'
    is_hf = first.path.suffix.casefold() == '.safetensors'
    name = base.name if is_hf else base.stem
    if label.casefold() not in name.casefold():
        name += '-' + label
    resource = ModelArtifactResource(id=identifier, name=name, architecture=first.architecture, asset_kind=first.kind,
        format='hf' if is_hf else 'gguf' if is_gguf_path(first.path) else 'tq', shard_count=len(files), total_bytes=sum(info.st_size for info in stats),
        owned_file_bytes=sum(info.st_size for info in stats), tensor_count=sum(part.tensor_count for part in parts),
        record_count=sum(part.tensor_count for part in parts), dtypes=sorted({dtype for part in parts for dtype in part.dtypes}),
        complete=True, loadable=False, modified_at=datetime.fromtimestamp(max(info.st_mtime for info in stats), timezone.utc))
    return resource, files
