"""MFQ model discovery without exposing host filesystem paths."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import struct
import sys
from dataclasses import dataclass, replace
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from tyloinfer.architectures.tensor_schema import (
    graph_spec_for_source_names,
    map_source_tensor_name,
)
from tyloinfer.formats.assets import is_asset_record, MODEL_CONFIG_ASSET, HF_TOKENIZER_CONFIG_ASSET
from tyloinfer.formats.compat import canonical_dtype
from tyloinfer.formats.header import TYLOINFER_SUFFIXES
from tyloinfer.formats.io import open_mmap
from tyloinfer.formats.shards import matching_shard_paths as _matching_tq_shards, parse_shard_path as _parse_tq_shard
from tyloinfer.formats.gguf import gguf_is_table_component, gguf_part_paths, is_gguf_path, matching_gguf_shard_paths, parse_gguf_shard_path
from tyloinfer.server.protocol.models import (
    ModelArtifactList,
    ModelArtifactResource,
    ModelDirectoryEntry,
    ModelDirectoryFile,
    ModelDirectoryList,
)
from tyloinfer.server.state.model_weights import estimated_resident_weight_bytes, estimated_weight_roles, weight_role
from tyloinfer.server.services.model_context import declared_context_size, yarn_context_limits
from tyloinfer.server.services.desktop import directory_opener
from tyloinfer.server.state.table_components import TableComponent, component_base, component_resource, inspect_table_component, table_record_kind

MODEL_FILE_INDEX = ".tq-files.json"
_MODEL_SUFFIXES = (*TYLOINFER_SUFFIXES, '.gguf')
_ROUTED_EXPERT_RE = re.compile(
    r"\.mlp\.experts\.(?:\d+\.)?"
    r"(?:gate|up|down|gate_up)(?:_proj)?\."
    r"(?:weight|weight_scale|scales|biases)$"
)
_ALWAYS_STREAMED_RE = re.compile(
    r"(?:\.associative_memory\.embedding\.weight|\.engram\.embed\.weight"
    r"|\.position_embedding\.ngram\.shard\.\d+\.(?:weight|scales|biases))$"
)
_LOGGER = logging.getLogger(__name__)


class ModelArtifactNotFoundError(LookupError):
    pass


class DuplicateModelNameError(RuntimeError):
    pass


class ModelDirectoryNotFoundError(LookupError):
    pass


class ModelRegistrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiscoveredModel:
    resource: ModelArtifactResource
    path: Path
    routed_expert_bytes: int = 0
    always_streamed_bytes: int = 0
    source_config: dict | None = None
    source_tensors: dict | None = None
    file_paths: tuple[Path, ...] = ()


def parse_shard_path(path: Path):
    return parse_gguf_shard_path(path) if is_gguf_path(path) else _parse_tq_shard(path)


def matching_shard_paths(path: Path):
    if is_gguf_path(path):
        return matching_gguf_shard_paths(path)
    parsed = _parse_tq_shard(path)
    return _matching_tq_shards(parsed[0] if parsed else path)


def _is_model_path(path: Path) -> bool:
    return path.suffix.casefold() in _MODEL_SUFFIXES or is_gguf_path(path)


def _is_gguf_table(path: Path, cache: dict[Path, bool]) -> bool:
    if not is_gguf_path(path):
        return False
    try:
        first = gguf_part_paths(path, require_complete=False)[0]
        if first not in cache:
            cache[first] = gguf_is_table_component(path)
        return cache[first]
    except (OSError, ValueError):
        return False


def _is_routed_expert_tensor(name: str) -> bool:
    """Identify the canonical expert bank that SSD streaming can replace."""

    return bool(_ROUTED_EXPERT_RE.search(name)) or any(
        marker in name
        for marker in (
            ".ffn_gate_exps.",
            ".ffn_up_exps.",
            ".ffn_gate_up_exps.",
            ".ffn_down_exps.",
        )
    )


def _is_always_streamed_tensor(name: str) -> bool:
    """Identify payloads whose runtime storage is independently bounded."""

    return bool(_ALWAYS_STREAMED_RE.search(name))


def _native_architecture(metadata: dict, fallback: str = 'unknown') -> str:
    config = metadata['config']
    text = config.get('text_config', config)
    kind = text.get('model_type') if isinstance(text, dict) else None
    return kind if isinstance(kind, str) and kind else metadata.get('architecture') or fallback


class ModelCatalog:
    """Discover complete MFQ containers and native HF checkpoints."""

    def __init__(
        self,
        roots: list[str | Path],
        *,
        cache_seconds: float = 5.0,
        browse_roots: list[str | Path] | None = None,
        native_inspector: Callable[[Path], dict] | None = None,
    ) -> None:
        self.roots = tuple(Path(root).expanduser().resolve() for root in roots)
        self.browse_roots = (
            None
            if browse_roots is None
            else tuple(Path(root).expanduser().resolve() for root in browse_roots)
        )
        self.cache_seconds = max(0.0, cache_seconds)
        self._native_inspector = native_inspector
        self._source_metadata: dict[Path, tuple[tuple, dict]] = {}
        self._table_metadata: dict[Path, tuple[tuple, TableComponent | None]] = {}
        self._models: dict[str, DiscoveredModel] = {}
        self._last_scan = float("-inf")
        self._last_scan_attempt = float("-inf")
        self._lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[dict[str, DiscoveredModel]] | None = None
        self._refresh_is_background = False
        self._registration_lock = asyncio.Lock()
        self._directory_ids: dict[str, Path] = {}
        self._directory_secret = os.urandom(32)

    async def list(self, *, refresh: bool = False, include_components: bool = False) -> ModelArtifactList:
        models = await self._snapshot(refresh=refresh)
        return ModelArtifactList(
            data=sorted(
                (item.resource for item in models.values() if include_components or item.resource.asset_kind == 'model'),
                key=lambda item: (item.name.casefold(), item.id),
            )
        )

    async def get(self, model_id: str, *, refresh: bool = False) -> DiscoveredModel:
        models = await self._snapshot(refresh=refresh)
        result = models.get(model_id)
        if result is None and not refresh:
            models = await self._snapshot(refresh=True)
            result = models.get(model_id)
        if result is None:
            raise ModelArtifactNotFoundError(model_id)
        return result

    async def resolve(
        self,
        model: str,
        artifact_uri: str | None = None,
    ) -> DiscoveredModel:
        identifier = model
        if artifact_uri is not None:
            if not artifact_uri.startswith("mfq://"):
                raise ModelArtifactNotFoundError(
                    "only catalog-backed mfq:// artifact URIs are accepted"
                )
            identifier = artifact_uri.removeprefix("mfq://")
        models = await self._snapshot()
        exact = models.get(identifier)
        if exact is not None:
            return exact
        matches = [item for item in models.values() if item.resource.name == identifier and item.resource.asset_kind == 'model']
        if not matches:
            models = await self._snapshot(refresh=True)
            exact = models.get(identifier)
            if exact is not None:
                return exact
            matches = [item for item in models.values() if item.resource.name == identifier and item.resource.asset_kind == 'model']
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ModelArtifactNotFoundError(f"ambiguous model name: {identifier}")
        raise ModelArtifactNotFoundError(identifier)

    async def resolve_path(self, path: str | Path) -> DiscoveredModel:
        """Resolve a configured model path to the catalog artifact that owns it."""

        requested = Path(path).expanduser().resolve()
        target = self._canonical_model_path(requested)

        # ``mfq serve --model /exact/path`` is already a trusted, configured
        # local path. Resolve it before enumerating sibling artifacts so an HF
        # source directory and its converted MFQ (or two quantized variants)
        # may share a display name without preventing exact-path startup.
        privacy_root = next(
            (root for root in self.roots if requested.is_relative_to(root)),
            None,
        )
        if privacy_root is not None:
            inspect_path = requested
            if requested.is_file():
                try:
                    parsed = parse_shard_path(requested)
                except ValueError:
                    parsed = None
                if parsed is not None:
                    first = next(
                        (
                            candidate
                            for candidate in matching_shard_paths(requested)
                            if parse_shard_path(candidate)[1] == 1
                        ),
                        None,
                    )
                    if first is not None:
                        inspect_path = first
            supported_path = (
                inspect_path.is_file() and _is_model_path(inspect_path)
            ) or (inspect_path.is_dir() and self._is_hf_model_directory(inspect_path))
            if supported_path:
                direct = await asyncio.to_thread(self._inspect_source, privacy_root, inspect_path)
                if self._canonical_model_path(direct.path) == target:
                    return direct

        for refresh in (False, True):
            models = await self._snapshot(refresh=refresh)
            matches = [
                item for item in models.values() if self._canonical_model_path(item.path) == target
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ModelArtifactNotFoundError(f"ambiguous model path: {path}")
        raise ModelArtifactNotFoundError(str(path))

    async def browse_directories(
        self,
        directory_id: str | None = None,
        *,
        path: str | Path | None = None,
    ) -> ModelDirectoryList:
        """List server-side directories by opaque identifier or explicit path."""

        return await asyncio.to_thread(self._browse_directories, directory_id, path)

    def directory_path(self, directory_id: str) -> Path:
        path = self._directory_ids.get(directory_id)
        if path is None:
            raise ModelDirectoryNotFoundError(directory_id)
        try:
            path = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ModelDirectoryNotFoundError(directory_id) from error
        if not path.is_dir():
            raise ModelDirectoryNotFoundError(directory_id)
        return path

    async def register_directory(
        self,
        *,
        directory_id: str | None = None,
        path: str | Path | None = None,
    ) -> ModelArtifactList:
        """Register every MFQ container below one server-side directory."""

        if (directory_id is None) == (path is None):
            raise ModelRegistrationError("exactly one directory source is required")
        if directory_id is not None:
            selected = self._directory_ids.get(directory_id)
            if selected is None:
                raise ModelDirectoryNotFoundError(directory_id)
        else:
            try:
                selected = Path(path or "").expanduser().resolve(strict=True)
            except OSError as error:
                raise ModelDirectoryNotFoundError(str(path)) from error
        if not selected.is_dir():
            raise ModelRegistrationError("selected model path is not a directory")
        if not self.roots:
            raise ModelRegistrationError("the server has no writable model catalog")

        async with self._registration_lock:
            registration_paths = await asyncio.to_thread(
                self._registration_paths,
                selected,
            )
            if not registration_paths:
                raise ModelRegistrationError("selected directory contains no supported models")
            index_path = self.roots[0] / MODEL_FILE_INDEX
            previous = index_path.read_bytes() if index_path.is_file() else None
            await asyncio.to_thread(self._write_registered_paths, index_path, registration_paths)
            try:
                models = await self._snapshot(refresh=True)
            except Exception:
                await asyncio.to_thread(self._restore_index, index_path, previous)
                await self._snapshot(refresh=True)
                raise

        selected_models = [
            item.resource
            for item in models.values()
            if item.path.resolve().is_relative_to(selected)
        ]
        return ModelArtifactList(
            data=sorted(selected_models, key=lambda item: (item.name.casefold(), item.id))
        )

    def _directory_id(self, path: Path) -> str:
        resolved = path.resolve()
        identifier = hashlib.sha256(self._directory_secret + os.fsencode(resolved)).hexdigest()[:32]
        self._directory_ids[identifier] = resolved
        return identifier

    def _browse_roots(self) -> list[Path]:
        candidates = (
            [*self.roots, Path.home(), Path(Path.home().anchor or os.sep)]
            if self.browse_roots is None
            else list(self.browse_roots)
        )
        result: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve(strict=True)
            except OSError:
                continue
            if not resolved.is_dir() or resolved in seen:
                continue
            seen.add(resolved)
            result.append(resolved)
        return result

    @staticmethod
    def _directory_name(path: Path) -> str:
        return path.name or path.anchor or str(path)

    @staticmethod
    def _immediate_model_count(path: Path) -> int:
        try:
            mfq_models: set[Path] = set()
            tables: dict[Path, bool] = {}
            for item in path.iterdir():
                if not item.is_file() or not _is_model_path(item):
                    continue
                if _is_gguf_table(item, tables):
                    continue
                try:
                    parsed = parse_shard_path(item)
                except ValueError:
                    parsed = None
                mfq_models.add(parsed[0] if parsed else item)
            hf_count = int(ModelCatalog._is_hf_model_directory(path)) + sum(
                item.is_dir() and ModelCatalog._is_hf_model_directory(item)
                for item in path.iterdir()
            )
            return len(mfq_models) + hf_count
        except OSError:
            return 0

    def _directory_entry(self, path: Path) -> ModelDirectoryEntry:
        return ModelDirectoryEntry(
            id=self._directory_id(path),
            name=self._directory_name(path),
            model_file_count=self._immediate_model_count(path),
        )

    def _browse_directories(
        self,
        directory_id: str | None,
        path: str | Path | None,
    ) -> ModelDirectoryList:
        if directory_id is None and path is None:
            return ModelDirectoryList(
                data=[self._directory_entry(path) for path in self._browse_roots()]
            )
        if path is not None:
            try:
                current = Path(path).expanduser().resolve(strict=True)
            except OSError as error:
                raise ModelDirectoryNotFoundError(str(path)) from error
        else:
            current = self._directory_ids.get(directory_id or "")
        if current is None or not current.is_dir():
            raise ModelDirectoryNotFoundError(str(path) if path is not None else directory_id)
        try:
            entries = list(current.iterdir())
            children = sorted(
                (item.resolve() for item in entries if item.is_dir()),
                key=lambda item: (item.name.casefold(), item.name),
            )
        except OSError as error:
            raise ModelDirectoryNotFoundError(self._directory_name(current)) from error
        files = []
        for item in entries:
            with contextlib.suppress(OSError):
                if item.is_file() and _is_model_path(item):
                    files.append(ModelDirectoryFile(name=item.name, byte_size=item.stat().st_size))
        files.sort(key=lambda item: (item.name.casefold(), item.name))
        parent = current.parent if current.parent != current else None
        return ModelDirectoryList(
            current_id=self._directory_id(current),
            current_name=self._directory_name(current),
            current_path=str(current),
            parent_id=self._directory_id(parent) if parent is not None else None,
            model_file_count=self._immediate_model_count(current),
            files=files,
            can_open_directory=directory_opener() is not None,
            can_open_in_finder=sys.platform == "darwin",
            data=[self._directory_entry(path) for path in children],
        )

    @staticmethod
    def _registration_paths(directory: Path) -> list[Path]:
        grouped: dict[Path, list[Path]] = {}
        try:
            candidates = sorted(path for path in directory.rglob("*") if _is_model_path(path))
        except OSError as error:
            raise ModelRegistrationError("cannot scan selected model directory") from error
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                parsed = parse_shard_path(candidate)
            except ValueError as error:
                raise ModelRegistrationError(str(error)) from error
            grouped.setdefault(parsed[0] if parsed is not None else candidate, []).append(candidate)
        result: list[Path] = []
        for paths in grouped.values():
            result.append(
                next(
                    (
                        item
                        for item in paths
                        if (parsed := parse_shard_path(item)) is None or parsed[1] == 1
                    ),
                    paths[0],
                ).resolve()
            )
        hf_directories = {
            config.parent.resolve()
            for config in directory.rglob("config.json")
            if ModelCatalog._is_hf_model_directory(config.parent)
        }
        if ModelCatalog._is_hf_model_directory(directory):
            hf_directories.add(directory.resolve())
        result.extend(sorted(hf_directories))
        return result

    @staticmethod
    def _write_registered_paths(index: Path, paths: list[Path]) -> None:
        index.parent.mkdir(parents=True, exist_ok=True)
        document: dict[str, object] = {"version": 1, "files": []}
        if index.is_file():
            try:
                loaded = json.loads(index.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ModelRegistrationError("the model catalog index is invalid") from error
            if not isinstance(loaded, dict) or loaded.get("version") != 1:
                raise ModelRegistrationError("the model catalog index version is unsupported")
            document = loaded
        existing = document.get("files")
        files = (
            {value for value in existing if isinstance(value, str)}
            if isinstance(existing, list)
            else set()
        )
        files.update(str(path) for path in paths)
        payload = (
            json.dumps(
                {"version": 1, "files": sorted(files)},
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            + b"\n"
        )
        temporary = index.with_name(f".{index.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(payload)
            temporary.chmod(0o600)
            os.replace(temporary, index)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _restore_index(index: Path, previous: bytes | None) -> None:
        if previous is None:
            index.unlink(missing_ok=True)
            return
        temporary = index.with_name(f".{index.name}.{os.getpid()}.restore")
        try:
            temporary.write_bytes(previous)
            temporary.chmod(0o600)
            os.replace(temporary, index)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _canonical_model_path(path: Path) -> Path:
        try:
            parsed = parse_shard_path(path)
        except ValueError:
            parsed = None
        return (parsed[0] if parsed is not None else path).resolve()

    async def invalidate(self) -> None:
        while True:
            async with self._lock:
                task = self._refresh_task
                if task is None:
                    self._models.clear()
                    self._last_scan = float("-inf")
                    return
            with contextlib.suppress(Exception):
                await asyncio.shield(task)

    async def _snapshot(self, *, refresh: bool = False) -> dict[str, DiscoveredModel]:
        if refresh or self.cache_seconds <= 0:
            return await self._synchronous_snapshot()

        async with self._lock:
            has_snapshot = self._last_scan != float("-inf")
            if has_snapshot:
                if (
                    monotonic()
                    - max(self._last_scan, self._last_scan_attempt)
                    >= self.cache_seconds
                    and self._refresh_task is None
                ):
                    self._start_refresh_locked(background=True)
                # Model discovery may touch every shard on a slow external
                # volume.  Keep serving the last complete snapshot while one
                # shared refresh runs instead of blocking all readers on I/O.
                return dict(self._models)
            task = self._refresh_task
            if task is None:
                task = self._start_refresh_locked(background=False)
        return dict(await asyncio.shield(task))

    async def _synchronous_snapshot(self) -> dict[str, DiscoveredModel]:
        while True:
            async with self._lock:
                task = self._refresh_task
                background = self._refresh_is_background
                if task is None:
                    task = self._start_refresh_locked(background=False)
                    background = False
            if not background:
                return dict(await asyncio.shield(task))
            # A force refresh or registration must observe a scan that started
            # after any opportunistic stale-cache refresh it encountered.
            with contextlib.suppress(Exception):
                await asyncio.shield(task)

    def _start_refresh_locked(
        self,
        *,
        background: bool,
    ) -> asyncio.Task[dict[str, DiscoveredModel]]:
        if self._refresh_task is not None:
            raise RuntimeError("model catalog refresh already active")
        task = asyncio.create_task(
            self._refresh_snapshot(),
            name="mfq-model-catalog-refresh",
        )
        self._last_scan_attempt = monotonic()
        self._refresh_task = task
        self._refresh_is_background = background
        task.add_done_callback(self._observe_refresh)
        return task

    async def _refresh_snapshot(self) -> dict[str, DiscoveredModel]:
        current = asyncio.current_task()
        try:
            models = await asyncio.to_thread(self._scan)
        except BaseException:
            async with self._lock:
                if self._refresh_task is current:
                    self._refresh_task = None
                    self._refresh_is_background = False
            raise
        async with self._lock:
            if self._refresh_task is current:
                self._models = models
                self._last_scan = monotonic()
                self._refresh_task = None
                self._refresh_is_background = False
        return dict(models)

    @staticmethod
    def _observe_refresh(
        task: asyncio.Task[dict[str, DiscoveredModel]],
    ) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            _LOGGER.warning("model catalog refresh failed: %s", error)

    def _scan(self) -> dict[str, DiscoveredModel]:
        result: dict[str, DiscoveredModel] = {}
        model_paths: dict[str, Path] = {}
        models_by_name: dict[str, DiscoveredModel] = {}
        tables: dict[Path, bool] = {}
        components: dict[tuple[Path, str], dict[Path, TableComponent]] = {}
        mixed_components: set[tuple[Path, str]] = set()
        for root in self.roots:
            if not root.is_dir():
                continue
            grouped: dict[Path, list[Path]] = {}
            paths = {path for path in root.rglob("*") if _is_model_path(path)}
            paths.update(self._registered_paths(root))
            paths.update(
                config.parent
                for config in root.rglob("config.json")
                if self._is_hf_model_directory(config.parent)
            )
            if self._is_hf_model_directory(root):
                paths.add(root)
            for path in sorted(paths):
                component = self._table_component(path)
                if component is not None:
                    if component.mixed_kinds:
                        mixed_components.update((component_base(component.path), kind) for kind in component.mixed_kinds)
                    else:
                        components.setdefault((component_base(component.path), component.kind), {})[component.path] = component
                        continue
                if _is_gguf_table(path, tables):
                    continue
                try:
                    parsed = parse_shard_path(path)
                except ValueError:
                    parsed = None
                grouped.setdefault(parsed[0] if parsed is not None else path, []).append(path)
            for paths in grouped.values():
                path = paths[0]
                for item in paths:
                    try:
                        parsed = parse_shard_path(item)
                    except ValueError:
                        continue
                    if parsed is None or parsed[1] == 1:
                        path = item
                        break
                privacy_root = root if path.is_relative_to(root) else path.parent
                discovered = self._inspect_source(privacy_root, path)
                if discovered.resource.format == 'hf' and discovered.resource.complete:
                    config = json.loads((discovered.path / 'config.json').read_text(encoding='utf-8'))
                    for shard in discovered.file_paths:
                        component = self._table_component(shard, config)
                        if component is not None:
                            if component.mixed_kinds:
                                mixed_components.update((component_base(component.path), kind) for kind in component.mixed_kinds)
                            else:
                                components.setdefault((component_base(component.path), component.kind), {})[component.path] = component
                canonical_path = self._canonical_model_path(discovered.path)
                previous_path = model_paths.get(discovered.resource.name)
                if previous_path is not None and previous_path != canonical_path:
                    previous_named = models_by_name[discovered.resource.name]
                    formats = {
                        previous_named.resource.format,
                        discovered.resource.format,
                    }
                    if formats != {"hf", "tq"}:
                        raise DuplicateModelNameError(
                            f"duplicate catalog model name: {discovered.resource.name}"
                        )
                    # Converters commonly place ``name.tq`` beside their
                    # ``name/`` HF source.  They are two representations of
                    # one logical model, not an ambiguous pair for Studio's
                    # model picker.  Prefer the portable/loadable MFQ while
                    # retaining strict duplicate detection within a format.
                    if discovered.resource.format == "hf":
                        continue
                    result.pop(previous_named.resource.id, None)
                model_paths[discovered.resource.name] = canonical_path
                models_by_name[discovered.resource.name] = discovered
                previous = result.get(discovered.resource.id)
                if previous is None or str(path) < str(previous.path):
                    result[discovered.resource.id] = discovered
        table_files: set[Path] = set()
        for key, group in components.items():
            if key in mixed_components:
                continue
            parts = list(group.values())
            resource, files = component_resource(parts)
            table_files.update(files)
            result[resource.id] = DiscoveredModel(resource, parts[0].path, file_paths=files)
        for key, artifact in list(result.items()):
            if artifact.resource.asset_kind != 'model' or not artifact.file_paths:
                continue
            owned_bytes = sum(path.stat().st_size for path in set(artifact.file_paths) if path.resolve() not in table_files)
            result[key] = replace(artifact, resource=artifact.resource.model_copy(update={'owned_file_bytes': owned_bytes}))
        return result

    def _table_component(self, path: Path, config: dict | None = None) -> TableComponent | None:
        if not path.is_file():
            return None
        try:
            files = gguf_part_paths(path) if is_gguf_path(path) else (path,)
            first = files[0]
            signature = tuple((str(item), item.stat().st_size, item.stat().st_mtime_ns) for item in files)
            is_hf = path.suffix.casefold() == '.safetensors'
            if is_hf:
                source = path.parent / 'config.json'
                signature += ((str(source), source.stat().st_size, source.stat().st_mtime_ns),)
            cached = self._table_metadata.get(first)
            if cached is not None and cached[0] == signature:
                return cached[1]
            if is_hf:
                tensors = {name: value for name, value in self._safetensors_header(first).items() if name != '__metadata__'}
                kinds = set()
                for name in tensors:
                    mapped = map_source_tensor_name(name, config or {})
                    kinds.add(table_record_kind(mapped.canonical_name if mapped is not None else name))
                table_kinds = kinds - {None}
                pure = None not in kinds and len(kinds) == 1
                component = TableComponent(first, (first,), next(iter(table_kinds)) if pure else 'mixed',
                    (config or {}).get('model_type', 'unknown'), len(tensors), tuple(sorted({entry['dtype'] for entry in tensors.values()})),
                    tuple(sorted(table_kinds)) if not pure else ()) if table_kinds else None
            else:
                component = inspect_table_component(first, include_mixed=True)
            self._table_metadata[first] = (signature, component)
            return component
        except (OSError, ValueError, TypeError, struct.error, UnicodeError):
            return None

    @staticmethod
    def _registered_paths(root: Path) -> set[Path]:
        index = root / MODEL_FILE_INDEX
        try:
            document = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return set()
        if not isinstance(document, dict) or document.get("version") != 1:
            return set()
        values = document.get("files")
        if not isinstance(values, list):
            return set()
        result: set[Path] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            path = Path(value).expanduser()
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                continue
            if resolved.is_dir() and ModelCatalog._is_hf_model_directory(resolved):
                result.add(resolved)
                continue
            if not _is_model_path(resolved):
                continue
            if resolved.is_file():
                try:
                    parsed = parse_shard_path(resolved)
                except ValueError:
                    continue
                if parsed is None:
                    result.add(resolved)
                else:
                    result.update(matching_shard_paths(resolved))
        return result

    def _inspect_source(self, root: Path, path: Path) -> DiscoveredModel:
        if not is_gguf_path(path):
            artifact = self._inspect(root, path)
            if artifact.resource.format != 'hf' or not artifact.resource.complete or 'U32' not in artifact.resource.dtypes:
                return artifact
            try:
                sources = [path / 'config.json', *sorted(path.glob('*.safetensors'))]
                index = path / 'model.safetensors.index.json'
                if index.is_file():
                    sources.append(index)
                signature = tuple((str(source), source.stat().st_size, source.stat().st_mtime_ns) for source in sources)
                metadata = self._inspect_native_metadata(path, signature)
                config, tensors = metadata['config'], metadata['tensors']
                roles = {role: 0 for role in ('dense', 'experts', 'embedding')}
                streamed = 0
                for name, record in tensors.items():
                    size = int(record['nbytes'])
                    if _is_always_streamed_tensor(name):
                        streamed += size
                    else:
                        roles[weight_role(name)] += size
                resource = artifact.resource.model_copy(update={
                    'loadable': True, 'error': None,
                    'architecture': _native_architecture(metadata, artifact.resource.architecture),
                    'estimated_resident_weight_bytes': sum(roles.values()),
                    'estimated_weight_bytes_by_role': roles, 'ssd_ple_bytes': streamed,
                    'tensor_count': len(tensors), 'record_count': len(tensors),
                    'dtypes': sorted({item['dtype'] for item in tensors.values()}),
                })
                return replace(artifact, resource=resource, routed_expert_bytes=roles['experts'],
                    always_streamed_bytes=streamed, source_config=config, source_tensors=tensors)
            except Exception as failure:
                return replace(artifact, resource=artifact.resource.model_copy(update={
                    'loadable': False, 'error': str(failure).replace(str(root), '<model-root>'),
                }))
        parse_error = None
        try:
            parsed = parse_gguf_shard_path(path)
        except ValueError as failure:
            parsed, parse_error = None, failure
        paths = tuple(candidate for candidate in matching_shard_paths(path)
            if parse_shard_path(candidate)[2] == parsed[2]) if parsed else (path,)
        stats = tuple(candidate.stat() for candidate in paths)
        count = parsed[2] if parsed else 1
        missing = max(0, count - len(paths))
        name = (parsed[0] if parsed else path).stem
        signature = tuple((str(candidate), stat.st_size, stat.st_mtime_ns) for candidate, stat in zip(paths, stats, strict=True))
        identity = tuple((candidate.name, stat.st_size) for candidate, stat in zip(paths, stats, strict=True))
        identifier = hashlib.sha256(f'gguf\0{name}\0{identity}'.encode()).hexdigest()[:32]
        config: dict = {}
        tensors: dict = {}
        architecture = 'unknown'
        error = None
        loadable = False
        roles = {role: 0 for role in ('dense', 'experts', 'embedding')}
        experts = streamed = 0
        try:
            if parse_error is not None:
                raise parse_error
            if missing:
                raise ValueError(f'GGUF checkpoint is missing {missing} shard(s)')
            if gguf_is_table_component(path):
                raise ValueError('This GGUF is a shared table, not a standalone model')
            metadata = self._inspect_native_metadata(path, signature)
            config = metadata['config']
            tensors = metadata['tensors']
            architecture = _native_architecture(metadata)
            supported = {'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K',
                'Q8_K', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ3_XXS', 'IQ3_S', 'IQ4_NL', 'IQ4_XS', 'Q2_0',
                'PTQ1_0', 'TQ2_T', 'TQK4', 'TQK5', 'TQK6', 'TQK7', 'TQP5', 'TQP6', 'TQP7',
                'MXFP4', 'MIXED', 'ENGRAM_MXFP8'}
            unsupported = set()
            for tensor, record in tensors.items():
                dtype, size = record['dtype'], int(record['nbytes'])
                if dtype.startswith('GGUF_') and dtype[5:] not in supported:
                    unsupported.add(dtype)
                if _is_always_streamed_tensor(tensor):
                    streamed += size
                    continue
                role = weight_role(tensor)
                roles[role] += size
                if role == 'experts':
                    experts += size
            if unsupported:
                raise ValueError('Unsupported GGUF weight format: ' + ', '.join(sorted(unsupported)))
            loadable = True
        except Exception as failure:
            error = str(failure).replace(str(root), '<model-root>')
        capacity, yarn_capacity, yarn_factor = yarn_context_limits(config, architecture)
        text = config.get('text_config', config)
        companion_required = isinstance(text, dict) and any(
            f'model.block.{layer}.associative_memory.embedding.weight' not in tensors
            for layer in text.get('engram_layer_ids', []) or [])
        resource = ModelArtifactResource(id=identifier, name=name, architecture=architecture, format='gguf',
            gguf_companion_required=companion_required,
            shard_count=count, missing_shards=missing, total_bytes=sum(stat.st_size for stat in stats),
            estimated_resident_weight_bytes=sum(roles.values()) if tensors else None,
            estimated_weight_bytes_by_role=roles if tensors else {}, ssd_ple_bytes=streamed if tensors else None,
            context_capacity=capacity, yarn_context_capacity=yarn_capacity, yarn_max_factor=yarn_factor,
            tensor_count=len(tensors), record_count=len(tensors), dtypes=sorted({item['dtype'] for item in tensors.values()}),
            complete=not missing, loadable=loadable, error=error,
            modified_at=datetime.fromtimestamp(max((stat.st_mtime for stat in stats), default=path.stat().st_mtime), timezone.utc))
        return DiscoveredModel(resource, path, experts, streamed, config, tensors, paths)

    def _inspect_native_metadata(self, path: Path, signature: tuple) -> dict:
        if self._native_inspector is None:
            raise ValueError('Native model metadata inspection is unavailable')
        cached = self._source_metadata.get(path)
        if cached is None or cached[0] != signature:
            metadata = self._native_inspector(path)
            self._source_metadata[path] = (signature, metadata)
            return metadata
        return cached[1]

    @staticmethod
    def _inspect(root: Path, path: Path) -> DiscoveredModel:
        if path.is_dir():
            return ModelCatalog._inspect_hf(root, path)
        try:
            parsed = parse_shard_path(path)
        except ValueError:
            parsed = None
        name = (parsed[0] if parsed is not None else path).stem
        relative = path.relative_to(root).as_posix()
        file_paths: tuple[Path, ...] = ()
        try:
            with open_mmap(path) as store:
                paths = tuple(store.paths)
                file_paths = paths
                stats = tuple(item.stat() for item in paths)
                dtypes = sorted({record.dtype for record in store.records.values()})
                tensor_count = sum(
                    not is_asset_record(record.name) for record in store.records.values()
                )
                routed_expert_bytes = sum(
                    int(record.nbytes)
                    for record in store.records.values()
                    if not is_asset_record(record.name)
                    and _is_routed_expert_tensor(record.name)
                )
                always_streamed_bytes = sum(
                    int(record.nbytes)
                    for record in store.records.values()
                    if not is_asset_record(record.name)
                    and _is_always_streamed_tensor(record.name)
                )
                weight_bytes_by_dtype: dict[str, int] = {}
                by_role: dict[str, dict[str, int]] = {role: {} for role in ('dense', 'experts', 'embedding')}
                for record in store.records.values():
                    if is_asset_record(record.name) or _is_always_streamed_tensor(record.name):
                        continue
                    dtype = canonical_dtype(record.dtype)
                    weight_bytes_by_dtype[dtype] = weight_bytes_by_dtype.get(dtype, 0) + int(record.nbytes)
                    role_dtypes = by_role[weight_role(record.name)]
                    role_dtypes[dtype] = role_dtypes.get(dtype, 0) + int(record.nbytes)
                weight_bytes = sum(weight_bytes_by_dtype.values())
                def json_asset(asset: str) -> dict:
                    record = store.records.get(asset)
                    if record is None or record.nbytes > 4 * 1024 * 1024:
                        return {}
                    value = json.loads(store.read_blob(asset))
                    return value if isinstance(value, dict) else {}
                config = json_asset(MODEL_CONFIG_ASSET) or store.header.extra.get("hf_config", {})
                capacity = declared_context_size(config if isinstance(config, dict) else {}, json_asset(HF_TOKENIZER_CONFIG_ASSET))
                native, yarn_capacity, yarn_factor = yarn_context_limits(config if isinstance(config, dict) else {}, store.header.model_arch)
                capacity = native or capacity
                fingerprint = "\0".join(
                    [
                        store.header.model_arch,
                        name,
                        *(
                            f"{item.name}:{stat.st_size}"
                            for item, stat in zip(paths, stats, strict=True)
                        ),
                    ]
                )
                identifier = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
                resource = ModelArtifactResource(
                    id=identifier,
                    name=name,
                    architecture=store.header.model_arch or "unknown",
                    shard_count=len(paths),
                    total_bytes=sum(stat.st_size for stat in stats),
                    estimated_resident_weight_bytes=estimated_resident_weight_bytes(weight_bytes, weight_bytes_by_dtype),
                    estimated_weight_bytes_by_role=estimated_weight_roles(by_role),
                    ssd_ple_bytes=always_streamed_bytes,
                    context_capacity=capacity,
                    yarn_context_capacity=yarn_capacity,
                    yarn_max_factor=yarn_factor,
                    tensor_count=tensor_count,
                    record_count=len(store.records),
                    dtypes=dtypes,
                    complete=True,
                    loadable=True,
                    modified_at=datetime.fromtimestamp(
                        max(stat.st_mtime for stat in stats), timezone.utc
                    ),
                )
        except Exception as error:
            stat = path.stat()
            remaining = tuple(candidate for candidate in matching_shard_paths(parsed[0])
                if candidate.is_file() and parse_shard_path(candidate)[2] == parsed[2]) if parsed is not None else (path,)
            remaining_stats = tuple(item.stat() for item in remaining) or (stat,)
            file_paths = remaining
            identifier = hashlib.sha256(f"{relative}\0{stat.st_size}".encode()).hexdigest()[:32]
            resource = ModelArtifactResource(
                id=identifier,
                name=name,
                architecture="unknown",
                shard_count=parsed[2] if parsed is not None else 1,
                missing_shards=max(0, parsed[2] - len(remaining)) if parsed else 0,
                total_bytes=sum(item.st_size for item in remaining_stats),
                tensor_count=0,
                record_count=0,
                complete=False,
                loadable=False,
                modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                error=str(error).replace(str(root), "<model-root>"),
            )
        return DiscoveredModel(
            resource=resource,
            path=path,
            routed_expert_bytes=(
                routed_expert_bytes if resource.complete else 0
            ),
            always_streamed_bytes=(
                always_streamed_bytes if resource.complete else 0
            ),
            file_paths=file_paths,
        )

    @staticmethod
    def _is_hf_model_directory(path: Path) -> bool:
        if not (path / "config.json").is_file():
            return False
        if (path / "model.safetensors.index.json").is_file():
            return True
        try:
            return any(item.is_file() for item in path.glob("*.safetensors"))
        except OSError:
            return False

    @staticmethod
    def _tokenizer_config(path: Path) -> dict:
        try:
            source = path / "tokenizer_config.json"
            if source.stat().st_size > 4 * 1024 * 1024:
                return {}
            value = json.loads(source.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _safetensors_header(path: Path) -> dict[str, object]:
        with path.open("rb") as stream:
            raw_size = stream.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"truncated Safetensors header: {path.name}")
            size = struct.unpack("<Q", raw_size)[0]
            if size > 16 * 1024 * 1024 or size > path.stat().st_size - 8:
                raise ValueError(f"invalid Safetensors header length: {path.name}")
            raw_header = stream.read(size)
        if len(raw_header) != size:
            raise ValueError(f"truncated Safetensors header: {path.name}")
        header = json.loads(raw_header)
        if not isinstance(header, dict):
            raise ValueError(f"invalid Safetensors header: {path.name}")
        return header

    @staticmethod
    def _inspect_hf(root: Path, path: Path) -> DiscoveredModel:
        name = path.name
        relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else name
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            model_type = config.get("model_type") if isinstance(config, dict) else None
            if not isinstance(model_type, str) or not model_type:
                raise ValueError("HF config.json has no model_type")
            index_path = path / "model.safetensors.index.json"
            indexed_names: set[str] | None = None
            if index_path.is_file():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map") if isinstance(index, dict) else None
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError("invalid Safetensors weight_map")
                if not all(
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in weight_map.items()
                ):
                    raise ValueError("invalid Safetensors weight_map entry")
                indexed_names = set(weight_map)
                shard_paths = tuple(path / item for item in sorted(set(weight_map.values())))
            else:
                shard_paths = tuple(sorted(path.glob("*.safetensors")))
            if not shard_paths or any(not item.is_file() for item in shard_paths):
                raise ValueError("HF checkpoint is missing Safetensors shards")
            tensors: set[str] = set()
            tensor_sizes: dict[str, int] = {}
            tensor_dtypes: dict[str, str] = {}
            dtypes: set[str] = set()
            for shard in shard_paths:
                for tensor_name, entry in ModelCatalog._safetensors_header(shard).items():
                    if tensor_name == "__metadata__":
                        continue
                    if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str):
                        raise ValueError(f"invalid Safetensors tensor: {tensor_name}")
                    if tensor_name in tensors:
                        raise ValueError(f"duplicate Safetensors tensor: {tensor_name}")
                    offsets = entry.get("data_offsets")
                    if (
                        not isinstance(offsets, list)
                        or len(offsets) != 2
                        or not all(isinstance(value, int) for value in offsets)
                        or offsets[0] < 0
                        or offsets[1] < offsets[0]
                    ):
                        raise ValueError(
                            f"invalid Safetensors offsets: {tensor_name}"
                        )
                    tensors.add(tensor_name)
                    tensor_sizes[tensor_name] = offsets[1] - offsets[0]
                    tensor_dtypes[tensor_name] = entry["dtype"]
                    dtypes.add(entry["dtype"])
            if indexed_names is not None and not indexed_names.issubset(tensors):
                raise ValueError("Safetensors index references missing tensors")
            stats = tuple(item.stat() for item in shard_paths)
            fingerprint = "\0".join(
                [
                    model_type,
                    name,
                    *(
                        f"{item.name}:{stat.st_size}"
                        for item, stat in zip(shard_paths, stats, strict=True)
                    ),
                ]
            )
            identifier = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
            loadable = graph_spec_for_source_names(config, sorted(tensors)) is not None
            routed_expert_bytes = 0
            always_streamed_bytes = 0
            weight_bytes_by_dtype: dict[str, int] = {}
            by_role: dict[str, dict[str, int]] = {role: {} for role in ('dense', 'experts', 'embedding')}
            for tensor_name, tensor_bytes in tensor_sizes.items():
                mapped = map_source_tensor_name(tensor_name, config)
                canonical_name = (
                    mapped.canonical_name if mapped is not None else tensor_name
                )
                if _is_routed_expert_tensor(canonical_name):
                    routed_expert_bytes += tensor_bytes
                if _is_always_streamed_tensor(canonical_name):
                    always_streamed_bytes += tensor_bytes
                else:
                    dtype = canonical_dtype(tensor_dtypes[tensor_name])
                    weight_bytes_by_dtype[dtype] = weight_bytes_by_dtype.get(dtype, 0) + tensor_bytes
                    role_dtypes = by_role[weight_role(canonical_name)]
                    role_dtypes[dtype] = role_dtypes.get(dtype, 0) + tensor_bytes
            native, yarn_capacity, yarn_factor = yarn_context_limits(config, model_type)
            resource = ModelArtifactResource(
                id=identifier,
                name=name,
                architecture=f"{model_type}-hf-full-mfq",
                format="hf",
                shard_count=len(shard_paths),
                total_bytes=sum(stat.st_size for stat in stats),
                estimated_resident_weight_bytes=estimated_resident_weight_bytes(sum(weight_bytes_by_dtype.values()), weight_bytes_by_dtype),
                estimated_weight_bytes_by_role=estimated_weight_roles(by_role),
                ssd_ple_bytes=always_streamed_bytes,
                context_capacity=native or declared_context_size(config, ModelCatalog._tokenizer_config(path)),
                yarn_context_capacity=yarn_capacity,
                yarn_max_factor=yarn_factor,
                tensor_count=len(indexed_names if indexed_names is not None else tensors),
                record_count=len(indexed_names if indexed_names is not None else tensors) + 1,
                dtypes=sorted(dtypes),
                complete=True,
                loadable=loadable,
                modified_at=datetime.fromtimestamp(
                    max(stat.st_mtime for stat in stats), timezone.utc
                ),
                error=(
                    None
                    if loadable
                    else (
                        f"{model_type} has no registered canonical tensor schema"
                    )
                ),
            )
        except Exception as error:
            stat = path.stat()
            identifier = hashlib.sha256(f"{relative}\0{stat.st_mtime_ns}".encode()).hexdigest()[:32]
            resource = ModelArtifactResource(
                id=identifier,
                name=name,
                architecture="unknown",
                format="hf",
                shard_count=max(1, len(tuple(path.glob("*.safetensors")))),
                total_bytes=0,
                tensor_count=0,
                record_count=0,
                complete=False,
                loadable=False,
                modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                error=str(error).replace(str(root), "<model-root>"),
            )
        return DiscoveredModel(
            resource=resource,
            path=path,
            routed_expert_bytes=(
                routed_expert_bytes if resource.complete else 0
            ),
            always_streamed_bytes=(
                always_streamed_bytes if resource.complete else 0
            ),
            file_paths=shard_paths if resource.complete else (),
        )
