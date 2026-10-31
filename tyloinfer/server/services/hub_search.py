from __future__ import annotations

import asyncio
import re
import secrets
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from tyloinfer.server.protocol.models import HubModelSearchResult, HubModelSummary


class HubSearchCursorError(ValueError):
    pass


@dataclass(frozen=True)
class HubSearchQuery:
    provider: str
    query: str
    formats: tuple[str, ...]
    sort: str
    direction: str
    limit: int


@dataclass
class _Stream:
    format: str
    position: str | None = None
    started: bool = False
    exhausted: bool = False
    buffer: list[HubModelSummary] = field(default_factory=list)

    def copy(self):
        return _Stream(self.format, self.position, self.started, self.exhausted, self.buffer.copy())


@dataclass
class _Search:
    query: HubSearchQuery
    streams: list[_Stream]
    seen: set[str] = field(default_factory=set)

    def copy(self):
        return _Search(self.query, [stream.copy() for stream in self.streams], self.seen.copy())


@dataclass
class _Cursor:
    query: HubSearchQuery
    state: _Search | None
    expires: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    result: HubModelSearchResult | None = None


def _words(value: str) -> list[str]:
    return re.findall(r"[^\W\d_]+|\d+", unicodedata.normalize("NFKC", value).casefold())


def search_order(model: HubModelSummary, query: HubSearchQuery) -> tuple:
    if query.sort == "downloads":
        return (model.downloads, model.repo_id.casefold())
    if query.sort == "likes":
        return (model.likes, model.repo_id.casefold())
    if query.sort == "updated":
        return (model.updated_at.timestamp() if model.updated_at else 0, model.repo_id.casefold())
    if not query.query:
        return (model.trending_score, model.downloads, model.repo_id.casefold())
    name = model.repo_id.rsplit('/', 1)[-1]
    words, requested = _words(name), _words(query.query)
    joined, term = ''.join(words), ''.join(requested)
    base = re.sub(r'[-_.](?:gguf|mlx|tq)(?:[-_].*)?$', '', name, flags=re.IGNORECASE)
    tier = (6 if joined == term else 5 if ''.join(_words(base)) == term else 4 if joined.startswith(term) else
        3 if all(word in words for word in requested) else 2 if term in joined else
        1 if all(word in _words(model.repo_id) for word in requested) else 0)
    return (tier, sum(word in words for word in requested), model.downloads, -len(name), model.repo_id.casefold())


class HubBrowserSearch:
    def __init__(self, fetch: Callable[[HubSearchQuery, str, str | None], Awaitable[tuple[list[HubModelSummary], str | None]]]):
        self._fetch = fetch
        self._cursors: OrderedDict[str, _Cursor] = OrderedDict()

    def _prune(self):
        now = time.monotonic()
        for token in [token for token, entry in self._cursors.items() if entry.expires < now]:
            del self._cursors[token]
        while len(self._cursors) > 128:
            self._cursors.popitem(last=False)

    async def _fill(self, state: _Search, stream: _Stream):
        while not stream.buffer and not stream.exhausted:
            values, position = await self._fetch(state.query, stream.format, stream.position)
            if stream.started and position is not None and position == stream.position:
                raise RuntimeError("Model source returned a repeated page cursor.")
            stream.started = True
            stream.position = position
            stream.exhausted = position is None
            stream.buffer = [item for item in values if item.repo_id not in state.seen]
            if state.query.sort == 'relevance':
                stream.buffer.sort(key=lambda item: search_order(item, state.query), reverse=state.query.direction == 'desc')

    async def _page(self, state: _Search) -> HubModelSearchResult:
        result = []
        async with asyncio.timeout(45):
            while len(result) < state.query.limit:
                await asyncio.gather(*(self._fill(state, stream) for stream in state.streams))
                available = [stream for stream in state.streams if stream.buffer]
                if not available:
                    break
                choose = max if state.query.direction == 'desc' else min
                stream = choose(available, key=lambda value: search_order(value.buffer[0], state.query))
                item = stream.buffer.pop(0)
                if item.repo_id not in state.seen:
                    state.seen.add(item.repo_id)
                    result.append(item)
        next_cursor = None
        if any(stream.buffer or not stream.exhausted for stream in state.streams):
            next_cursor = secrets.token_urlsafe(24)
            self._cursors[next_cursor] = _Cursor(state.query, state, time.monotonic() + 900)
            self._prune()
        return HubModelSearchResult(data=result, next_cursor=next_cursor)

    async def search(self, query: HubSearchQuery, cursor: str | None = None) -> HubModelSearchResult:
        self._prune()
        if cursor is None:
            return await self._page(_Search(query, [_Stream(format) for format in query.formats]))
        entry = self._cursors.get(cursor)
        if entry is None or entry.query != query:
            raise HubSearchCursorError("The search has expired or its filters changed. Start a new search.")
        async with entry.lock:
            if entry.result is None:
                if entry.state is None:
                    raise HubSearchCursorError("The search has expired. Start a new search.")
                entry.result = await self._page(entry.state.copy())
                entry.state = None
            if cursor in self._cursors:
                self._cursors.move_to_end(cursor)
            return entry.result.model_copy(deep=True)
