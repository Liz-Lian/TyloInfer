from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tyloinfer.server.protocol.models import HubModelSummary
from tyloinfer.server.services import hub_search
from tyloinfer.server.services.hub_search import (
    HubBrowserSearch,
    HubSearchCursorError,
    HubSearchQuery,
)


def query(**kwargs):
    return replace(HubSearchQuery('huggingface', 'model', ('gguf',), 'downloads', 'desc', 40), **kwargs)


def model(index, format='gguf', **kwargs):
    return HubModelSummary(provider='huggingface', repo_id=f'owner/{format}-{index:04}',
        formats=[format], downloads=10000 - index, **kwargs)


def test_continuations_pass_one_hundred_results_and_retry_is_idempotent():
    calls = []

    async def fetch(config, format, position):
        calls.append((format, position))
        start = int(position or '0')
        end = min(135, start + 50)
        values = [model(index * 2 + (format == 'mlx'), format) for index in range(start, end)]
        return values, str(end) if end < 135 else None

    async def run():
        browser = HubBrowserSearch(fetch)
        config = query(formats=('gguf', 'mlx'))
        first = await browser.search(config)
        second = await browser.search(config, first.next_cursor)
        before_retry = len(calls)
        repeated = await browser.search(config, first.next_cursor)
        assert repeated == second
        assert len(calls) == before_retry
        values = first.data + second.data
        cursor = second.next_cursor
        while cursor:
            page = await browser.search(config, cursor)
            values.extend(page.data)
            cursor = page.next_cursor
        assert len(values) == 270
        assert len({item.repo_id for item in values}) == 270
        assert [item.downloads for item in values] == sorted((item.downloads for item in values), reverse=True)
        assert ('gguf', '100') in calls and ('mlx', '100') in calls

    asyncio.run(run())


def test_relevance_prioritizes_name_matches_over_downloads():
    async def fetch(config, format, position):
        return [
            model(0).model_copy(update={'repo_id': 'other/Unrelated-Qwen3.5-9B-adapter', 'downloads': 10000000}),
            model(1).model_copy(update={'repo_id': 'owner/Qwen3.5-9B-GGUF', 'downloads': 1}),
            model(2).model_copy(update={'repo_id': 'owner/Qwen3.5-9B', 'downloads': 0}),
        ], None

    result = asyncio.run(HubBrowserSearch(fetch).search(query(query='qwen3.5 9b', sort='relevance')))
    assert [item.repo_id for item in result.data] == ['owner/Qwen3.5-9B', 'owner/Qwen3.5-9B-GGUF', 'other/Unrelated-Qwen3.5-9B-adapter']


def test_recommended_uses_live_trending_scores():
    async def fetch(config, format, position):
        return [model(0, trending_score=1), model(1, trending_score=100)], None

    result = asyncio.run(HubBrowserSearch(fetch).search(query(query='', sort='relevance')))
    assert result.data[0].trending_score == 100


def test_duplicate_repositories_across_formats_only_appear_once():
    async def fetch(config, format, position):
        return [model(0).model_copy(update={'repo_id': 'owner/common'}), model(1, format)], None

    result = asyncio.run(HubBrowserSearch(fetch).search(query(formats=('gguf', 'mlx'))))
    assert len(result.data) == 3
    assert [item.repo_id for item in result.data].count('owner/common') == 1


def test_failed_continuation_does_not_lose_buffered_items():
    fail = True

    async def fetch(config, format, position):
        if position is None:
            return [model(index) for index in range(4)], 'second'
        if fail:
            raise OSError('offline')
        return [model(index) for index in range(4, 8)], None

    async def run():
        nonlocal fail
        browser = HubBrowserSearch(fetch)
        config = query(limit=3)
        first = await browser.search(config)
        with pytest.raises(OSError, match='offline'):
            await browser.search(config, first.next_cursor)
        fail = False
        second = await browser.search(config, first.next_cursor)
        assert [item.repo_id for item in second.data] == [model(index).repo_id for index in range(3, 6)]

    asyncio.run(run())


def test_cursor_is_bound_to_filters_and_expires():
    async def fetch(config, format, position):
        return [model(index) for index in range(50)], 'next'

    async def run():
        browser = HubBrowserSearch(fetch)
        config = query()
        first = await browser.search(config)
        for changed in (replace(config, query='changed'), replace(config, direction='asc'), replace(config, formats=('mlx',))):
            with pytest.raises(HubSearchCursorError):
                await browser.search(changed, first.next_cursor)
        browser._cursors[first.next_cursor].expires = 0
        with pytest.raises(HubSearchCursorError):
            await browser.search(config, first.next_cursor)

    asyncio.run(run())


def test_cursor_cache_has_a_fixed_entry_bound():
    async def fetch(config, format, position):
        return [model(index) for index in range(50)], 'next'

    async def run():
        browser = HubBrowserSearch(fetch)
        for index in range(140):
            await browser.search(query(query=str(index)))
        assert len(browser._cursors) == 128

    asyncio.run(run())


def test_search_supports_python_without_asyncio_timeout(monkeypatch):
    """Search and continuation work when Python 3.11's timeout API is unavailable."""
    monkeypatch.delattr(asyncio, 'timeout', raising=False)

    async def fetch(config, format, position):
        """Return enough results to exercise the continuation state."""
        return [model(index) for index in range(3)], None

    async def run():
        """Read both pages without relying on asyncio.timeout."""
        browser = HubBrowserSearch(fetch)
        config = query(limit=2)
        first = await browser.search(config)
        second = await browser.search(config, first.next_cursor)
        assert [item.repo_id for item in first.data + second.data] == [model(index).repo_id for index in range(3)]
        assert second.next_cursor is None

    asyncio.run(run())


def test_page_deadline_spans_multiple_upstream_fetches(monkeypatch):
    """Multiple individually fast upstream pages share one overall deadline."""
    monkeypatch.setattr(hub_search, '_PAGE_TIMEOUT', 0.2)
    calls = []
    cancelled = []

    async def fetch(config, format, position):
        """Spend less than the page budget on each individual upstream fetch."""
        calls.append(position)
        try:
            await asyncio.sleep(0.12)
        except asyncio.CancelledError:
            cancelled.append(position)
            raise
        index = int(position or '0')
        return [model(index)], str(index + 1)

    async def run():
        """Reject an incomplete page rather than publishing its partial cursor."""
        browser = HubBrowserSearch(fetch)
        with pytest.raises(TimeoutError):
            await browser.search(query(limit=3))
        assert calls == [None, '1']
        assert cancelled == ['1']
        assert not browser._cursors

    asyncio.run(run())


def test_timed_out_continuation_cancels_fetch_and_preserves_retry_state(monkeypatch):
    """A timed-out continuation leaves its buffered state available for a retry."""
    blocked = True
    cancelled = False

    async def run():
        """Force the real wait_for cancellation path after the next fetch starts."""
        nonlocal blocked, cancelled
        started = asyncio.Event()
        wait_for = asyncio.wait_for

        async def fetch(config, format, position):
            """Block the continuation fetch until its page is cancelled."""
            nonlocal cancelled
            if position is None:
                return [model(index) for index in range(4)], 'second'
            if blocked:
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled = True
            return [model(index) for index in range(4, 8)], None

        async def expire_started_page(awaitable, timeout):
            """Expire the page deterministically once its upstream fetch is suspended."""
            assert timeout == 45
            task = asyncio.create_task(awaitable)
            await started.wait()
            return await wait_for(task, timeout=0)

        browser = HubBrowserSearch(fetch)
        config = query(limit=3)
        first = await browser.search(config)
        token = first.next_cursor
        original = browser._cursors[token].state.copy()
        tokens = list(browser._cursors)
        monkeypatch.setattr(asyncio, 'wait_for', expire_started_page)
        with pytest.raises(TimeoutError):
            await browser.search(config, token)
        assert cancelled
        assert list(browser._cursors) == tokens
        assert browser._cursors[token].state == original
        assert browser._cursors[token].result is None

        blocked = False
        monkeypatch.setattr(asyncio, 'wait_for', wait_for)
        second = await browser.search(config, token)
        assert [item.repo_id for item in second.data] == [model(index).repo_id for index in range(3, 6)]
        assert await browser.search(config, token) == second

    asyncio.run(run())


def test_external_cancellation_propagates_and_cancels_all_fetches():
    """Cancelling a search waits for its upstream fetches to stop and publishes no cursor."""
    async def run():
        """Cancel only after both format streams have started their upstream fetch."""
        started = {format: asyncio.Event() for format in ('gguf', 'mlx')}
        cancelled = set()

        async def fetch(config, format, position):
            """Record cancellation of each independently suspended format stream."""
            started[format].set()
            try:
                await asyncio.Future()
            finally:
                cancelled.add(format)

        browser = HubBrowserSearch(fetch)
        task = asyncio.create_task(browser.search(query(formats=('gguf', 'mlx'))))
        await asyncio.gather(*(event.wait() for event in started.values()))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled == {'gguf', 'mlx'}
        assert not browser._cursors

    asyncio.run(run())


def test_legacy_asyncio_timeout_preserves_builtin_timeout_error(monkeypatch):
    """Python 3.10's distinct asyncio timeout class preserves callers' public error type."""
    class LegacyTimeoutError(Exception):
        """Represent the separate asyncio.TimeoutError class used before Python 3.11."""

    async def wait_for(awaitable, timeout):
        """Raise the legacy timeout without leaving an unawaited page coroutine."""
        awaitable.close()
        raise LegacyTimeoutError('page deadline')

    async def fetch(config, format, position):
        """Provide a valid fetch callback that the expired page never executes."""
        return [], None

    monkeypatch.setattr(hub_search, 'asyncio', SimpleNamespace(
        wait_for=wait_for, TimeoutError=LegacyTimeoutError, gather=asyncio.gather,
    ))
    with pytest.raises(TimeoutError, match='page deadline') as error:
        asyncio.run(HubBrowserSearch(fetch).search(query()))
    assert isinstance(error.value.__cause__, LegacyTimeoutError)
