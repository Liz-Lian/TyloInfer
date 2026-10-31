from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from tyloinfer.server.protocol.models import HubModelSummary
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
