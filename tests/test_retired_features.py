import asyncio
import sys

import httpx

from tyloinfer.cli import _build_parser
from tyloinfer.server.api import create_app
from tyloinfer.server.services.tool_jobs import ToolJobHandlers, ToolJobPaths
from tyloinfer.server.state.catalog import ModelCatalog
from tyloinfer.server.state.storage import SessionStore


def test_removed_features_have_no_routes_or_internal_credential():
    async def run():
        app = create_app()
        assert not hasattr(app.state, "agent_token")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for path in (
                "/api/v1/agent/state",
                "/api/v1/agent/start",
                "/api/v1/evaluations",
                "/api/v1/evaluations/tools",
                "/api/v1/datasets",
            ):
                assert (await client.get(path)).status_code == 404
                assert (await client.post(path, json={})).status_code == 404
            assert (await client.get("/health")).status_code == 200
        assert not any(
            path.startswith(("/api/v1/agent/", "/api/v1/evaluations", "/api/v1/datasets"))
            for path in app.openapi()["paths"]
        )

    asyncio.run(run())


def test_download_jobs_remain_without_evaluation_handlers(tmp_path):
    from pathlib import Path

    handlers = ToolJobHandlers(
        ModelCatalog([]),
        ToolJobPaths(
            tmp_path,
            Path(sys.executable),
            None,
            None,
            None,
            internal_modelscope=True,
            internal_huggingface=True,
        ),
    ).handlers()
    assert set(handlers) == {"model.validate", "download.modelscope", "download.huggingface"}
    parser = _build_parser()
    assert "wt2-reference" not in parser.format_help()


def test_storage_keeps_sessions_without_creating_evaluation_tables(tmp_path):
    import sqlite3

    path = tmp_path / "sessions.sqlite3"
    SessionStore(path)
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "sessions" in tables
    assert not tables & {"datasets", "evaluation_results"}
