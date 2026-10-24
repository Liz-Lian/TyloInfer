from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tyloinfer.formats.header import FileHeader
from tyloinfer.formats.io import save
from tyloinfer.server.protocol.models import CreateJobRequest, JobStatus
from tyloinfer.server.services.jobs import JobManager
from tyloinfer.server.services.tool_jobs import (
    ToolJobHandlers,
    ToolJobPaths,
)
from tyloinfer.server.state.catalog import ModelCatalog
from tyloinfer.server.state.storage import SessionStore


@pytest.fixture(autouse=True)
def _python_script_executables_on_windows(tmp_path, monkeypatch):
    if os.name != 'nt':
        return
    spawn = asyncio.create_subprocess_exec

    async def run(*argv, **kwargs):
        path = Path(argv[0])
        # Windows does not interpret the Python shebangs of our fake tools.
        # Keep argv as a list so the shell-metacharacter tests remain valid.
        if (path.is_relative_to(tmp_path) and path.is_file() and
                path.read_bytes().startswith(b'#!/usr/bin/env python')):
            argv = (sys.executable, *argv)
        return await spawn(*argv, **kwargs)

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', run)




def test_download_reports_real_written_bytes_and_speed(tmp_path, monkeypatch):
    async def run():
        handlers = ToolJobHandlers(ModelCatalog([]), ToolJobPaths(tmp_path, tmp_path / 'tq', None, None, None, None))
        captured = []
        class Context:
            async def progress(self, value, **kwargs): captured.append((value, kwargs['data']))
        async def transfer(*args, **kwargs):
            await asyncio.sleep(0.1)
            (tmp_path / 'shard.tq').write_bytes(b'x' * 1024)
            (tmp_path / 'part.incomplete').write_bytes(b'x' * 1024)
            (tmp_path / 'ignored.lock').write_bytes(b'x' * 1024)
            await asyncio.sleep(1.05)
        monkeypatch.setattr(handlers, '_run', transfer)
        await handlers._run_download(Context(), ['fake-download'], {}, tmp_path, 4096)
        assert captured[-1][0] == 0.5
        data = captured[-1][1]
        assert data['downloaded_bytes'] == 2048 and data['files_completed'] == 1
        assert data['bytes_per_second'] > 0 and data['total_bytes'] == 4096
    asyncio.run(run())


def test_standalone_cli_jobs_reinvoke_the_unified_mfq_binary(tmp_path: Path) -> None:
    catalog = ModelCatalog([tmp_path])
    executable = tmp_path / "tq"
    handlers = ToolJobHandlers(
        catalog,
        ToolJobPaths(
            tmp_path,
            executable,
            None,
            None,
            None,
            None,
            standalone_cli=True,
        ),
    )

    assert handlers._mfq_command("wt2-reference", "input", "output") == [
        str(executable),
        "wt2-reference",
        "input",
        "output",
    ]


def _executable(path: Path, source: str) -> Path:
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _model(path: Path) -> None:
    save(
        path,
        FileHeader(version=2, model_arch="test"),
        {"weight": np.ones((2, 2), dtype=np.float16)},
    )


def test_tool_process_retains_only_a_bounded_output_tail(
    tmp_path: Path, monkeypatch
) -> None:
    import tyloinfer.server.services.tool_jobs as tool_jobs

    class Context:
        def __init__(self) -> None:
            self.cleanup = []
            self.logs: list[str] = []

        def add_cleanup(self, callback) -> None:
            self.cleanup.append(callback)

        def raise_if_cancelled(self) -> None:
            return None

        async def log(self, message: str) -> None:
            self.logs.append(message)

        async def progress(self, _value: float, *, message: str) -> None:
            return None

    async def run() -> None:
        executable = _executable(
            tmp_path / "verbose-tool",
            """\
            #!/usr/bin/env python3
            for index in range(6):
                print(f'line-{index}')
            """,
        )
        handlers = ToolJobHandlers(
            ModelCatalog([]),
            ToolJobPaths(tmp_path, executable, None, None, None, None),
        )
        context = Context()
        lines = await handlers._run(context, [str(executable)])
        assert lines == ["line-3", "line-4", "line-5"]
        assert context.logs == [f"line-{index}" for index in range(6)]
        for callback in context.cleanup:
            await callback()

    monkeypatch.setattr(tool_jobs, "_TOOL_OUTPUT_TAIL_LINES", 3)
    asyncio.run(run())


@pytest.mark.parametrize('system', ['nt', 'posix'])
@pytest.mark.parametrize('force_kill', [False, True])
def test_cancelled_tool_stops_process_and_releases_download_locks(tmp_path, monkeypatch, system, force_kill):
    from types import SimpleNamespace
    import tyloinfer.server.services.tool_jobs as tool_jobs
    from tyloinfer.server.services.jobs import JobCancelledError

    async def run():
        calls = []
        class Process:
            pid = 123
            returncode = None
            def __init__(self):
                self.stdout = asyncio.StreamReader()
                self.stdout.feed_data(b'downloading\n')
            def terminate(self): calls.append('terminate')
            def kill(self): calls.append('kill')
            async def wait(self):
                if force_kill and len(calls) == 1:
                    raise asyncio.TimeoutError()
                self.returncode = 0
                return 0
        process = Process()
        async def spawn(*args, **kwargs): return process
        def killpg(pid, signum):
            assert system == 'posix' and pid == process.pid
            calls.append('terminate' if signum == tool_jobs.signal.SIGTERM else 'kill')
        monkeypatch.setattr(tool_jobs, 'os', SimpleNamespace(name=system, killpg=killpg))
        monkeypatch.setattr(tool_jobs, 'signal', SimpleNamespace(SIGTERM=15, SIGKILL=9))
        monkeypatch.setattr(tool_jobs.asyncio, 'create_subprocess_exec', spawn)
        class Context:
            def __init__(self): self.cleanup = []
            def add_cleanup(self, callback): self.cleanup.append(callback)
            def raise_if_cancelled(self): raise JobCancelledError()
        handlers = ToolJobHandlers(ModelCatalog([]), ToolJobPaths(tmp_path, tmp_path / 'tq', None, None, None, None))
        context = Context()
        with pytest.raises(JobCancelledError):
            await handlers._run(context, ['fake-download'])
        # Child-held disk locks must be gone before the enclosing download
        # handler releases its reservation, before final JobContext cleanup.
        assert process.returncode == 0
        assert calls == (['terminate', 'kill'] if force_kill else ['terminate'])
        for callback in context.cleanup:
            await callback()
        assert calls == (['terminate', 'kill'] if force_kill else ['terminate'])
        assert process.returncode == 0
    asyncio.run(run())


async def _wait(store: SessionStore, job_id) -> object:
    for _ in range(300):
        job = store.get_job(job_id)
        if job.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return job
        await asyncio.sleep(0.01)
    return store.get_job(job_id)


def test_container_validation_and_workspace_path_boundary(tmp_path: Path) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "tiny.tq")
        runtime = _executable(
            tmp_path / "runtime",
            """\
            #!/usr/bin/env python3
            import sys
            print('MFQ container OK version=2 architecture=test shards=1 records=1')
            """,
        )
        diagnostics = _executable(
            tmp_path / "mfq-diagnostics",
            """\
            #!/usr/bin/env python3
            import sys
            assert '--check-mfq-container' in sys.argv
            print('MFQ container OK via diagnostics')
            """,
        )
        python = _executable(
            tmp_path / "python",
            """\
            #!/usr/bin/env python3
            import pathlib, sys
            assert '--staged-blobs' in sys.argv
            output = pathlib.Path(sys.argv[5])
            output.write_bytes(b'MFQ1')
            print('100% done')
            """,
        )
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        handlers = ToolJobHandlers(
            catalog,
            ToolJobPaths(
                work_root=tmp_path,
                python=python,
                modelscope=None,
                huggingface=None,
                runtime=runtime,
                perplexity=None,
                diagnostics=diagnostics,
            ),
        )
        store = SessionStore(tmp_path / "jobs.sqlite3")
        manager = JobManager(store, handlers.handlers())
        kinds = {item.kind: item for item in manager.kinds().data}
        assert not {'model.quantize', 'calibrate.imatrix', 'artifact.import'} & kinds.keys()
        artifact = (await catalog.list()).data[0]

        checked = await manager.submit(
            CreateJobRequest(kind="model.validate", payload={"model": artifact.name})
        )
        result = await _wait(store, checked.id)
        assert result.status == JobStatus.SUCCEEDED
        assert result.result["summary"] == "MFQ container OK via diagnostics"

        with pytest.raises(Exception, match='workspace-relative'):
            handlers._input(str(tmp_path))
        await manager.close()

    asyncio.run(run())









def test_download_argv_does_not_execute_shell_metacharacters(tmp_path: Path) -> None:
    async def run() -> None:
        downloader = _executable(
            tmp_path / "modelscope",
            """\
            #!/usr/bin/env python3
            import pathlib, sys
            root = pathlib.Path(sys.argv[sys.argv.index('--local-dir') + 1])
            root.mkdir(parents=True, exist_ok=True)
            (root / 'weight.tq').write_bytes(b'MFQ1')
            print('downloaded')
            """,
        )
        handlers = ToolJobHandlers(
            ModelCatalog([]),
            ToolJobPaths(
                work_root=tmp_path,
                python=Path(__import__("sys").executable),
                modelscope=downloader,
                huggingface=None,
                runtime=None,
                perplexity=None,
            ),
        )
        store = SessionStore(tmp_path / "jobs.sqlite3")
        manager = JobManager(store, handlers.handlers())
        submitted = await manager.submit(
            CreateJobRequest(
                kind="download.modelscope",
                payload={
                    "repo_id": "owner/model",
                    "destination": "downloads/value;touch-owned",
                    "expected_bytes": 4,
                },
            )
        )
        result = await _wait(store, submitted.id)
        assert result.status == JobStatus.SUCCEEDED
        assert not (tmp_path / "owned").exists()
        assert result.result["files"] == 1
        lineage = store.list_artifact_lineage()[0]
        assert lineage.source_uris == ["modelscope://owner/model@master"]
        removed = handlers.remove_workspace_artifact(result.result["artifact"])
        assert removed == {"files": 1, "total_bytes": 4}
        assert not (tmp_path / "downloads" / "value;touch-owned").exists()
        await manager.close()

    asyncio.run(run())


def test_packaged_download_reinvokes_the_unified_cli(tmp_path: Path) -> None:
    async def run() -> None:
        cli = _executable(
            tmp_path / "tyloinfer-cli",
            """\
            #!/usr/bin/env python3
            import pathlib, sys
            assert sys.argv[1:6] == [
                '_hub-download', '--provider', 'modelscope', '--repo-id', 'owner/model'
            ]
            root = pathlib.Path(sys.argv[sys.argv.index('--local-dir') + 1])
            root.mkdir(parents=True, exist_ok=True)
            (root / 'config.json').write_text('{}')
            """,
        )
        model_root = tmp_path / "managed" / "models"
        (tmp_path / "work").mkdir()
        handlers = ToolJobHandlers(
            ModelCatalog([model_root]),
            ToolJobPaths(
                work_root=tmp_path / "work",
                python=cli,
                modelscope=None,
                huggingface=None,
                runtime=None,
                perplexity=None,
                standalone_cli=True,
                internal_modelscope=True,
            ),
        )
        store = SessionStore(tmp_path / "jobs.sqlite3")
        manager = JobManager(store, handlers.handlers())
        assert "download.modelscope" in {item.kind for item in manager.kinds().data}
        submitted = await manager.submit(
            CreateJobRequest(
                kind="download.modelscope",
                payload={"repo_id": "owner/model", "destination": "models/owner/model", "expected_bytes": 2},
            )
        )
        result = await _wait(store, submitted.id)
        assert result.status == JobStatus.SUCCEEDED
        assert result.result["files"] == 1
        assert result.result["artifact"] == "workspace://models/owner/model"
        assert (model_root / "owner" / "model" / "config.json").is_file()
        await manager.close()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ("modelscope", "huggingface"))
@pytest.mark.parametrize("internal", (False, True))
@pytest.mark.parametrize("proxy", (False, True))
def test_download_passes_selected_route_to_sdk_and_external_cli(tmp_path, monkeypatch, provider, internal, proxy) -> None:
    import os
    import tyloinfer.server.services.tool_jobs as tool_jobs

    async def select(endpoint, *, direct=False):
        assert endpoint == ("https://modelscope.cn" if provider == "modelscope" else "https://huggingface.co")
        env = dict(os.environ)
        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            env.pop(key, None)
        env["NO_PROXY"] = "localhost" if proxy else "*"
        env["TYLOINFER_EXPECT_PROXY"] = str(proxy)
        if proxy:
            env["HTTPS_PROXY"] = "http://proxy.test:8080"
        return env, proxy

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(tool_jobs, "download_environment", select)

    async def run():
        executable = _executable(tmp_path / "download-cli", """\
            #!/usr/bin/env python3
            import os, pathlib, sys
            proxy = os.environ['TYLOINFER_EXPECT_PROXY'] == 'True'
            assert os.environ.get('HTTPS_PROXY') == ('http://proxy.test:8080' if proxy else None)
            assert os.environ['NO_PROXY'] == ('localhost' if proxy else '*')
            root = pathlib.Path(sys.argv[sys.argv.index('--local-dir') + 1])
            root.mkdir(parents=True, exist_ok=True)
            (root / 'config.json').write_text('{}')
            """)
        paths = ToolJobPaths(tmp_path, executable, executable, executable, None, None,
            standalone_cli=internal, internal_modelscope=internal, internal_huggingface=internal)
        handlers = ToolJobHandlers(ModelCatalog([]), paths)
        store = SessionStore(tmp_path / "jobs.sqlite3")
        manager = JobManager(store, handlers.handlers())
        submitted = await manager.submit(CreateJobRequest(kind=f"download.{provider}",
            payload={"repo_id": "owner/model", "destination": "downloads/model", "expected_bytes": 2}))
        result = await _wait(store, submitted.id)
        assert result.status == JobStatus.SUCCEEDED and result.result["files"] == 1
        await manager.close()

    asyncio.run(run())


def test_perplexity_job_parses_result_and_publishes_logits(tmp_path: Path) -> None:
    async def run() -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _model(model_dir / "tiny.tq")
        (tmp_path / "dataset.txt").write_text("evaluation text", encoding="utf-8")
        perplexity = _executable(
            tmp_path / "perplexity",
            """\
            #!/usr/bin/env python3
            import pathlib, sys
            if '--logits-file' in sys.argv:
                path = pathlib.Path(sys.argv[sys.argv.index('--logits-file') + 1])
                path.write_bytes(b'logits')
            print('[1]12.3000,')
            print('Final estimate: PPL = 12.3000 +/- 0.12000 (255 scored tokens)')
            """,
        )
        catalog = ModelCatalog([model_dir], cache_seconds=0)
        handlers = ToolJobHandlers(
            catalog,
            ToolJobPaths(
                work_root=tmp_path,
                python=Path(__import__("sys").executable),
                modelscope=None,
                huggingface=None,
                runtime=None,
                perplexity=perplexity,
            ),
        )
        store = SessionStore(tmp_path / "jobs.sqlite3")
        assert "evaluate.perplexity" not in handlers.handlers()
        manager = JobManager(store, {"evaluate.perplexity": handlers.perplexity})
        artifact = (await catalog.list()).data[0]
        submitted = await manager.submit(
            CreateJobRequest(
                kind="evaluate.perplexity",
                payload={
                    "model": artifact.name,
                    "dataset_file": "dataset.txt",
                    "dataset": "fixture",
                    "chunks": 1,
                    "logits_file": "results/reference.logits",
                },
            )
        )
        result = await _wait(store, submitted.id)
        assert result.status == JobStatus.SUCCEEDED
        assert result.result["perplexity"] == 12.3
        assert result.result["uncertainty"] == 0.12
        assert result.result["model_id"] == artifact.name
        assert result.result["artifact_id"] == artifact.id
        assert result.result["logits"] == "workspace://results/reference.logits"
        evaluation = store.list_evaluations()[0]
        assert str(evaluation.id) == result.result["evaluation_id"]
        assert evaluation.kind == "perplexity"
        assert evaluation.metrics["perplexity"] == 12.3
        assert evaluation.dataset_manifest["byte_size"] == 15
        assert evaluation.hardware_identity["machine"]
        assert evaluation.runtime_identity["executable"] == "perplexity"
        assert evaluation.model_id == artifact.name
        assert evaluation.runtime_identity["artifact_id"] == artifact.id
        events = store.list_job_events(submitted.id)
        assert any(event.type.value == "artifact" for event in events)
        await manager.close()

    asyncio.run(run())
