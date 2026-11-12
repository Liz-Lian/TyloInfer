"""Bind offline references to one explicit and verifiable MFQ source checkout."""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
import tempfile
import types
import uuid
from pathlib import Path

_MODULES = (
    'mfq.formats.nepq',
    'mfq.formats.npq0_s',
    'mfq.formats.npq0_l',
    'mfq.formats.nint8_zero',
    'mfq.formats.mxfp4_sq',
    'mfq.formats.fp8_sq',
    'mfq.formats.io',
    'mfq.formats.header',
    'mfq.formats.shards',
    'mfq.formats.mfe',
    'mfq.formats.nint',
    'mfq.formats.nvq',
    'mfq.formats.nvq1_l',
    'mfq.formats.nvq1_s',
    'mfq.quantize.nvq1_l_quant',
    'mfq.quantize.nvq1_s_quant',
    'mfq.quantize.nvq_jsc',
    'mfq.quantize.fp8_sq',
)
_SOURCE_ROOT = None
_SOURCE_FILES = None


def file_hash(path):
    """Return the SHA-256 identity of one source or artifact file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fixture(root, relative):
    """Load a standalone reference generator directly from verified source bytes.

    This entry is for explicit external fixture tools, never test modules. It
    bypasses bytecode caches and does not run their command-line entry points.
    """
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if (
        not path.is_relative_to(root)
        or path.suffix != '.py'
        or 'tests' in path.relative_to(root).parts
    ):
        raise ValueError('Fixture source must be a Python tool inside the selected checkout')
    source = path.read_bytes()
    module = types.ModuleType('mfq_numeric_fixture')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module, {path.relative_to(root).as_posix(): hashlib.sha256(source).hexdigest()}


def _loaded_files(root):
    """Reject foreign MFQ modules and describe every loaded MFQ source file."""
    files = {}
    for name, module in tuple(sys.modules.items()):
        if name != 'mfq' and not name.startswith('mfq.'):
            continue
        path = getattr(module, '__file__', None)
        if path is None:
            raise ValueError(f'Loaded MFQ module has no verifiable source: {name}')
        path = Path(path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Loaded MFQ module is outside the source checkout: {name}')
        files[path.relative_to(root).as_posix()] = file_hash(path)
    return dict(sorted(files.items()))


def _checkout(root):
    """Identify the requested Git checkout without lazy fetching or changing it."""
    env = os.environ.copy()
    env['GIT_NO_LAZY_FETCH'] = '1'
    env['GIT_OPTIONAL_LOCKS'] = '0'

    def git(*args):
        """Read one local Git value without allowing a network fetch."""
        return subprocess.check_output(['git', '-C', str(root), *args], env=env, text=True).strip()

    top = Path(git('rev-parse', '--show-toplevel')).resolve()
    if top != root:
        raise ValueError('MFQ source root must be the Git checkout root')
    return git('rev-parse', '--verify', 'HEAD')


def open_source(source_root):
    """Import one explicit MFQ checkout and bind it to its immutable module hashes."""
    global _SOURCE_ROOT, _SOURCE_FILES
    root = Path(source_root).resolve(strict=True)
    existing = _loaded_files(root)
    if _SOURCE_ROOT is None and existing:
        raise ValueError('MFQ modules were already loaded outside this verified adapter')
    if _SOURCE_ROOT is not None and (root != _SOURCE_ROOT or existing != _SOURCE_FILES):
        raise ValueError('Loaded MFQ checkout or its source files have changed')
    commit = _checkout(root)
    old_bytecode = sys.dont_write_bytecode
    old_pycache_prefix = sys.pycache_prefix
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(Path(tempfile.gettempdir()) / f'mfq-reference-{uuid.uuid4().hex}')
    sys.path.insert(0, str(root))
    try:
        modules = {name: importlib.import_module(name) for name in _MODULES}
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = old_bytecode
        sys.pycache_prefix = old_pycache_prefix
    files = _loaded_files(root)
    if _SOURCE_FILES is not None and files != _SOURCE_FILES:
        raise ValueError('MFQ module closure changed after its first import')
    _SOURCE_ROOT, _SOURCE_FILES = root, files
    return modules, {'commit': commit, 'files_sha256': files}
