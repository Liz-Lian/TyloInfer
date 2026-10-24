"""Smoke tests ensuring the package imports, the version is readable, and the CLI parses."""

from __future__ import annotations

import tyloinfer
from tyloinfer import cli


def test_version_is_string():
    assert isinstance(tyloinfer.__version__, str)
    assert tyloinfer.__version__


def test_cli_help_exits_zero():
    import pytest

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_cli_no_subcommand_returns_zero():
    assert cli.main([]) == 0
