"""Tests for workspace path resolution (directory chain + file)."""

from __future__ import annotations

from pathlib import Path

import pytest

from zclaw.path_resolve import resolve_target_directory, resolve_target_file


def test_resolve_multilevel_file_after_directory_chain(tmp_path: Path) -> None:
    (tmp_path / "src" / "zclaw").mkdir(parents=True)
    f = tmp_path / "src" / "zclaw" / "llm.py"
    f.write_text("ok", encoding="utf-8")
    p, note = resolve_target_file(str(tmp_path), "zclaw/llm.py")
    assert p == str(f.resolve())
    assert "ok" not in note


def test_resolve_directory_multisegment(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    d = tmp_path / "a" / "b"
    p, _ = resolve_target_directory(str(tmp_path), "a/b")
    assert p == str(d.resolve())


def test_resolve_file_tools_basename(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    f = tmp_path / "pkg" / "tools.py"
    f.write_text("x", encoding="utf-8")
    p, _ = resolve_target_file(str(tmp_path), "tools.py")
    assert p == str(f.resolve())
