"""Tests for local tools (no LLM)."""

from __future__ import annotations

from pathlib import Path

from zclaw.tools import get_file_content, get_project_directory


def test_get_project_directory_smoke(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "b.py").write_text("y", encoding="utf-8")

    out = get_project_directory(
        str(tmp_path),
        exclude_dirs=[".git", "venv", "__pycache__"],
        max_depth=3,
        max_lines=200,
    )
    assert "工程目录：" in out
    assert "a.txt" in out
    assert "pkg/" in out or "pkg" in out


def test_get_file_content_utf8(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("你好\nline2", encoding="utf-8")
    assert get_file_content(str(p)) == "你好\nline2"


def test_get_file_content_missing() -> None:
    assert "错误" in get_file_content("/nonexistent/path/that/does/not/exist.txt")


def test_get_project_directory_bad_path() -> None:
    assert "错误" in get_project_directory("/nonexistent/dir/zclaw_test_12345")
