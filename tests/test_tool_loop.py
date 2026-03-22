"""Tests for JSON parsing and tool dispatch (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zclaw.tool_loop import (
    build_tool_call_prompt,
    execute_tool_call,
    extract_json_object,
    parse_tool_call,
    resolve_path,
    resolve_path_with_meta,
)


def test_extract_json_object_with_fence() -> None:
    text = 'Here:\n```json\n{"tool_name": null, "tool_params": {}, "reason": "x"}\n```'
    raw = extract_json_object(text)
    assert json.loads(raw)["tool_name"] is None


def test_parse_tool_call_extra_text() -> None:
    text = 'prefix {"tool_name": "get_project_directory", "tool_params": {"root_dir": "/tmp"}, "reason": "r"} suffix'
    p = parse_tool_call(text)
    assert p["tool_name"] == "get_project_directory"


def test_parse_tool_call_json_then_markdown_with_braces() -> None:
    """Model sometimes appends fake code after JSON; must not use rfind('}')."""
    text = """{
  "tool_name": "get_file_content",
  "tool_params": {"file_path": "a.py"},
  "reason": "x"
}
### 生成回复
```python
def foo():
    return {"nested": 1}
```
"""
    p = parse_tool_call(text)
    assert p["tool_name"] == "get_file_content"
    assert p["tool_params"]["file_path"] == "a.py"


def test_build_tool_call_prompt_format() -> None:
    s = build_tool_call_prompt(
        default_root_dir="/abs/proj",
        user_instruction="hello",
    )
    assert "/abs/proj" in s
    assert "hello" in s
    assert "get_project_directory" in s
    assert "相对默认工程根" in s


def test_resolve_path(tmp_path: Path) -> None:
    root = str(tmp_path)
    assert resolve_path("sub", root) == str((tmp_path / "sub").resolve())
    assert resolve_path(str(tmp_path / "sub"), root) == str((tmp_path / "sub").resolve())


def test_resolve_path_prefers_workspace_over_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "only_here.txt").write_text("a", encoding="utf-8")
    p, fb = resolve_path_with_meta("only_here.txt", str(ws))
    assert fb is False
    assert p == str((ws / "only_here.txt").resolve())


def test_resolve_path_fallback_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "proj"
    ws.mkdir()
    (tmp_path / "only_cwd.txt").write_text("b", encoding="utf-8")
    p, fb = resolve_path_with_meta("only_cwd.txt", str(ws))
    assert fb is True
    assert p == str((tmp_path / "only_cwd.txt").resolve())


def test_execute_tool_directory_relative_subdir(tmp_path: Path) -> None:
    llm = tmp_path / "llm"
    llm.mkdir()
    (llm / "a.txt").write_text("1", encoding="utf-8")
    parsed = {
        "tool_name": "get_project_directory",
        "tool_params": {"root_dir": "llm", "max_depth": 2},
        "reason": "t",
    }
    name, out = execute_tool_call(parsed, default_root_dir=str(tmp_path))
    assert name == "get_project_directory"
    assert "a.txt" in out


def test_execute_tool_directory(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("1", encoding="utf-8")
    parsed = {
        "tool_name": "get_project_directory",
        "tool_params": {"root_dir": str(tmp_path), "max_depth": 2},
        "reason": "t",
    }
    name, out = execute_tool_call(parsed, default_root_dir=str(tmp_path))
    assert name == "get_project_directory"
    assert "x.txt" in out


def test_execute_directory_name_discovery_casefold(tmp_path: Path) -> None:
    (tmp_path / "src" / "zclaw").mkdir(parents=True)
    (tmp_path / "src" / "zclaw" / "mod.py").write_text("x", encoding="utf-8")
    parsed = {
        "tool_name": "get_project_directory",
        "tool_params": {"root_dir": "ZClaw", "max_depth": 3},
        "reason": "t",
    }
    name, out = execute_tool_call(parsed, default_root_dir=str(tmp_path))
    assert name == "get_project_directory"
    assert "默认工程根" in out and "mod.py" in out


def test_execute_file_basename_discovery(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir(parents=True)
    (tmp_path / "pkg" / "tools.py").write_text("toolbody", encoding="utf-8")
    parsed = {
        "tool_name": "get_file_content",
        "tool_params": {"file_path": "tools.py"},
        "reason": "",
    }
    name, out = execute_tool_call(parsed, default_root_dir=str(tmp_path))
    assert name == "get_file_content"
    assert "toolbody" in out
    assert "文件名" in out


def test_execute_no_tool() -> None:
    name, out = execute_tool_call(
        {"tool_name": None, "tool_params": {}, "reason": "n"},
        default_root_dir="/tmp",
    )
    assert name is None
    assert "未调用工具" in out
