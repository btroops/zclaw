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
)
def test_extract_json_object_with_fence() -> None:
    text = 'Here:\n```json\n{"tool_name": null, "tool_params": {}, "reason": "x"}\n```'
    raw = extract_json_object(text)
    assert json.loads(raw)["tool_name"] is None


def test_parse_tool_call_extra_text() -> None:
    text = 'prefix {"tool_name": "get_project_directory", "tool_params": {"root_dir": "/tmp"}, "reason": "r"} suffix'
    p = parse_tool_call(text)
    assert p["tool_name"] == "get_project_directory"


def test_build_tool_call_prompt_format() -> None:
    s = build_tool_call_prompt(
        default_root_dir="/abs/proj",
        user_instruction="hello",
    )
    assert "/abs/proj" in s
    assert "hello" in s
    assert "get_project_directory" in s


def test_resolve_path(tmp_path: Path) -> None:
    root = str(tmp_path)
    assert resolve_path("sub", root) == str((tmp_path / "sub").resolve())
    assert resolve_path(str(tmp_path / "sub"), root) == str((tmp_path / "sub").resolve())


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


def test_execute_no_tool() -> None:
    name, out = execute_tool_call(
        {"tool_name": None, "tool_params": {}, "reason": "n"},
        default_root_dir="/tmp",
    )
    assert name is None
    assert "未调用工具" in out
