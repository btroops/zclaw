#!/usr/bin/env python3
"""
不依赖 pytest：用标准库 unittest 跑本地测试。

用法（在项目根）::

    PYTHONPATH=src python tests/run_unit_tests.py

或在项目根::

    PYTHONPATH=src python -m unittest discover -s tests -p 'run_unit_tests.py'
"""

from __future__ import annotations

import unittest

# --- unittest 用例（与 test_tools.py / test_tool_loop.py 逻辑对齐）---

import json
import tempfile
from pathlib import Path

from zclaw.tool_loop import (
    build_tool_call_prompt,
    execute_tool_call,
    extract_json_object,
    parse_tool_call,
    resolve_path,
)
from zclaw.tools import get_file_content, get_project_directory


class TestTools(unittest.TestCase):
    def test_get_project_directory_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "a.txt").write_text("x", encoding="utf-8")
            sub = tmp / "pkg"
            sub.mkdir()
            (sub / "b.py").write_text("y", encoding="utf-8")
            out = get_project_directory(
                str(tmp),
                exclude_dirs=[".git", "venv", "__pycache__"],
                max_depth=3,
                max_lines=200,
            )
            self.assertIn("工程目录：", out)
            self.assertIn("a.txt", out)

    def test_get_file_content_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.txt"
            p.write_text("你好\nline2", encoding="utf-8")
            self.assertEqual(get_file_content(str(p)), "你好\nline2")

    def test_get_file_content_missing(self) -> None:
        self.assertIn("错误", get_file_content("/nonexistent/path/that/does/not/exist.txt"))


class TestToolLoop(unittest.TestCase):
    def test_extract_json_object_with_fence(self) -> None:
        text = 'Here:\n```json\n{"tool_name": null, "tool_params": {}, "reason": "x"}\n```'
        raw = extract_json_object(text)
        self.assertIsNone(json.loads(raw)["tool_name"])

    def test_parse_tool_call_extra_text(self) -> None:
        text = 'prefix {"tool_name": "get_project_directory", "tool_params": {"root_dir": "/tmp"}, "reason": "r"} suffix'
        p = parse_tool_call(text)
        self.assertEqual(p["tool_name"], "get_project_directory")

    def test_build_tool_call_prompt_format(self) -> None:
        s = build_tool_call_prompt(
            default_root_dir="/abs/proj",
            user_instruction="hello",
        )
        self.assertIn("/abs/proj", s)
        self.assertIn("hello", s)
        self.assertIn("相对默认工程根", s)

    def test_resolve_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(resolve_path("sub", str(root)), str((root / "sub").resolve()))

    def test_execute_tool_directory_relative_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            llm = tmp / "llm"
            llm.mkdir()
            (llm / "a.txt").write_text("1", encoding="utf-8")
            parsed = {
                "tool_name": "get_project_directory",
                "tool_params": {"root_dir": "llm", "max_depth": 2},
                "reason": "t",
            }
            name, out = execute_tool_call(parsed, default_root_dir=str(tmp))
            self.assertEqual(name, "get_project_directory")
            self.assertIn("a.txt", out)

    def test_execute_tool_directory(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "x.txt").write_text("1", encoding="utf-8")
            parsed = {
                "tool_name": "get_project_directory",
                "tool_params": {"root_dir": str(tmp), "max_depth": 2},
                "reason": "t",
            }
            name, out = execute_tool_call(parsed, default_root_dir=str(tmp))
            self.assertEqual(name, "get_project_directory")
            self.assertIn("x.txt", out)

    def test_execute_no_tool(self) -> None:
        name, out = execute_tool_call(
            {"tool_name": None, "tool_params": {}, "reason": "n"},
            default_root_dir="/tmp",
        )
        self.assertIsNone(name)
        self.assertIn("未调用工具", out)


if __name__ == "__main__":
    unittest.main()
