"""Tests for append_file structural guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from zclaw.append_guard import append_content_rejection_reason
from zclaw.tool_loop import execute_tool_call


def test_no_file_allows_anything(tmp_path: Path) -> None:
    """目标不存在时由 append 创建，不拦截。"""
    p = tmp_path / "new.java"
    assert append_content_rejection_reason(str(p), "public class X {}\n") is None


def test_rejects_java_second_public_class(tmp_path: Path) -> None:
    f = tmp_path / "HelloWorld.java"
    f.write_text("public class HelloWorld {}\n", encoding="utf-8")
    r = append_content_rejection_reason(str(f), "public class Fibonacci {\n}\n")
    assert r is not None
    assert "顶层" in r or "声明" in r or "write_file" in r


def test_rejects_java_second_main(tmp_path: Path) -> None:
    f = tmp_path / "HelloWorld.java"
    f.write_text("public class HelloWorld {}\n", encoding="utf-8")
    r = append_content_rejection_reason(
        str(f),
        "    public static void main(String[] args) {}\n",
    )
    assert r is not None
    assert "入口" in r or "main" in r.lower()


def test_allows_java_method_after_closed_class(tmp_path: Path) -> None:
    """闭合类后追加普通方法仍会命中「尾部 }」——但不应是顶层 class；方法片段应放行。"""
    f = tmp_path / "HelloWorld.java"
    f.write_text("public class HelloWorld {}\n", encoding="utf-8")
    assert (
        append_content_rejection_reason(
            str(f),
            "    public static int fib(int n) { return n < 2 ? n : fib(n-1)+fib(n-2); }\n",
        )
        is None
    )


def test_rejects_json_second_root(tmp_path: Path) -> None:
    f = tmp_path / "a.json"
    f.write_text('{"a":1}\n', encoding="utf-8")
    r = append_content_rejection_reason(str(f), '{"b":2}')
    assert r is not None
    assert "JSON" in r


def test_rejects_html_second_document(tmp_path: Path) -> None:
    f = tmp_path / "a.html"
    f.write_text("<!DOCTYPE html><html></html>\n", encoding="utf-8")
    r = append_content_rejection_reason(str(f), "<!DOCTYPE html><html>")
    assert r is not None
    assert "html" in r.lower()


def test_execute_append_blocked_preserves_file(tmp_path: Path) -> None:
    java = tmp_path / "HelloWorld.java"
    java.write_text("public class HelloWorld {}\n", encoding="utf-8")
    parsed = {
        "tool_name": "append_file",
        "tool_params": {
            "file_path": "HelloWorld.java",
            "content": 'public class X { public static void main(String[] a) {} }\n',
        },
        "reason": "t",
    }
    name, out = execute_tool_call(parsed, default_root_dir=str(tmp_path))
    assert name == "append_file"
    assert "错误" in out
    assert "write_file" in out
    assert java.read_text(encoding="utf-8") == "public class HelloWorld {}\n"
