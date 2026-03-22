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
    assert "名称检索" in out and "mod.py" in out


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


def test_execute_create_directory_coerced_to_file_from_user_java_intent(tmp_path: Path) -> None:
    (tmp_path / "src" / "functest").mkdir(parents=True)
    parsed = {
        "tool_name": "create_directory",
        "tool_params": {"dir_path": "src/functest"},
        "reason": "t",
    }
    name, out = execute_tool_call(
        parsed,
        default_root_dir=str(tmp_path),
        user_instruction="在src/functest目录创建一个HelloWold.java",
    )
    assert name == "create_file"
    assert "工具校正" in out
    assert (tmp_path / "src" / "functest" / "HelloWold.java").is_file()
    assert parsed["tool_name"] == "create_file"


def test_execute_create_directory_coerced_inline_create_hello_world(tmp_path: Path) -> None:
    (tmp_path / "src" / "functest").mkdir(parents=True)
    parsed = {
        "tool_name": "create_directory",
        "tool_params": {"dir_path": "src/functest"},
        "reason": "t",
    }
    name, out = execute_tool_call(
        parsed,
        default_root_dir=str(tmp_path),
        user_instruction="在src/functest创建HelloWorld.java",
    )
    assert name == "create_file"
    assert "工具校正" in out
    assert (tmp_path / "src" / "functest" / "HelloWorld.java").is_file()
    assert parsed["tool_name"] == "create_file"


def test_execute_coerce_create_directory_to_create_file(tmp_path: Path) -> None:
    (tmp_path / "functest").mkdir()
    parsed = {
        "tool_name": "create_directory",
        "tool_params": {"dir_path": "functest/Hello.java"},
        "reason": "t",
    }
    name, out = execute_tool_call(
        parsed,
        default_root_dir=str(tmp_path),
        user_instruction="在functest目录下创建Hello.java",
    )
    assert name == "create_file"
    assert "工具校正" in out
    assert (tmp_path / "functest" / "Hello.java").is_file()
    assert parsed["tool_name"] == "create_file"


def test_execute_create_file_aligns_path_and_becomes_write_when_exists(tmp_path: Path) -> None:
    d = tmp_path / "src" / "functest"
    d.mkdir(parents=True)
    target = d / "HelloWold.java"
    target.write_text("// old\n", encoding="utf-8")
    parsed = {
        "tool_name": "create_file",
        "tool_params": {
            "file_path": "src/functest/HelloWorld.java",
            "content": "public class HelloWold {}",
        },
        "reason": "t",
    }
    name, out = execute_tool_call(
        parsed,
        default_root_dir=str(tmp_path),
        user_instruction="在src/functest/HelloWold.java中写快速排序",
    )
    assert name == "write_file"
    assert "路径校正" in out
    assert "工具校正" in out
    assert parsed["tool_name"] == "write_file"
    assert parsed["tool_params"]["file_path"] == "src/functest/HelloWold.java"
    assert target.read_text(encoding="utf-8") == "public class HelloWold {}"


def test_execute_delete_directory_corrects_path(tmp_path: Path) -> None:
    d = tmp_path / "src" / "functest"
    d.mkdir(parents=True)
    parsed = {
        "tool_name": "delete_directory",
        "tool_params": {"dir_path": "functest"},
        "reason": "t",
    }
    name, out = execute_tool_call(
        parsed,
        default_root_dir=str(tmp_path),
        user_instruction="在src目录删除functest目录",
    )
    assert name == "delete_directory"
    assert "路径校正" in out
    assert not d.exists()
    assert parsed["tool_params"]["dir_path"] == "src/functest"


def test_execute_create_directory_corrects_path_from_user_instruction(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    parsed = {
        "tool_name": "create_directory",
        "tool_params": {"dir_path": "functest"},
        "reason": "t",
    }
    name, out = execute_tool_call(
        parsed,
        default_root_dir=str(tmp_path),
        user_instruction="在src目录下创建一个functest目录",
    )
    assert name == "create_directory"
    assert "路径校正" in out
    assert (tmp_path / "src" / "functest").is_dir()
    assert parsed["tool_params"]["dir_path"] == "src/functest"


def test_execute_create_directory_and_write_file(tmp_path: Path) -> None:
    parsed = {
        "tool_name": "create_directory",
        "tool_params": {"dir_path": "sub/nested"},
        "reason": "t",
    }
    name, out = execute_tool_call(parsed, default_root_dir=str(tmp_path))
    assert name == "create_directory"
    assert "成功" in out
    assert (tmp_path / "sub" / "nested").is_dir()

    parsed2 = {
        "tool_name": "write_file",
        "tool_params": {"file_path": "sub/nested/hello.txt", "content": "hi"},
        "reason": "t",
    }
    name2, out2 = execute_tool_call(parsed2, default_root_dir=str(tmp_path))
    assert name2 == "write_file"
    assert "成功" in out2
    assert (tmp_path / "sub" / "nested" / "hello.txt").read_text(encoding="utf-8") == "hi"


def test_execute_append_delete_rename(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    ap = {
        "tool_name": "append_file",
        "tool_params": {"file_path": "a.txt", "content": "y"},
        "reason": "t",
    }
    n1, o1 = execute_tool_call(ap, default_root_dir=str(tmp_path))
    assert n1 == "append_file" and "成功" in o1
    assert f.read_text(encoding="utf-8") == "xy"

    rp = {
        "tool_name": "rename_file",
        "tool_params": {"old_path": "a.txt", "new_path": "b.txt"},
        "reason": "t",
    }
    n2, o2 = execute_tool_call(rp, default_root_dir=str(tmp_path))
    assert n2 == "rename_file" and "成功" in o2
    assert not f.is_file()
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "xy"

    dp = {
        "tool_name": "delete_file",
        "tool_params": {"file_path": "b.txt"},
        "reason": "t",
    }
    n3, o3 = execute_tool_call(dp, default_root_dir=str(tmp_path))
    assert n3 == "delete_file" and "成功" in o3
    assert not (tmp_path / "b.txt").exists()
